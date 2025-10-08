"""
PVGIS Integration Module for Solar Energy Calculations

This module integrates with the PVGIS (Photovoltaic Geographical Information System) API
to calculate solar energy generation for specific buildings and determine optimal
solar cell configurations.
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SolarCellSpecs:
    """Specifications for solar cell technology."""
    technology: str
    efficiency: float  # Percentage
    power_per_m2: float  # Watts per square meter
    cost_per_m2: float  # Cost in EUR per square meter
    lifespan_years: int
    degradation_rate: float  # Annual degradation percentage


@dataclass
class PVGISResult:
    """Results from PVGIS calculation."""
    building_id: str
    latitude: float
    longitude: float
    annual_energy_kwh: float
    monthly_energy: Dict[str, float]
    optimal_tilt: float
    optimal_azimuth: float
    peak_power_kw: float
    performance_ratio: float
    solar_irradiance_kwh_m2: float


class PVGISClient:
    """Client for interacting with PVGIS API."""
    
    BASE_URL = "https://re.jrc.ec.europa.eu/api/v5_2/"
    
    # Solar cell technology specifications
    SOLAR_TECHNOLOGIES = {
        "mono_crystalline": SolarCellSpecs(
            technology="Crystalline silicon",
            efficiency=22.0,
            power_per_m2=220,  # W/m²
            cost_per_m2=180,   # EUR/m²
            lifespan_years=25,
            degradation_rate=0.5
        ),
        "poly_crystalline": SolarCellSpecs(
            technology="Crystalline silicon",
            efficiency=18.0,
            power_per_m2=180,  # W/m²
            cost_per_m2=140,   # EUR/m²
            lifespan_years=25,
            degradation_rate=0.5
        ),
        "thin_film": SolarCellSpecs(
            technology="Thin film",
            efficiency=12.0,
            power_per_m2=120,  # W/m²
            cost_per_m2=100,   # EUR/m²
            lifespan_years=20,
            degradation_rate=0.8
        ),
        "perovskite": SolarCellSpecs(
            technology="Perovskite",
            efficiency=25.0,
            power_per_m2=250,  # W/m²
            cost_per_m2=200,   # EUR/m²
            lifespan_years=15,
            degradation_rate=1.0
        )
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TorinoEnergy/1.0 (Research Project)'
        })
    
    def calculate_solar_energy(self, 
                             latitude: float, 
                             longitude: float,
                             peak_power: float = 1.0,  # kW
                             system_loss: float = 14.0,  # Percentage
                             mounting_type: str = "building",
                             tilt: float = 35.0,
                             azimuth: float = 0.0,
                             technology: str = "crystSi") -> Dict:
        """
        Calculate solar energy using PVGIS API.
        
        Args:
            latitude: Building latitude
            longitude: Building longitude
            peak_power: System peak power in kW
            system_loss: System losses in percentage
            mounting_type: Type of mounting (building, free-standing)
            tilt: Tilt angle in degrees
            azimuth: Azimuth angle in degrees (0 = South)
            technology: PV technology type
            
        Returns:
            Dictionary with PVGIS results
        """
        
        params = {
            'lat': latitude,
            'lon': longitude,
            'peakpower': peak_power,
            'system_loss': system_loss,
            'mountingplace': mounting_type,
            'angle': tilt,
            'aspect': azimuth,
            'pvtechchoice': technology,
            'outputformat': 'json'
        }
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}PVcalc",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"PVGIS calculation completed for lat={latitude}, lon={longitude}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"PVGIS API error: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"PVGIS response parsing error: {e}")
            raise
    
    def get_optimal_angles(self, latitude: float, longitude: float) -> Tuple[float, float]:
        """
        Get optimal tilt and azimuth angles for a location.
        
        Args:
            latitude: Building latitude
            longitude: Building longitude
            
        Returns:
            Tuple of (optimal_tilt, optimal_azimuth)
        """
        # Optimal tilt is typically latitude ± 10-15 degrees
        optimal_tilt = max(0, min(90, latitude + 10))
        
        # Optimal azimuth is 0 (South) for Northern Hemisphere
        optimal_azimuth = 0.0
        
        return optimal_tilt, optimal_azimuth


class BuildingSolarAnalyzer:
    """Analyzer for building-specific solar energy calculations."""
    
    def __init__(self):
        self.pvgis_client = PVGISClient()
        self.selected_building = None
    
    def select_building(self, buildings_df: pd.DataFrame, building_id: str = None, 
                       index: int = None) -> pd.Series:
        """
        Select a building for analysis.
        
        Args:
            buildings_df: DataFrame with building data
            building_id: OSM ID of the building to select
            index: Index of the building to select
            
        Returns:
            Selected building data as Series
        """
        if building_id:
            mask = buildings_df['OSM ID'] == building_id
            if not mask.any():
                raise ValueError(f"Building with ID {building_id} not found")
            self.selected_building = buildings_df[mask].iloc[0]
        elif index is not None:
            if index >= len(buildings_df):
                raise ValueError(f"Index {index} out of range")
            self.selected_building = buildings_df.iloc[index]
        else:
            # Select the building with highest roof area
            max_roof_idx = buildings_df['Roof Area (m²)'].idxmax()
            self.selected_building = buildings_df.loc[max_roof_idx]
        
        logger.info(f"Selected building: {self.selected_building['OSM ID']}")
        return self.selected_building
    
    def calculate_optimal_system(self, 
                               roof_area: float = None,
                               target_energy_kwh: float = None,
                               technology: str = "mono_crystalline") -> Dict:
        """
        Calculate optimal solar system configuration for the selected building.
        
        Args:
            roof_area: Available roof area (uses building's roof area if None)
            target_energy_kwh: Target annual energy production
            technology: Solar cell technology to use
            
        Returns:
            Dictionary with system configuration and results
        """
        if self.selected_building is None:
            raise ValueError("No building selected. Use select_building() first.")
        
        building = self.selected_building
        lat = building['Latitude']
        lon = building['Longitude']
        
        # Use building's roof area if not specified
        if roof_area is None:
            roof_area = building['Roof Area (m²)']
        
        # Get solar cell specifications
        cell_specs = self.pvgis_client.SOLAR_TECHNOLOGIES[technology]
        
        # Calculate optimal angles
        optimal_tilt, optimal_azimuth = self.pvgis_client.get_optimal_angles(lat, lon)
        
        # Calculate system capacity based on roof area
        max_power_kw = (roof_area * cell_specs.power_per_m2) / 1000
        
        # If target energy is specified, adjust system size
        if target_energy_kwh:
            # Estimate system size needed for target energy
            # This is a rough calculation - actual PVGIS call will refine it
            estimated_irradiance = 1400  # kWh/m²/year (rough estimate for Torino)
            estimated_efficiency = cell_specs.efficiency / 100
            estimated_system_loss = 0.14
            
            required_power_kw = target_energy_kwh / (
                estimated_irradiance * estimated_efficiency * (1 - estimated_system_loss)
            )
            max_power_kw = min(max_power_kw, required_power_kw)
        
        # Calculate with PVGIS
        pvgis_result = self.pvgis_client.calculate_solar_energy(
            latitude=lat,
            longitude=lon,
            peak_power=max_power_kw,
            tilt=optimal_tilt,
            azimuth=optimal_azimuth
        )
        
        # Extract results
        annual_energy = pvgis_result.get('outputs', {}).get('totals', {}).get('E_y', 0)
        monthly_data = pvgis_result.get('outputs', {}).get('monthly', {})
        
        # Calculate system metrics
        system_config = {
            'building_id': building['OSM ID'],
            'building_address': building.get('Address', 'N/A'),
            'roof_area_m2': roof_area,
            'technology': technology,
            'cell_specifications': {
                'efficiency_percent': cell_specs.efficiency,
                'power_per_m2_w': cell_specs.power_per_m2,
                'cost_per_m2_eur': cell_specs.cost_per_m2,
                'lifespan_years': cell_specs.lifespan_years
            },
            'system_configuration': {
                'peak_power_kw': max_power_kw,
                'optimal_tilt_degrees': optimal_tilt,
                'optimal_azimuth_degrees': optimal_azimuth,
                'number_of_panels': int((max_power_kw * 1000) / (cell_specs.power_per_m2 * 2)),  # Assuming 2m² per panel
                'panel_area_m2': max_power_kw * 1000 / cell_specs.power_per_m2
            },
            'energy_production': {
                'annual_energy_kwh': annual_energy,
                'monthly_energy': monthly_data,
                'energy_per_m2_kwh': annual_energy / roof_area if roof_area > 0 else 0,
                'capacity_factor': annual_energy / (max_power_kw * 8760) if max_power_kw > 0 else 0
            },
            'economic_analysis': {
                'total_system_cost_eur': roof_area * cell_specs.cost_per_m2,
                'cost_per_kwh_eur': (roof_area * cell_specs.cost_per_m2) / annual_energy if annual_energy > 0 else 0,
                'payback_period_years': (roof_area * cell_specs.cost_per_m2) / (annual_energy * 0.15) if annual_energy > 0 else 0  # Assuming 0.15 EUR/kWh
            }
        }
        
        return system_config
    
    def compare_technologies(self, roof_area: float = None) -> pd.DataFrame:
        """
        Compare different solar cell technologies for the selected building.
        
        Args:
            roof_area: Available roof area (uses building's roof area if None)
            
        Returns:
            DataFrame comparing different technologies
        """
        if self.selected_building is None:
            raise ValueError("No building selected. Use select_building() first.")
        
        if roof_area is None:
            roof_area = self.selected_building['Roof Area (m²)']
        
        comparison_data = []
        
        for tech_name, tech_specs in self.pvgis_client.SOLAR_TECHNOLOGIES.items():
            try:
                config = self.calculate_optimal_system(roof_area=roof_area, technology=tech_name)
                
                comparison_data.append({
                    'Technology': tech_name,
                    'Efficiency (%)': tech_specs.efficiency,
                    'Power per m² (W)': tech_specs.power_per_m2,
                    'Cost per m² (EUR)': tech_specs.cost_per_m2,
                    'Annual Energy (kWh)': config['energy_production']['annual_energy_kwh'],
                    'Energy per m² (kWh/m²)': config['energy_production']['energy_per_m2_kwh'],
                    'Total Cost (EUR)': config['economic_analysis']['total_system_cost_eur'],
                    'Cost per kWh (EUR)': config['economic_analysis']['cost_per_kwh_eur'],
                    'Payback Period (years)': config['economic_analysis']['payback_period_years']
                })
            except Exception as e:
                logger.warning(f"Failed to calculate for {tech_name}: {e}")
                continue
        
        return pd.DataFrame(comparison_data)


def main():
    """Example usage of the PVGIS integration."""
    from osm_roofs import OSMBuildingProcessor
    
    # Load building data
    processor = OSMBuildingProcessor()
    sample_data = create_sample_data()  # You'll need to implement this
    processor.load_from_dataframe(sample_data)
    
    # Initialize solar analyzer
    analyzer = BuildingSolarAnalyzer()
    
    # Select a building
    building = analyzer.select_building(processor.df, index=0)
    print(f"Selected building: {building['OSM ID']}")
    
    # Calculate optimal system
    system_config = analyzer.calculate_optimal_system()
    print(f"Annual energy production: {system_config['energy_production']['annual_energy_kwh']:.0f} kWh")
    
    # Compare technologies
    comparison = analyzer.compare_technologies()
    print("\nTechnology Comparison:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
