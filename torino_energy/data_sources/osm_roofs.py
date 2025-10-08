"""
OpenStreetMap Building Data Processing Module

This module handles the processing and analysis of building data from OpenStreetMap,
including roof area calculations, population estimates, and building characteristics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BuildingData:
    """Data class for building information from OSM."""
    osm_id: str
    name: Optional[str] = None
    building_type: Optional[str] = None
    category: Optional[str] = None
    height: Optional[float] = None
    floors: Optional[int] = None
    footprint_area: Optional[float] = None
    roof_area: Optional[float] = None
    estimated_population: Optional[int] = None
    flats: Optional[int] = None
    units: Optional[int] = None
    apartments: Optional[int] = None
    rooms: Optional[int] = None
    address: Optional[str] = None
    construction_year: Optional[int] = None
    material: Optional[str] = None
    energy_class: Optional[str] = None
    data_sources: str = "osm"
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class OSMBuildingProcessor:
    """Processor for OpenStreetMap building data."""
    
    def __init__(self):
        self.buildings_data: List[BuildingData] = []
        self.df: Optional[pd.DataFrame] = None
    
    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load building data from CSV file.
        
        Args:
            file_path: Path to the CSV file containing building data
            
        Returns:
            DataFrame with building data
        """
        try:
            self.df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(self.df)} buildings from {file_path}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
    
    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Load building data from existing DataFrame.
        
        Args:
            df: DataFrame containing building data
        """
        self.df = df.copy()
        logger.info(f"Loaded {len(self.df)} buildings from DataFrame")
    
    def validate_data(self) -> Dict[str, int]:
        """
        Validate the loaded building data and return statistics.
        
        Returns:
            Dictionary with validation statistics
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load data first.")
        
        stats = {
            'total_buildings': len(self.df),
            'missing_osm_id': self.df['OSM ID'].isna().sum(),
            'missing_coordinates': (self.df['Latitude'].isna() | self.df['Longitude'].isna()).sum(),
            'missing_height': self.df['Height (m)'].isna().sum(),
            'missing_roof_area': self.df['Roof Area (m²)'].isna().sum(),
            'missing_population': self.df['Estimated Population'].isna().sum(),
            'residential_buildings': len(self.df[self.df['Category'] == 'Residential']),
            'apartment_buildings': len(self.df[self.df['Building Type'] == 'apartments'])
        }
        
        logger.info("Data validation completed:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        return stats
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean and standardize the building data.
        
        Returns:
            Cleaned DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load data first.")
        
        df_clean = self.df.copy()
        
        # Remove rows with missing essential data
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=['OSM ID', 'Latitude', 'Longitude'])
        
        # Standardize column names
        df_clean.columns = df_clean.columns.str.strip()
        
        # Convert numeric columns
        numeric_columns = ['Height (m)', 'Floors', 'Footprint Area (m²)', 'Roof Area (m²)', 
                          'Estimated Population', 'Flats', 'Units', 'Apartments', 'Rooms',
                          'Latitude', 'Longitude', 'Construction Year']
        
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Fill missing roof areas with footprint areas if available
        missing_roof = df_clean['Roof Area (m²)'].isna()
        has_footprint = df_clean['Footprint Area (m²)'].notna()
        df_clean.loc[missing_roof & has_footprint, 'Roof Area (m²)'] = \
            df_clean.loc[missing_roof & has_footprint, 'Footprint Area (m²)']
        
        # Estimate missing population based on building characteristics
        df_clean = self._estimate_missing_population(df_clean)
        
        logger.info(f"Data cleaning completed. Removed {initial_count - len(df_clean)} invalid rows.")
        return df_clean
    
    def _estimate_missing_population(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate missing population data based on building characteristics.
        
        Args:
            df: DataFrame with building data
            
        Returns:
            DataFrame with estimated population filled
        """
        df_est = df.copy()
        
        # Estimate population based on roof area for residential buildings
        residential_mask = (df_est['Category'] == 'Residential') & df_est['Estimated Population'].isna()
        
        if residential_mask.any():
            # Average population density per m² of roof area (rough estimate)
            avg_density = 0.15  # people per m²
            df_est.loc[residential_mask, 'Estimated Population'] = \
                (df_est.loc[residential_mask, 'Roof Area (m²)'] * avg_density).round().astype('Int64')
        
        return df_est
    
    def calculate_energy_potential(self, solar_efficiency: float = 0.20) -> pd.DataFrame:
        """
        Calculate solar energy potential for buildings.
        
        Args:
            solar_efficiency: Solar panel efficiency (default 0.20 for 20%)
            
        Returns:
            DataFrame with energy potential calculations
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load data first.")
        
        df_energy = self.df.copy()
        
        # Solar irradiance in Torino (kWh/m²/year) - approximate value
        solar_irradiance = 1400  # kWh/m²/year
        
        # Calculate potential solar energy generation
        df_energy['Solar_Potential_kWh_year'] = (
            df_energy['Roof Area (m²)'] * solar_irradiance * solar_efficiency
        )
        
        # Calculate potential energy per person
        df_energy['Energy_per_Person_kWh_year'] = (
            df_energy['Solar_Potential_kWh_year'] / df_energy['Estimated Population']
        )
        
        # Calculate potential energy per m² of building
        df_energy['Energy_per_m2_kWh_year'] = (
            df_energy['Solar_Potential_kWh_year'] / df_energy['Roof Area (m²)']
        )
        
        logger.info("Energy potential calculations completed")
        return df_energy
    
    def get_building_statistics(self) -> Dict[str, float]:
        """
        Calculate comprehensive building statistics.
        
        Returns:
            Dictionary with building statistics
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load data first.")
        
        stats = {
            'total_buildings': len(self.df),
            'total_roof_area_m2': self.df['Roof Area (m²)'].sum(),
            'total_population': self.df['Estimated Population'].sum(),
            'avg_building_height': self.df['Height (m)'].mean(),
            'avg_floors': self.df['Floors'].mean(),
            'avg_roof_area': self.df['Roof Area (m²)'].mean(),
            'avg_population_per_building': self.df['Estimated Population'].mean(),
            'residential_percentage': (self.df['Category'] == 'Residential').mean() * 100,
            'apartment_percentage': (self.df['Building Type'] == 'apartments').mean() * 100
        }
        
        return stats
    
    def filter_by_area(self, min_lat: float, max_lat: float, 
                      min_lon: float, max_lon: float) -> pd.DataFrame:
        """
        Filter buildings by geographic area.
        
        Args:
            min_lat, max_lat: Latitude bounds
            min_lon, max_lon: Longitude bounds
            
        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load data first.")
        
        mask = (
            (self.df['Latitude'] >= min_lat) & (self.df['Latitude'] <= max_lat) &
            (self.df['Longitude'] >= min_lon) & (self.df['Longitude'] <= max_lon)
        )
        
        filtered_df = self.df[mask].copy()
        logger.info(f"Filtered to {len(filtered_df)} buildings in specified area")
        
        return filtered_df
    
    def export_to_csv(self, file_path: str, include_energy: bool = True) -> None:
        """
        Export processed data to CSV file.
        
        Args:
            file_path: Output file path
            include_energy: Whether to include energy potential calculations
        """
        if self.df is None:
            raise ValueError("No data loaded. Please load data first.")
        
        export_df = self.df.copy()
        
        if include_energy:
            export_df = self.calculate_energy_potential()
        
        export_df.to_csv(file_path, index=False)
        logger.info(f"Data exported to {file_path}")


def main():
    """Example usage of the OSMBuildingProcessor."""
    processor = OSMBuildingProcessor()
    
    # Example: Load data from CSV
    # processor.load_from_csv('building-data-All-2025-10-06T12-26-43.csv')
    
    # Example: Process and analyze data
    # processor.validate_data()
    # cleaned_df = processor.clean_data()
    # energy_df = processor.calculate_energy_potential()
    # stats = processor.get_building_statistics()
    
    print("OSM Building Data Processor initialized successfully!")


if __name__ == "__main__":
    main()
