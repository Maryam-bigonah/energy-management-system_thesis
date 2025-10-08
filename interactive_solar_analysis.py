#!/usr/bin/env python3
"""
Interactive Building Solar Analysis with User Prompts

This script allows users to input their own parameters for:
1. PV Capacity Estimation from Roof Area
2. PVGIS Simulation Parameters
3. Building-specific data

Based on the methodology from your research.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torino_energy.data_sources.osm_roofs import OSMBuildingProcessor
from torino_energy.data_sources.pvgis_integration import BuildingSolarAnalyzer
import pandas as pd
import json
from typing import Dict, Any, Optional


class InteractiveSolarAnalyzer:
    """Interactive solar analyzer with user prompts for all parameters."""
    
    def __init__(self):
        self.processor = OSMBuildingProcessor()
        self.analyzer = BuildingSolarAnalyzer()
        
        # Default parameters from your methodology
        self.pv_params = {
            'coverage_factor': 0.65,  # Conservative coverage factor for residential buildings
            'module_area': 2.0,       # Standard crystalline-silicon PV module area [m²]
            'module_power': 420,      # Rated power [W]
            'cells_per_module': 144   # Number of cells per module
        }
        
        self.pvgis_params = {
            'tilt_angle': 35,         # Typical for Torino latitude [degrees]
            'azimuth': 0,             # South-facing [degrees]
            'system_losses': 14,      # System losses [%]
            'mounting_position': 'building_integrated',
            'location': 'Torino, Italy'
        }
        
        self.building_data = None
    
    def prompt_for_pv_parameters(self):
        """Prompt user for PV capacity estimation parameters."""
        print("\n" + "="*60)
        print("🔧 PV CAPACITY ESTIMATION PARAMETERS")
        print("="*60)
        print("Configure the parameters for PV capacity estimation from roof area.")
        print("Press Enter to use default values (shown in brackets).")
        
        # Coverage factor
        coverage_input = input(f"\n📐 Rooftop Coverage Factor [0.65]: ").strip()
        if coverage_input:
            try:
                self.pv_params['coverage_factor'] = float(coverage_input)
                if not (0.1 <= self.pv_params['coverage_factor'] <= 1.0):
                    print("⚠️  Warning: Coverage factor should be between 0.1 and 1.0")
            except ValueError:
                print("❌ Invalid input. Using default value 0.65")
        
        # Module area
        area_input = input(f"📦 PV Module Area [2.0 m²]: ").strip()
        if area_input:
            try:
                self.pv_params['module_area'] = float(area_input)
            except ValueError:
                print("❌ Invalid input. Using default value 2.0 m²")
        
        # Module power
        power_input = input(f"⚡ Module Rated Power [420 W]: ").strip()
        if power_input:
            try:
                self.pv_params['module_power'] = float(power_input)
            except ValueError:
                print("❌ Invalid input. Using default value 420 W")
        
        # Cells per module
        cells_input = input(f"🔋 Cells per Module [144]: ").strip()
        if cells_input:
            try:
                self.pv_params['cells_per_module'] = int(cells_input)
            except ValueError:
                print("❌ Invalid input. Using default value 144")
        
        self.display_pv_parameters()
    
    def prompt_for_pvgis_parameters(self):
        """Prompt user for PVGIS simulation parameters."""
        print("\n" + "="*60)
        print("🌍 PVGIS SIMULATION PARAMETERS")
        print("="*60)
        print("Configure the parameters for PVGIS hourly simulation.")
        print("Press Enter to use default values (shown in brackets).")
        
        # Location
        location_input = input(f"\n📍 Location [Torino, Italy]: ").strip()
        if location_input:
            self.pvgis_params['location'] = location_input
        
        # Tilt angle
        tilt_input = input(f"📐 Tilt Angle [35°]: ").strip()
        if tilt_input:
            try:
                self.pvgis_params['tilt_angle'] = float(tilt_input)
                if not (0 <= self.pvgis_params['tilt_angle'] <= 90):
                    print("⚠️  Warning: Tilt angle should be between 0° and 90°")
            except ValueError:
                print("❌ Invalid input. Using default value 35°")
        
        # Azimuth
        azimuth_input = input(f"🧭 Azimuth (South-facing = 0°) [0°]: ").strip()
        if azimuth_input:
            try:
                self.pvgis_params['azimuth'] = float(azimuth_input)
                if not (-180 <= self.pvgis_params['azimuth'] <= 180):
                    print("⚠️  Warning: Azimuth should be between -180° and 180°")
            except ValueError:
                print("❌ Invalid input. Using default value 0°")
        
        # System losses
        losses_input = input(f"📉 System Losses [14%]: ").strip()
        if losses_input:
            try:
                self.pvgis_params['system_losses'] = float(losses_input)
                if not (0 <= self.pvgis_params['system_losses'] <= 50):
                    print("⚠️  Warning: System losses should be between 0% and 50%")
            except ValueError:
                print("❌ Invalid input. Using default value 14%")
        
        # Mounting position
        print("\n🏗️  Mounting Position Options:")
        print("1. Building integrated")
        print("2. Free standing")
        print("3. Pitched roof")
        
        mount_input = input("Select mounting position [1]: ").strip()
        if mount_input == "2":
            self.pvgis_params['mounting_position'] = 'free_standing'
        elif mount_input == "3":
            self.pvgis_params['mounting_position'] = 'pitched_roof'
        else:
            self.pvgis_params['mounting_position'] = 'building_integrated'
        
        self.display_pvgis_parameters()
    
    def prompt_for_building_data(self):
        """Prompt user for building-specific data."""
        print("\n" + "="*60)
        print("🏢 BUILDING DATA INPUT")
        print("="*60)
        print("Enter building-specific information.")
        print("Press Enter to use default values or skip optional fields.")
        
        building = {}
        
        # Required fields
        osm_id = input("\n🆔 OSM ID (e.g., way/123456): ").strip()
        if not osm_id:
            print("❌ OSM ID is required!")
            return None
        building['OSM ID'] = osm_id
        
        # Optional fields with defaults
        building['Name'] = input("🏷️  Building Name (optional): ").strip()
        building['Building Type'] = input("🏠 Building Type [apartments]: ").strip() or 'apartments'
        building['Category'] = input("📂 Category [Residential]: ").strip() or 'Residential'
        
        # Numeric fields
        try:
            building['Height (m)'] = float(input("📏 Height [18 m]: ").strip() or "18")
            building['Floors'] = int(input("🏢 Number of Floors [6]: ").strip() or "6")
            building['Footprint Area (m²)'] = float(input("📐 Footprint Area [968.77 m²]: ").strip() or "968.77")
            building['Roof Area (m²)'] = float(input("🏠 Roof Area [1000.99 m²]: ").strip() or "1000.99")
            building['Estimated Population'] = int(input("👥 Estimated Population [142]: ").strip() or "142")
        except ValueError:
            print("❌ Invalid numeric input. Using default values.")
            building.update({
                'Height (m)': 18,
                'Floors': 6,
                'Footprint Area (m²)': 968.77,
                'Roof Area (m²)': 1000.99,
                'Estimated Population': 142
            })
        
        # Location
        try:
            building['Latitude'] = float(input("📍 Latitude [45.0447177]: ").strip() or "45.0447177")
            building['Longitude'] = float(input("📍 Longitude [7.6367993]: ").strip() or "7.6367993")
        except ValueError:
            print("❌ Invalid coordinates. Using Torino defaults.")
            building['Latitude'] = 45.0447177
            building['Longitude'] = 7.6367993
        
        # Optional fields
        building['Address'] = input("🏠 Address (optional): ").strip()
        building['Construction Year'] = input("📅 Construction Year (optional): ").strip()
        building['Material'] = input("🧱 Material (optional): ").strip()
        building['Energy Class'] = input("⚡ Energy Class (optional): ").strip()
        building['Data Sources'] = 'user_input'
        
        self.building_data = building
        self.display_building_data()
        return building
    
    def calculate_pv_capacity(self, roof_area: float) -> Dict[str, float]:
        """Calculate PV capacity based on user parameters."""
        # A_PV = 0.65 × A_roof (or user-defined coverage factor)
        usable_pv_area = self.pv_params['coverage_factor'] * roof_area
        
        # N_modules = A_PV / A_module
        num_modules = usable_pv_area / self.pv_params['module_area']
        
        # P_DC = N_modules × (P_Wp / 1000) [kW]
        installed_capacity = num_modules * (self.pv_params['module_power'] / 1000)
        
        # N_cells = N_modules × 144 (or user-defined cells per module)
        total_cells = num_modules * self.pv_params['cells_per_module']
        
        return {
            'roof_area_m2': roof_area,
            'usable_pv_area_m2': usable_pv_area,
            'num_modules': num_modules,
            'installed_capacity_kw': installed_capacity,
            'total_cells': total_cells,
            'coverage_factor': self.pv_params['coverage_factor']
        }
    
    def display_pv_parameters(self):
        """Display current PV parameters."""
        print("\n📋 CURRENT PV PARAMETERS:")
        print(f"   Coverage Factor: {self.pv_params['coverage_factor']}")
        print(f"   Module Area: {self.pv_params['module_area']} m²")
        print(f"   Module Power: {self.pv_params['module_power']} W")
        print(f"   Cells per Module: {self.pv_params['cells_per_module']}")
    
    def display_pvgis_parameters(self):
        """Display current PVGIS parameters."""
        print("\n📋 CURRENT PVGIS PARAMETERS:")
        print(f"   Location: {self.pvgis_params['location']}")
        print(f"   Tilt Angle: {self.pvgis_params['tilt_angle']}°")
        print(f"   Azimuth: {self.pvgis_params['azimuth']}°")
        print(f"   System Losses: {self.pvgis_params['system_losses']}%")
        print(f"   Mounting Position: {self.pvgis_params['mounting_position']}")
    
    def display_building_data(self):
        """Display current building data."""
        if not self.building_data:
            return
        
        print("\n📋 BUILDING DATA:")
        print(f"   OSM ID: {self.building_data['OSM ID']}")
        print(f"   Name: {self.building_data.get('Name', 'N/A')}")
        print(f"   Type: {self.building_data['Building Type']}")
        print(f"   Category: {self.building_data['Category']}")
        print(f"   Height: {self.building_data['Height (m)']} m")
        print(f"   Floors: {self.building_data['Floors']}")
        print(f"   Roof Area: {self.building_data['Roof Area (m²)']} m²")
        print(f"   Population: {self.building_data['Estimated Population']}")
        print(f"   Location: {self.building_data['Latitude']}, {self.building_data['Longitude']}")
    
    def run_analysis(self):
        """Run the complete solar analysis with user prompts."""
        print("🌞 INTERACTIVE TORINO SOLAR ANALYSIS")
        print("="*60)
        print("This tool allows you to customize all parameters for PV analysis.")
        
        # Step 1: Get PV parameters
        self.prompt_for_pv_parameters()
        
        # Step 2: Get PVGIS parameters
        self.prompt_for_pvgis_parameters()
        
        # Step 3: Get building data
        building = self.prompt_for_building_data()
        if not building:
            print("❌ Building data is required. Exiting.")
            return
        
        # Step 4: Calculate PV capacity
        print("\n" + "="*60)
        print("⚡ PV CAPACITY CALCULATION")
        print("="*60)
        
        roof_area = building['Roof Area (m²)']
        pv_capacity = self.calculate_pv_capacity(roof_area)
        
        print(f"\n🏠 Building: {building['OSM ID']}")
        print(f"📐 Total Roof Area: {roof_area:.2f} m²")
        print(f"🔧 Coverage Factor: {pv_capacity['coverage_factor']}")
        print(f"📦 Usable PV Area: {pv_capacity['usable_pv_area_m2']:.2f} m²")
        print(f"🔋 Number of Modules: {pv_capacity['num_modules']:.0f}")
        print(f"⚡ Installed Capacity: {pv_capacity['installed_capacity_kw']:.2f} kW")
        print(f"🔋 Total Cells: {pv_capacity['total_cells']:.0f}")
        
        # Step 5: Run PVGIS simulation (if available)
        print("\n" + "="*60)
        print("🌍 PVGIS SIMULATION")
        print("="*60)
        
        try:
            # Create DataFrame for the analyzer
            df = pd.DataFrame([building])
            self.analyzer.select_building(df, index=0)
            
            # Update analyzer with custom parameters
            self.analyzer.pvgis_client.tilt = self.pvgis_params['tilt_angle']
            self.analyzer.pvgis_client.azimuth = self.pvgis_params['azimuth']
            self.analyzer.pvgis_client.system_loss = self.pvgis_params['system_losses'] / 100
            
            # Run analysis
            system_config = self.analyzer.calculate_optimal_system(
                roof_area=roof_area
            )
            
            print("✅ PVGIS simulation completed successfully!")
            self.display_system_analysis(system_config)
            
        except Exception as e:
            print(f"⚠️  PVGIS simulation failed: {e}")
            print("Using estimated calculations instead...")
            self.display_estimated_analysis(building, pv_capacity)
        
        # Step 6: Save results
        self.save_results(building, pv_capacity, pv_params=self.pv_params, pvgis_params=self.pvgis_params)
    
    def display_system_analysis(self, system_config):
        """Display detailed system analysis results."""
        print("\n📊 DETAILED ANALYSIS RESULTS:")
        
        energy = system_config['energy_production']
        economic = system_config['economic_analysis']
        config = system_config['system_configuration']
        
        print(f"\n⚡ Energy Production:")
        print(f"   Annual Energy: {energy['annual_energy_kwh']:.0f} kWh/year")
        print(f"   Energy per m²: {energy['energy_per_m2_kwh']:.1f} kWh/m²/year")
        print(f"   Capacity Factor: {energy['capacity_factor']:.2%}")
        
        print(f"\n💰 Economic Analysis:")
        print(f"   Total Cost: €{economic['total_system_cost_eur']:,.0f}")
        print(f"   Cost per kWh: €{economic['cost_per_kwh_eur']:.3f}")
        print(f"   Payback Period: {economic['payback_period_years']:.1f} years")
        
        print(f"\n⚙️  System Configuration:")
        print(f"   Peak Power: {config['peak_power_kw']:.2f} kW")
        print(f"   Optimal Tilt: {config['optimal_tilt_degrees']:.1f}°")
        print(f"   Optimal Azimuth: {config['optimal_azimuth_degrees']:.1f}°")
    
    def display_estimated_analysis(self, building, pv_capacity):
        """Display estimated analysis when PVGIS is not available."""
        print("\n📊 ESTIMATED ANALYSIS RESULTS:")
        
        roof_area = building['Roof Area (m²)']
        lat = building['Latitude']
        
        # Estimated calculations
        optimal_tilt = max(0, min(90, lat + 10))
        solar_irradiance = 1400  # kWh/m²/year for Torino
        efficiency = 0.22  # 22% for crystalline silicon
        system_loss = self.pvgis_params['system_losses'] / 100
        
        annual_energy = roof_area * solar_irradiance * efficiency * (1 - system_loss)
        peak_power = pv_capacity['installed_capacity_kw']
        
        print(f"\n⚡ Energy Production:")
        print(f"   Annual Energy: {annual_energy:.0f} kWh/year")
        print(f"   Energy per m²: {annual_energy/roof_area:.1f} kWh/m²/year")
        print(f"   Capacity Factor: 0.18 (estimated)")
        
        print(f"\n⚙️  System Configuration:")
        print(f"   Peak Power: {peak_power:.2f} kW")
        print(f"   Optimal Tilt: {optimal_tilt:.1f}°")
        print(f"   Azimuth: {self.pvgis_params['azimuth']}° (south-facing)")
    
    def save_results(self, building, pv_capacity, pv_params, pvgis_params):
        """Save analysis results to file."""
        results = {
            'building_data': building,
            'pv_capacity': pv_capacity,
            'pv_parameters': pv_params,
            'pvgis_parameters': pvgis_params,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        filename = f"solar_analysis_{building['OSM ID'].replace('/', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")


def main():
    """Main function to run interactive solar analysis."""
    analyzer = InteractiveSolarAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
