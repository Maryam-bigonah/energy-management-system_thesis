#!/usr/bin/env python3
"""
Test script for the interactive solar analysis with custom parameters.

This script demonstrates how to use the new interactive analysis features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from interactive_solar_analysis import InteractiveSolarAnalyzer

def test_with_custom_parameters():
    """Test the interactive analyzer with custom parameters."""
    
    print("🧪 Testing Interactive Solar Analysis with Custom Parameters")
    print("="*60)
    
    # Initialize analyzer
    analyzer = InteractiveSolarAnalyzer()
    
    # Set custom PV parameters (based on your methodology)
    analyzer.pv_params = {
        'coverage_factor': 0.65,  # Conservative coverage factor for residential
        'module_area': 2.0,       # Standard crystalline-silicon PV module area [m²]
        'module_power': 420,      # Rated power [W]
        'cells_per_module': 144   # Number of cells per module
    }
    
    # Set custom PVGIS parameters
    analyzer.pvgis_params = {
        'tilt_angle': 35,         # Typical for Torino latitude [degrees]
        'azimuth': 0,             # South-facing [degrees]
        'system_losses': 14,      # System losses [%]
        'mounting_position': 'building_integrated',
        'location': 'Torino, Italy'
    }
    
    # Test building data
    test_building = {
        'OSM ID': 'way/test123456',
        'Name': 'Test Building',
        'Building Type': 'apartments',
        'Category': 'Residential',
        'Height (m)': 20,
        'Floors': 7,
        'Footprint Area (m²)': 800.0,
        'Roof Area (m²)': 850.0,
        'Estimated Population': 150,
        'Latitude': 45.0447177,
        'Longitude': 7.6367993,
        'Address': 'Test Address, Torino',
        'Construction Year': '2020',
        'Material': 'Concrete',
        'Energy Class': 'B',
        'Data Sources': 'test'
    }
    
    print("\n📋 Test Parameters:")
    analyzer.display_pv_parameters()
    analyzer.display_pvgis_parameters()
    
    print("\n🏢 Test Building:")
    analyzer.building_data = test_building
    analyzer.display_building_data()
    
    # Calculate PV capacity
    print("\n⚡ PV Capacity Calculation:")
    roof_area = test_building['Roof Area (m²)']
    pv_capacity = analyzer.calculate_pv_capacity(roof_area)
    
    print(f"   Roof Area: {roof_area} m²")
    print(f"   Coverage Factor: {pv_capacity['coverage_factor']}")
    print(f"   Usable PV Area: {pv_capacity['usable_pv_area_m2']:.2f} m²")
    print(f"   Number of Modules: {pv_capacity['num_modules']:.0f}")
    print(f"   Installed Capacity: {pv_capacity['installed_capacity_kw']:.2f} kW")
    print(f"   Total Cells: {pv_capacity['total_cells']:.0f}")
    
    # Test estimated analysis
    print("\n📊 Estimated Analysis (without PVGIS):")
    analyzer.display_estimated_analysis(test_building, pv_capacity)
    
    print("\n✅ Test completed successfully!")
    print("\n💡 To run the full interactive version, use:")
    print("   python3 interactive_solar_analysis.py")

if __name__ == "__main__":
    test_with_custom_parameters()
