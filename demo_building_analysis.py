#!/usr/bin/env python3
"""
Demo Building Solar Analysis with PVGIS Integration

This script demonstrates the solar analysis capabilities without requiring interactive input.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torino_energy.data_sources.osm_roofs import OSMBuildingProcessor
from torino_energy.data_sources.pvgis_integration import BuildingSolarAnalyzer
import pandas as pd


def create_sample_data():
    """Create sample data based on the building data shown in the image."""
    
    sample_buildings = [
        {
            'OSM ID': 'way/49062146',
            'Name': '',
            'Building Type': 'apartments',
            'Category': 'Residential',
            'Height (m)': 18,
            'Floors': 6,
            'Footprint Area (m²)': 968.77,
            'Roof Area (m²)': 1000.99,
            'Estimated Population': 142,
            'Flats': '',
            'Units': '',
            'Apartments': '',
            'Rooms': '',
            'Address': '',
            'Construction Year': '',
            'Material': '',
            'Energy Class': '',
            'Data Sources': 'osm',
            'Latitude': 45.0447177,
            'Longitude': 7.6367993
        },
        {
            'OSM ID': 'way/182810424',
            'Name': '',
            'Building Type': 'apartments',
            'Category': 'Residential',
            'Height (m)': 24,
            'Floors': 8,
            'Footprint Area (m²)': 675.73,
            'Roof Area (m²)': 675.73,
            'Estimated Population': 132,
            'Flats': '',
            'Units': '',
            'Apartments': '',
            'Rooms': '',
            'Address': '181, Via Filadelfia, Torino',
            'Construction Year': '',
            'Material': '',
            'Energy Class': '',
            'Data Sources': 'osm',
            'Latitude': 45.0450191,
            'Longitude': 7.6359563
        },
        {
            'OSM ID': 'way/123456789',
            'Name': '',
            'Building Type': 'apartments',
            'Category': 'Residential',
            'Height (m)': 17,
            'Floors': 5,
            'Footprint Area (m²)': 749.27,
            'Roof Area (m²)': 774.18,
            'Estimated Population': 94,
            'Flats': '',
            'Units': '',
            'Apartments': '',
            'Rooms': '',
            'Address': '',
            'Construction Year': '',
            'Material': '',
            'Energy Class': '',
            'Data Sources': 'osm',
            'Latitude': 45.0439202,
            'Longitude': 7.639273
        }
    ]
    
    return pd.DataFrame(sample_buildings)


def display_building_list(buildings_df):
    """Display available buildings for selection."""
    print("\n🏢 Available Buildings:")
    print("=" * 80)
    
    for idx, building in buildings_df.iterrows():
        print(f"{idx + 1:2d}. {building['OSM ID']}")
        print(f"    📍 Address: {building.get('Address', 'N/A')}")
        print(f"    🏠 Type: {building['Building Type']} | Category: {building['Category']}")
        print(f"    📏 Height: {building['Height (m)']}m | Floors: {building['Floors']}")
        print(f"    🏠 Roof Area: {building['Roof Area (m²)']:.1f} m²")
        print(f"    👥 Population: {building['Estimated Population']}")
        print(f"    📍 Coordinates: {building['Latitude']:.6f}, {building['Longitude']:.6f}")
        print()


def display_system_analysis(system_config):
    """Display detailed system analysis results."""
    print("\n" + "="*80)
    print("🔋 SOLAR SYSTEM ANALYSIS RESULTS")
    print("="*80)
    
    # Building Information
    print(f"\n🏢 Building Information:")
    print(f"   OSM ID: {system_config['building_id']}")
    print(f"   Address: {system_config['building_address']}")
    print(f"   Roof Area: {system_config['roof_area_m2']:.1f} m²")
    
    # Technology Specifications
    print(f"\n🔬 Solar Cell Technology:")
    specs = system_config['cell_specifications']
    print(f"   Technology: {system_config['technology'].replace('_', ' ').title()}")
    print(f"   Efficiency: {specs['efficiency_percent']:.1f}%")
    print(f"   Power per m²: {specs['power_per_m2_w']:.0f} W/m²")
    print(f"   Cost per m²: €{specs['cost_per_m2_eur']:.0f}")
    print(f"   Lifespan: {specs['lifespan_years']} years")
    
    # System Configuration
    print(f"\n⚙️ System Configuration:")
    config = system_config['system_configuration']
    print(f"   Peak Power: {config['peak_power_kw']:.2f} kW")
    print(f"   Optimal Tilt: {config['optimal_tilt_degrees']:.1f}°")
    print(f"   Optimal Azimuth: {config['optimal_azimuth_degrees']:.1f}°")
    print(f"   Number of Panels: {config['number_of_panels']}")
    print(f"   Panel Area: {config['panel_area_m2']:.1f} m²")
    
    # Energy Production
    print(f"\n⚡ Energy Production:")
    energy = system_config['energy_production']
    print(f"   Annual Energy: {energy['annual_energy_kwh']:.0f} kWh/year")
    print(f"   Energy per m²: {energy['energy_per_m2_kwh']:.1f} kWh/m²/year")
    print(f"   Capacity Factor: {energy['capacity_factor']:.2%}")
    
    # Economic Analysis
    print(f"\n💰 Economic Analysis:")
    economic = system_config['economic_analysis']
    print(f"   Total System Cost: €{economic['total_system_cost_eur']:,.0f}")
    print(f"   Cost per kWh: €{economic['cost_per_kwh_eur']:.3f}")
    print(f"   Payback Period: {economic['payback_period_years']:.1f} years")


def display_technology_comparison(comparison_df):
    """Display technology comparison results."""
    print("\n" + "="*100)
    print("🔬 SOLAR TECHNOLOGY COMPARISON")
    print("="*100)
    
    # Sort by annual energy production
    comparison_df = comparison_df.sort_values('Annual Energy (kWh)', ascending=False)
    
    print(f"\n{'Technology':<15} {'Efficiency':<12} {'Annual Energy':<15} {'Total Cost':<12} {'Payback':<10}")
    print(f"{'':<15} {'(%)':<12} {'(kWh)':<15} {'(EUR)':<12} {'(years)':<10}")
    print("-" * 100)
    
    for _, row in comparison_df.iterrows():
        print(f"{row['Technology'].replace('_', ' ').title():<15} "
              f"{row['Efficiency (%)']:<12.1f} "
              f"{row['Annual Energy (kWh)']:<15.0f} "
              f"{row['Total Cost (EUR)']:<12.0f} "
              f"{row['Payback Period (years)']:<10.1f}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    best_energy = comparison_df.loc[comparison_df['Annual Energy (kWh)'].idxmax()]
    best_cost = comparison_df.loc[comparison_df['Cost per kWh (EUR)'].idxmin()]
    best_payback = comparison_df.loc[comparison_df['Payback Period (years)'].idxmin()]
    
    print(f"   🏆 Highest Energy Production: {best_energy['Technology'].replace('_', ' ').title()}")
    print(f"   💰 Lowest Cost per kWh: {best_cost['Technology'].replace('_', ' ').title()}")
    print(f"   ⏱️ Shortest Payback: {best_payback['Technology'].replace('_', ' ').title()}")


def main():
    """Main demo function."""
    print("🌞 Torino Building Solar Analysis Demo with PVGIS")
    print("=" * 60)
    
    # Load building data
    processor = OSMBuildingProcessor()
    sample_data = create_sample_data()
    processor.load_from_dataframe(sample_data)
    cleaned_df = processor.clean_data()
    
    print(f"✅ Loaded {len(cleaned_df)} buildings from dataset")
    
    # Display available buildings
    display_building_list(cleaned_df)
    
    # Initialize solar analyzer
    analyzer = BuildingSolarAnalyzer()
    
    # Demo 1: Analyze the building with largest roof area
    print("\n" + "="*60)
    print("🔍 DEMO 1: Analyzing Building with Largest Roof Area")
    print("="*60)
    
    largest_roof_idx = cleaned_df['Roof Area (m²)'].idxmax()
    analyzer.select_building(cleaned_df, index=largest_roof_idx)
    
    try:
        print(f"🔄 Analyzing building with mono-crystalline technology...")
        system_config = analyzer.calculate_optimal_system(technology="mono_crystalline")
        display_system_analysis(system_config)
    except Exception as e:
        print(f"❌ Error analyzing building: {e}")
        print("This might be due to PVGIS API connectivity issues.")
        print("Using estimated calculations instead...")
        
        # Fallback calculation
        building = analyzer.selected_building
        roof_area = building['Roof Area (m²)']
        lat = building['Latitude']
        
        # Estimated calculations
        optimal_tilt = max(0, min(90, lat + 10))
        solar_irradiance = 1400  # kWh/m²/year for Torino
        efficiency = 0.22  # 22% for mono-crystalline
        system_loss = 0.14  # 14% system losses
        
        annual_energy = roof_area * solar_irradiance * efficiency * (1 - system_loss)
        peak_power = (roof_area * 220) / 1000  # 220 W/m²
        
        print(f"\n📊 ESTIMATED RESULTS (without PVGIS API):")
        print(f"   Building: {building['OSM ID']}")
        print(f"   Roof Area: {roof_area:.1f} m²")
        print(f"   Optimal Tilt: {optimal_tilt:.1f}°")
        print(f"   Peak Power: {peak_power:.2f} kW")
        print(f"   Annual Energy: {annual_energy:.0f} kWh/year")
        print(f"   Energy per m²: {annual_energy/roof_area:.1f} kWh/m²/year")
    
    # Demo 2: Technology comparison
    print("\n" + "="*60)
    print("🔍 DEMO 2: Technology Comparison")
    print("="*60)
    
    try:
        print(f"🔄 Comparing all technologies...")
        comparison_df = analyzer.compare_technologies()
        display_technology_comparison(comparison_df)
    except Exception as e:
        print(f"❌ Error comparing technologies: {e}")
        print("This might be due to PVGIS API connectivity issues.")
        print("Using estimated calculations instead...")
        
        # Fallback comparison
        building = analyzer.selected_building
        roof_area = building['Roof Area (m²)']
        
        print(f"\n📊 ESTIMATED TECHNOLOGY COMPARISON:")
        print(f"{'Technology':<15} {'Efficiency':<12} {'Annual Energy':<15} {'Cost/m²':<10}")
        print("-" * 60)
        
        technologies = {
            'Mono-crystalline': {'efficiency': 22, 'power_per_m2': 220, 'cost_per_m2': 180},
            'Poly-crystalline': {'efficiency': 18, 'power_per_m2': 180, 'cost_per_m2': 140},
            'Thin Film': {'efficiency': 12, 'power_per_m2': 120, 'cost_per_m2': 100},
            'Perovskite': {'efficiency': 25, 'power_per_m2': 250, 'cost_per_m2': 200}
        }
        
        for tech_name, specs in technologies.items():
            annual_energy = roof_area * 1400 * (specs['efficiency']/100) * 0.86
            print(f"{tech_name:<15} {specs['efficiency']:<12.0f}% {annual_energy:<15.0f} €{specs['cost_per_m2']:<10.0f}")
    
    # Demo 3: Show dataset statistics
    print("\n" + "="*60)
    print("🔍 DEMO 3: Dataset Statistics")
    print("="*60)
    
    stats = processor.get_building_statistics()
    print(f"\n📊 Building Dataset Statistics:")
    print("=" * 40)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    print(f"\n✅ Demo completed! You can now use the interactive version:")
    print(f"   python3 building_solar_analysis.py")


if __name__ == "__main__":
    main()
