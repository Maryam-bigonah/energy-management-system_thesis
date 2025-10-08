#!/usr/bin/env python3
"""
Interactive Building Solar Analysis with PVGIS Integration

This script allows you to:
1. Select a specific building from your dataset
2. Calculate solar energy potential using PVGIS
3. Compare different solar cell technologies
4. Get detailed metrics and economic analysis
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
        },
        {
            'OSM ID': 'way/987654321',
            'Name': '',
            'Building Type': 'apartments',
            'Category': 'Residential',
            'Height (m)': 30,
            'Floors': 10,
            'Footprint Area (m²)': 1180.39,
            'Roof Area (m²)': 1219.64,
            'Estimated Population': 110,
            'Flats': '',
            'Units': '',
            'Apartments': '',
            'Rooms': '',
            'Address': '',
            'Construction Year': '',
            'Material': '',
            'Energy Class': '',
            'Data Sources': 'osm',
            'Latitude': 45.0445000,
            'Longitude': 7.6370000
        },
        {
            'OSM ID': 'way/555666777',
            'Name': '',
            'Building Type': 'apartments',
            'Category': 'Residential',
            'Height (m)': 15,
            'Floors': 5,
            'Footprint Area (m²)': 850.00,
            'Roof Area (m²)': 850.00,
            'Estimated Population': 180,
            'Flats': '',
            'Units': '',
            'Apartments': '',
            'Rooms': '',
            'Address': '',
            'Construction Year': '',
            'Material': '',
            'Energy Class': '',
            'Data Sources': 'osm',
            'Latitude': 45.0455000,
            'Longitude': 7.6380000
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


def select_building_interactive(buildings_df):
    """Interactive building selection."""
    display_building_list(buildings_df)
    
    while True:
        try:
            choice = input("Enter building number (1-{}) or 'q' to quit: ".format(len(buildings_df)))
            
            if choice.lower() == 'q':
                return None
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(buildings_df):
                return choice_idx
            else:
                print("❌ Invalid choice. Please enter a number between 1 and {}.".format(len(buildings_df)))
                
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit.")


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
    
    # Monthly breakdown if available
    if 'monthly_energy' in energy and energy['monthly_energy']:
        print(f"\n📅 Monthly Energy Production:")
        monthly = energy['monthly_energy']
        if isinstance(monthly, dict):
            for month, value in monthly.items():
                if isinstance(value, dict) and 'E_m' in value:
                    print(f"   {month}: {value['E_m']:.0f} kWh")


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
    """Main interactive analysis function."""
    print("🌞 Torino Building Solar Analysis with PVGIS")
    print("=" * 60)
    
    # Load building data
    processor = OSMBuildingProcessor()
    sample_data = create_sample_data()
    processor.load_from_dataframe(sample_data)
    cleaned_df = processor.clean_data()
    
    print(f"✅ Loaded {len(cleaned_df)} buildings from dataset")
    
    # Initialize solar analyzer
    analyzer = BuildingSolarAnalyzer()
    
    while True:
        print(f"\n{'='*60}")
        print("📋 MAIN MENU")
        print("="*60)
        print("1. Select building and analyze solar potential")
        print("2. Compare all technologies for a building")
        print("3. Show building statistics")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Select building
            building_idx = select_building_interactive(cleaned_df)
            if building_idx is None:
                continue
            
            # Select technology
            print(f"\n🔬 Available Solar Technologies:")
            technologies = list(analyzer.pvgis_client.SOLAR_TECHNOLOGIES.keys())
            for i, tech in enumerate(technologies, 1):
                specs = analyzer.pvgis_client.SOLAR_TECHNOLOGIES[tech]
                print(f"{i}. {tech.replace('_', ' ').title()} (Efficiency: {specs.efficiency}%)")
            
            tech_choice = input(f"\nSelect technology (1-{len(technologies)}) or press Enter for mono_crystalline: ").strip()
            
            if tech_choice:
                try:
                    tech_idx = int(tech_choice) - 1
                    selected_tech = technologies[tech_idx]
                except (ValueError, IndexError):
                    selected_tech = "mono_crystalline"
            else:
                selected_tech = "mono_crystalline"
            
            # Analyze building
            try:
                print(f"\n🔄 Analyzing building with {selected_tech} technology...")
                analyzer.select_building(cleaned_df, index=building_idx)
                system_config = analyzer.calculate_optimal_system(technology=selected_tech)
                display_system_analysis(system_config)
                
                # Ask for custom roof area
                custom_area = input(f"\nEnter custom roof area (m²) or press Enter to use {system_config['roof_area_m2']:.1f} m²: ").strip()
                if custom_area:
                    try:
                        custom_area = float(custom_area)
                        system_config = analyzer.calculate_optimal_system(roof_area=custom_area, technology=selected_tech)
                        display_system_analysis(system_config)
                    except ValueError:
                        print("❌ Invalid roof area. Using default value.")
                
            except Exception as e:
                print(f"❌ Error analyzing building: {e}")
                print("This might be due to PVGIS API connectivity issues.")
        
        elif choice == '2':
            # Compare technologies
            building_idx = select_building_interactive(cleaned_df)
            if building_idx is None:
                continue
            
            try:
                print(f"\n🔄 Comparing all technologies...")
                analyzer.select_building(cleaned_df, index=building_idx)
                comparison_df = analyzer.compare_technologies()
                display_technology_comparison(comparison_df)
                
            except Exception as e:
                print(f"❌ Error comparing technologies: {e}")
                print("This might be due to PVGIS API connectivity issues.")
        
        elif choice == '3':
            # Show statistics
            stats = processor.get_building_statistics()
            print(f"\n📊 Building Dataset Statistics:")
            print("=" * 40)
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"{key.replace('_', ' ').title()}: {value:.2f}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value}")
        
        elif choice == '4':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()
