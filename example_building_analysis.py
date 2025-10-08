#!/usr/bin/env python3
"""
Example script demonstrating how to use the OSM Building Data Processor
with the building data from Torino.

This script shows how to:
1. Load building data from CSV
2. Validate and clean the data
3. Calculate energy potential
4. Generate statistics and insights
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torino_energy.data_sources.osm_roofs import OSMBuildingProcessor
import pandas as pd


def main():
    """Main function demonstrating building data analysis."""
    
    # Initialize the processor
    processor = OSMBuildingProcessor()
    
    print("🏢 Torino Building Data Analysis")
    print("=" * 50)
    
    # Example 1: Load data from CSV (uncomment when you have the CSV file)
    # csv_file = "building-data-All-2025-10-06T12-26-43.csv"
    # if os.path.exists(csv_file):
    #     processor.load_from_csv(csv_file)
    # else:
    #     print(f"❌ CSV file not found: {csv_file}")
    #     print("Please place your building data CSV file in the current directory.")
    #     return
    
    # Example 2: Create sample data based on the image you showed
    sample_data = create_sample_data()
    processor.load_from_dataframe(sample_data)
    
    print(f"✅ Loaded {len(processor.df)} buildings")
    
    # Validate the data
    print("\n📊 Data Validation:")
    validation_stats = processor.validate_data()
    
    # Clean the data
    print("\n🧹 Cleaning data...")
    cleaned_df = processor.clean_data()
    processor.df = cleaned_df
    
    # Calculate energy potential
    print("\n⚡ Calculating solar energy potential...")
    energy_df = processor.calculate_energy_potential()
    processor.df = energy_df
    
    # Get comprehensive statistics
    print("\n📈 Building Statistics:")
    stats = processor.get_building_statistics()
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Show top buildings by energy potential
    print("\n🔋 Top 5 Buildings by Solar Energy Potential:")
    top_buildings = energy_df.nlargest(5, 'Solar_Potential_kWh_year')[
        ['OSM ID', 'Building Type', 'Roof Area (m²)', 'Estimated Population', 'Solar_Potential_kWh_year']
    ]
    print(top_buildings.to_string(index=False))
    
    # Show energy potential summary
    print(f"\n🌞 Total Solar Energy Potential: {energy_df['Solar_Potential_kWh_year'].sum():,.0f} kWh/year")
    print(f"👥 Total Population: {energy_df['Estimated Population'].sum():,}")
    print(f"🏠 Average Energy per Person: {energy_df['Energy_per_Person_kWh_year'].mean():.1f} kWh/year")
    
    # Export results
    output_file = "processed_building_data.csv"
    processor.export_to_csv(output_file, include_energy=True)
    print(f"\n💾 Results exported to: {output_file}")


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


if __name__ == "__main__":
    main()
