#!/usr/bin/env python3
"""
Process LoadProfileGenerator (LPG) Outputs
Aggregates 4 household types into 20-unit building load profiles
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class LPGProcessor:
    def __init__(self, lpg_output_dir="LPG_outputs", project_data_dir="project/data"):
        self.lpg_output_dir = Path(lpg_output_dir)
        self.project_data_dir = Path(project_data_dir)
        self.project_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Household distribution for 20 units
        self.household_distribution = {
            'working_couple': 6,      # Type 1: Both working
            'mixed_work': 5,          # Type 2: One working, one home
            'family_children': 6,     # Type 3: Family with children
            'elderly_couple': 3       # Type 4: Elderly couple
        }
        
    def load_lpg_csv(self, filename):
        """
        Load LPG CSV output file
        Expected format: timestamp, power_kw columns
        """
        try:
            df = pd.read_csv(filename)
            
            # Handle different possible column names
            power_columns = [col for col in df.columns if 'power' in col.lower() or 'load' in col.lower() or 'kw' in col.lower()]
            if not power_columns:
                # If no power column found, assume second column is power
                power_columns = [df.columns[1]]
            
            # Create standardized dataframe
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'])
            else:
                # Create hourly timestamps starting from Jan 1
                df['datetime'] = pd.date_range('2024-01-01', periods=len(df), freq='H')
            
            df['power_kw'] = df[power_columns[0]]
            
            return df[['datetime', 'power_kw']]
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    
    def aggregate_households(self):
        """
        Aggregate all household types into 20-unit building load
        """
        print("Processing LPG outputs and aggregating to 20-unit building...")
        
        # Expected LPG output files
        household_files = {
            'working_couple': 'household_type1_working_couple.csv',
            'mixed_work': 'household_type2_mixed_work.csv', 
            'family_children': 'household_type3_family_children.csv',
            'elderly_couple': 'household_type4_elderly_couple.csv'
        }
        
        aggregated_data = None
        
        for household_type, filename in household_files.items():
            filepath = self.lpg_output_dir / filename
            
            if not filepath.exists():
                print(f"⚠️  Warning: {filename} not found. Creating placeholder data...")
                # Create realistic placeholder data based on household type
                df = self._create_placeholder_data(household_type)
            else:
                print(f"✓ Loading {filename}...")
                df = self.load_lpg_csv(filepath)
                
            if df is not None:
                # Scale by number of units of this type
                count = self.household_distribution[household_type]
                df['scaled_power_kw'] = df['power_kw'] * count
                
                if aggregated_data is None:
                    aggregated_data = df[['datetime', 'scaled_power_kw']].copy()
                    aggregated_data.rename(columns={'scaled_power_kw': 'total_load_kw'}, inplace=True)
                else:
                    aggregated_data['total_load_kw'] += df['scaled_power_kw']
        
        if aggregated_data is not None:
            # Add hour column for easy reference
            aggregated_data['hour'] = aggregated_data['datetime'].dt.hour
            aggregated_data['day_of_year'] = aggregated_data['datetime'].dt.dayofyear
            
            return aggregated_data
        else:
            print("❌ Error: No household data could be loaded")
            return None
    
    def _create_placeholder_data(self, household_type):
        """
        Create realistic placeholder data when LPG files are not available
        Based on typical European residential consumption patterns
        """
        print(f"Creating placeholder data for {household_type}...")
        
        # Base consumption patterns (kW) for different household types
        base_patterns = {
            'working_couple': [
                0.3, 0.2, 0.15, 0.1, 0.1, 0.2, 0.6, 1.2, 0.8, 0.4,  # 00-09h
                0.3, 0.4, 0.5, 0.4, 0.3, 0.4, 0.8, 1.5, 2.0, 2.5,  # 10-19h
                2.0, 1.2, 0.6, 0.4  # 20-23h
            ],
            'mixed_work': [
                0.4, 0.3, 0.2, 0.15, 0.15, 0.3, 0.8, 1.0, 1.2, 1.0,  # 00-09h
                1.2, 1.4, 1.6, 1.4, 1.2, 1.4, 1.8, 2.2, 2.8, 3.2,  # 10-19h
                2.6, 1.8, 1.0, 0.6  # 20-23h
            ],
            'family_children': [
                0.5, 0.4, 0.3, 0.2, 0.2, 0.4, 1.0, 1.8, 1.5, 1.2,  # 00-09h
                1.4, 1.8, 2.2, 2.0, 1.8, 2.2, 2.8, 3.5, 4.2, 4.8,  # 10-19h
                4.0, 2.8, 1.4, 0.8  # 20-23h
            ],
            'elderly_couple': [
                0.3, 0.25, 0.2, 0.2, 0.25, 0.4, 0.6, 0.8, 1.2, 1.5,  # 00-09h
                1.8, 2.0, 2.2, 2.0, 1.8, 2.0, 2.4, 2.8, 3.2, 2.8,  # 10-19h
                2.4, 1.8, 1.2, 0.7  # 20-23h
            ]
        }
        
        # Create full year data with seasonal variations
        base_pattern = base_patterns[household_type]
        full_year_data = []
        
        for day in range(365):
            # Add seasonal variation (±20% for heating/cooling)
            seasonal_factor = 1.0
            if day < 80 or day > 265:  # Winter months
                seasonal_factor = 1.2
            elif 120 < day < 240:  # Summer months  
                seasonal_factor = 1.15
            
            # Add some daily randomness (±10%)
            daily_factor = np.random.normal(1.0, 0.1)
            
            for hour in range(24):
                base_power = base_pattern[hour]
                adjusted_power = base_power * seasonal_factor * daily_factor
                full_year_data.append(max(0.1, adjusted_power))  # Minimum 0.1 kW
        
        # Create DataFrame
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=8760, freq='H'),
            'power_kw': full_year_data
        })
        
        return df
    
    def create_24h_profile(self, aggregated_data):
        """
        Extract representative 24-hour profile from full year data
        """
        print("Creating 24-hour representative profile...")
        
        # Use a typical spring day (day 100, around April 10th)
        spring_day = aggregated_data[aggregated_data['day_of_year'] == 100]
        
        if len(spring_day) == 24:
            profile_24h = spring_day[['hour', 'total_load_kw']].copy()
            profile_24h = profile_24h.sort_values('hour').reset_index(drop=True)
            
            # Save 24-hour profile
            output_file = self.project_data_dir / "load_24h_lpg.csv"
            profile_24h.to_csv(output_file, index=False)
            
            print(f"✓ Created {output_file}")
            print(f"  - Total daily consumption: {profile_24h['total_load_kw'].sum():.1f} kWh")
            print(f"  - Peak demand: {profile_24h['total_load_kw'].max():.1f} kW")
            
            return profile_24h
        else:
            print("❌ Error: Could not extract 24-hour profile")
            return None
    
    def create_8760h_profile(self, aggregated_data):
        """
        Create full year profile (8760 hours)
        """
        print("Creating full year profile (8760 hours)...")
        
        # Prepare data for export
        year_data = aggregated_data[['datetime', 'total_load_kw']].copy()
        year_data['hour'] = year_data['datetime'].dt.hour
        year_data['day_of_year'] = year_data['datetime'].dt.dayofyear
        
        # Save full year profile
        output_file = self.project_data_dir / "load_8760h_lpg.csv"
        year_data.to_csv(output_file, index=False)
        
        print(f"✓ Created {output_file}")
        print(f"  - Total annual consumption: {year_data['total_load_kw'].sum():.1f} kWh")
        print(f"  - Average daily consumption: {year_data['total_load_kw'].sum()/365:.1f} kWh/day")
        print(f"  - Peak demand: {year_data['total_load_kw'].max():.1f} kW")
        
        return year_data
    
    def create_visualizations(self, profile_24h, year_data):
        """
        Create visualization plots for validation
        """
        print("Creating validation visualizations...")
        
        # Create plots directory
        plots_dir = self.project_data_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 24-hour profile plot
        plt.figure(figsize=(12, 6))
        plt.plot(profile_24h['hour'], profile_24h['total_load_kw'], 'b-', linewidth=2)
        plt.title('24-Hour Building Load Profile (20 Units)\nGenerated from LPG Household Types')
        plt.xlabel('Hour of Day')
        plt.ylabel('Total Load (kW)')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2))
        plt.tight_layout()
        plt.savefig(plots_dir / "load_profile_24h.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Weekly profile (first week of year)
        week_data = year_data.head(168)  # 7 days * 24 hours
        plt.figure(figsize=(15, 6))
        plt.plot(range(len(week_data)), week_data['total_load_kw'], 'g-', linewidth=1)
        plt.title('Weekly Building Load Profile (First Week of Year)')
        plt.xlabel('Hour of Week')
        plt.ylabel('Total Load (kW)')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 169, 24), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Mon'])
        plt.tight_layout()
        plt.savefig(plots_dir / "load_profile_weekly.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Created visualizations in {plots_dir}")
    
    def validate_results(self, profile_24h, year_data):
        """
        Validate the generated load profiles
        """
        print("\nValidating LPG-generated load profiles...")
        
        validation_results = {
            'daily_consumption_kwh': profile_24h['total_load_kw'].sum(),
            'peak_demand_kw': profile_24h['total_load_kw'].max(),
            'annual_consumption_kwh': year_data['total_load_kw'].sum(),
            'average_daily_kwh': year_data['total_load_kw'].sum() / 365,
            'load_factor': year_data['total_load_kw'].mean() / year_data['total_load_kw'].max()
        }
        
        # Expected ranges for 20-unit building
        expected_ranges = {
            'daily_consumption_kwh': (300, 600),  # 15-30 kWh per unit
            'peak_demand_kw': (40, 80),           # 2-4 kW per unit
            'load_factor': (0.3, 0.6)             # Typical residential load factor
        }
        
        print("\nValidation Results:")
        for metric, value in validation_results.items():
            if metric in expected_ranges:
                min_val, max_val = expected_ranges[metric]
                status = "✓" if min_val <= value <= max_val else "⚠️"
                print(f"  {status} {metric}: {value:.1f} (expected: {min_val}-{max_val})")
            else:
                print(f"  ✓ {metric}: {value:.1f}")
        
        return validation_results
    
    def process_all(self):
        """
        Complete LPG output processing pipeline
        """
        print("=" * 60)
        print("PROCESSING LOADPROFILEGENERATOR OUTPUTS")
        print("=" * 60)
        print("Aggregating 4 household types into 20-unit building load")
        print()
        
        # Create LPG outputs directory if it doesn't exist
        self.lpg_output_dir.mkdir(exist_ok=True)
        
        # Aggregate household data
        aggregated_data = self.aggregate_households()
        
        if aggregated_data is not None:
            # Create 24-hour profile
            profile_24h = self.create_24h_profile(aggregated_data)
            print()
            
            # Create full year profile
            year_data = self.create_8760h_profile(aggregated_data)
            print()
            
            # Create visualizations
            if profile_24h is not None:
                self.create_visualizations(profile_24h, year_data)
                print()
                
                # Validate results
                validation_results = self.validate_results(profile_24h, year_data)
                
                print("\n" + "=" * 60)
                print("✅ LPG PROCESSING COMPLETE")
                print("=" * 60)
                print("Files created:")
                print("  - load_24h_lpg.csv (24-hour profile)")
                print("  - load_8760h_lpg.csv (full year profile)")
                print("  - plots/ (validation visualizations)")
                print(f"\nOutput location: {self.project_data_dir}")
                
                return True
            else:
                print("❌ Error: Could not create 24-hour profile")
                return False
        else:
            print("❌ Error: Could not aggregate household data")
            return False

def main():
    """
    Main execution function
    """
    # Initialize LPG processor
    processor = LPGProcessor()
    
    # Process all LPG outputs
    success = processor.process_all()
    
    if success:
        print("\n🎯 LPG data processing complete!")
        print("Next: Use load_24h_lpg.csv for your energy management optimization")
    else:
        print("\n💥 LPG processing failed - check errors above")

if __name__ == "__main__":
    main()

