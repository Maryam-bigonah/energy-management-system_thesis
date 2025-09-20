#!/usr/bin/env python3
"""
Aggregate Family Load Profiles to 20-Unit Building Load
Performs hour-by-hour aggregation using specified family distribution
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class BuildingLoadAggregator:
    def __init__(self, data_dir="project/data"):
        self.data_dir = data_dir
        
        # Family distribution for 20 units
        self.family_distribution = {
            'familyA': 6,  # Working couple with appliances
            'familyB': 4,  # Mixed work couple
            'familyC': 5,  # Family with children
            'familyD': 5   # Elderly couple
        }
        
        # Verify total units
        total_units = sum(self.family_distribution.values())
        if total_units != 20:
            raise ValueError(f"Total units must be 20, got {total_units}")
        
        print(f"Family distribution: {self.family_distribution}")
        print(f"Total units: {total_units}")
    
    def load_family_data(self):
        """
        Load all family CSV files
        """
        print("Loading family load profiles...")
        
        family_data = {}
        for family_key in ['familyA', 'familyB', 'familyC', 'familyD']:
            filename = f"{self.data_dir}/{family_key}.csv"
            
            if not os.path.exists(filename):
                raise FileNotFoundError(f"Family data file not found: {filename}")
            
            df = pd.read_csv(filename)
            
            # Validate data structure
            if len(df) != 8760:
                raise ValueError(f"{family_key}.csv should have 8760 rows, got {len(df)}")
            
            if 'consumption_kw' not in df.columns:
                raise ValueError(f"{family_key}.csv missing 'consumption_kw' column")
            
            family_data[family_key] = df
            print(f"  ✓ Loaded {family_key}: {len(df)} hours, {df['consumption_kw'].sum():.0f} kWh/year")
        
        return family_data
    
    def aggregate_building_load(self, family_data):
        """
        Aggregate family loads into building load using the formula:
        Load_building[t] = 6×LA[t] + 4×LB[t] + 5×LC[t] + 5×LD[t]
        """
        print("Aggregating building load...")
        
        # Initialize building load array
        building_load = np.zeros(8760)
        
        # Aggregate each family type
        for family_key, count in self.family_distribution.items():
            family_consumption = family_data[family_key]['consumption_kw'].values
            building_load += family_consumption * count
            
            print(f"  ✓ Added {count}×{family_key}: {family_consumption.sum() * count:.0f} kWh/year")
        
        # Create timestamps for the full year
        timestamps = pd.date_range('2024-01-01', periods=8760, freq='H')
        
        # Create aggregated DataFrame
        building_df = pd.DataFrame({
            'hour': range(1, 8761),  # 1-8760
            'timestamp': timestamps,
            'load_kw': building_load
        })
        
        # Calculate statistics
        total_annual = building_load.sum()
        daily_average = total_annual / 365
        peak_power = building_load.max()
        peak_hour = building_load.argmax() + 1  # Convert to 1-based hour
        
        print(f"  ✓ Building total annual consumption: {total_annual:.0f} kWh")
        print(f"  ✓ Building daily average: {daily_average:.1f} kWh/day")
        print(f"  ✓ Building peak demand: {peak_power:.1f} kW at hour {peak_hour}")
        
        return building_df
    
    def save_load_8760(self, building_df):
        """
        Save full year building load as load_8760.csv
        """
        filename = f"{self.data_dir}/load_8760.csv"
        
        # Save with hour and load columns
        output_df = building_df[['hour', 'load_kw']].copy()
        output_df.to_csv(filename, index=False)
        
        print(f"✓ Saved full year profile: {filename}")
        return filename
    
    def extract_24h_profile(self, building_df):
        """
        Extract representative 24-hour profile
        Use day 100 (around April 10th) - good representative day
        """
        print("Extracting 24-hour representative profile...")
        
        # Day 100 is around April 10th - good spring day
        day_100_start = 99 * 24  # 0-based indexing
        day_100_end = day_100_start + 24
        
        profile_24h = building_df.iloc[day_100_start:day_100_end].copy()
        
        # Reset hour to 0-23 for 24-hour profile
        profile_24h['hour'] = range(24)
        
        # Calculate 24-hour statistics
        daily_total = profile_24h['load_kw'].sum()
        peak_power = profile_24h['load_kw'].max()
        peak_hour = profile_24h['load_kw'].idxmax() - day_100_start  # Convert to 0-23
        
        print(f"  ✓ 24-hour total consumption: {daily_total:.1f} kWh")
        print(f"  ✓ 24-hour peak demand: {peak_power:.1f} kW at hour {peak_hour}")
        
        return profile_24h
    
    def save_load_24h(self, profile_24h):
        """
        Save 24-hour building load as load_24h.csv
        """
        filename = f"{self.data_dir}/load_24h.csv"
        
        # Save with hour and load columns
        output_df = profile_24h[['hour', 'load_kw']].copy()
        output_df.to_csv(filename, index=False)
        
        print(f"✓ Saved 24-hour profile: {filename}")
        return filename
    
    def create_validation_plots(self, building_df, profile_24h):
        """
        Create validation plots for the aggregated building load
        """
        print("Creating validation plots...")
        
        plots_dir = f"{self.data_dir}/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot 1: 24-hour profile
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(profile_24h['hour'], profile_24h['load_kw'], 'b-', linewidth=2)
        plt.title('24-Hour Building Load Profile (20 Units)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Building Load (kW)')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2))
        
        # Plot 2: Weekly profile (first week)
        plt.subplot(2, 2, 2)
        week_data = building_df.iloc[:168]  # First week
        plt.plot(range(168), week_data['load_kw'], 'g-', linewidth=1)
        plt.title('Weekly Building Load Profile (First Week)')
        plt.xlabel('Hour of Week')
        plt.ylabel('Building Load (kW)')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 169, 24), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Mon'])
        
        # Plot 3: Monthly averages
        plt.subplot(2, 2, 3)
        building_df['month'] = building_df['timestamp'].dt.month
        monthly_avg = building_df.groupby('month')['load_kw'].mean()
        plt.plot(monthly_avg.index, monthly_avg.values, 'r-o', linewidth=2)
        plt.title('Monthly Average Building Load')
        plt.xlabel('Month')
        plt.ylabel('Average Load (kW)')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, 13))
        
        # Plot 4: Family contribution breakdown
        plt.subplot(2, 2, 4)
        family_contributions = []
        family_names = []
        
        # Calculate annual contribution of each family type
        for family_key, count in self.family_distribution.items():
            # Load family data to calculate contribution
            family_df = pd.read_csv(f"{self.data_dir}/{family_key}.csv")
            annual_contribution = family_df['consumption_kw'].sum() * count
            family_contributions.append(annual_contribution)
            family_names.append(f"{count}×{family_key}")
        
        bars = plt.bar(range(len(family_names)), family_contributions)
        plt.title('Annual Consumption by Family Type')
        plt.xlabel('Family Type')
        plt.ylabel('Annual Consumption (kWh)')
        plt.xticks(range(len(family_names)), family_names, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, family_contributions):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                    f'{value:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/building_load_aggregation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Created validation plots in {plots_dir}")
    
    def validate_building_load(self, building_df, profile_24h):
        """
        Validate aggregated building load against expected ranges
        """
        print("Validating aggregated building load...")
        
        # Calculate key metrics
        annual_total = building_df['load_kw'].sum()
        daily_average = annual_total / 365
        peak_power = building_df['load_kw'].max()
        load_factor = building_df['load_kw'].mean() / peak_power
        
        # Expected ranges for 20-unit building
        expected_ranges = {
            'annual_consumption_kwh': (80000, 120000),  # 4-6 MWh per unit
            'daily_average_kwh': (220, 330),            # 11-16.5 kWh per unit
            'peak_demand_kw': (100, 200),               # 5-10 kW per unit
            'load_factor': (0.25, 0.45)                 # Typical residential
        }
        
        print("\nValidation Results:")
        validation_passed = True
        
        for metric, (min_val, max_val) in expected_ranges.items():
            if metric == 'annual_consumption_kwh':
                value = annual_total
            elif metric == 'daily_average_kwh':
                value = daily_average
            elif metric == 'peak_demand_kw':
                value = peak_power
            elif metric == 'load_factor':
                value = load_factor
            
            status = "✓" if min_val <= value <= max_val else "⚠️"
            if not (min_val <= value <= max_val):
                validation_passed = False
            
            print(f"  {status} {metric}: {value:.1f} (expected: {min_val}-{max_val})")
        
        # Additional validation: Check family distribution
        print(f"\nFamily Distribution Validation:")
        for family_key, count in self.family_distribution.items():
            family_df = pd.read_csv(f"{self.data_dir}/{family_key}.csv")
            family_annual = family_df['consumption_kw'].sum() * count
            family_daily = family_annual / 365
            print(f"  ✓ {count}×{family_key}: {family_annual:.0f} kWh/year, {family_daily:.1f} kWh/day")
        
        return validation_passed
    
    def aggregate_all(self):
        """
        Complete aggregation pipeline
        """
        print("=" * 60)
        print("AGGREGATING FAMILY LOADS TO 20-UNIT BUILDING")
        print("=" * 60)
        print("Formula: Load_building[t] = 6×LA[t] + 4×LB[t] + 5×LC[t] + 5×LD[t]")
        print()
        
        try:
            # Load family data
            family_data = self.load_family_data()
            print()
            
            # Aggregate building load
            building_df = self.aggregate_building_load(family_data)
            print()
            
            # Save full year profile
            load_8760_file = self.save_load_8760(building_df)
            print()
            
            # Extract and save 24-hour profile
            profile_24h = self.extract_24h_profile(building_df)
            load_24h_file = self.save_load_24h(profile_24h)
            print()
            
            # Create validation plots
            self.create_validation_plots(building_df, profile_24h)
            print()
            
            # Validate results
            validation_passed = self.validate_building_load(building_df, profile_24h)
            print()
            
            # Summary
            print("=" * 60)
            if validation_passed:
                print("✅ BUILDING LOAD AGGREGATION SUCCESSFUL")
            else:
                print("⚠️ BUILDING LOAD AGGREGATION COMPLETED WITH WARNINGS")
            print("=" * 60)
            print("Files created:")
            print(f"  - {load_8760_file} (full year: 8760 hours)")
            print(f"  - {load_24h_file} (representative day: 24 hours)")
            print(f"  - {self.data_dir}/plots/ (validation visualizations)")
            print(f"\nOutput location: {self.data_dir}")
            print("\n🎯 Ready for energy management optimization!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during aggregation: {str(e)}")
            return False

def main():
    """
    Main execution function
    """
    # Initialize aggregator
    aggregator = BuildingLoadAggregator()
    
    # Perform aggregation
    success = aggregator.aggregate_all()
    
    if success:
        print("\n🎯 Building load aggregation complete!")
        print("Use load_24h.csv for daily optimization or load_8760.csv for annual analysis")
    else:
        print("\n💥 Aggregation failed - check errors above")

if __name__ == "__main__":
    main()

