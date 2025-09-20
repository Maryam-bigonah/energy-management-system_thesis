#!/usr/bin/env python3
"""
Generate Italian ARERA TOU Pricing Data
Creates proper F1/F2/F3 tariff structure for Italian energy market
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_italian_tou_data():
    """
    Generate Italian ARERA TOU pricing data for full year
    Based on F1/F2/F3 tariff structure
    """
    print("Generating Italian ARERA TOU pricing data...")
    print("Structure: F1 (peak), F2 (flat), F3 (valley)")
    
    # Italian ARERA tariff bands (€/kWh)
    # These are realistic values - can be updated with official ARERA rates
    tariffs = {
        'F1_peak': 0.48,    # F1 - Peak hours (most expensive)
        'F2_flat': 0.34,    # F2 - Flat hours (mid-price)
        'F3_valley': 0.24,  # F3 - Valley hours (cheapest)
        'feed_in_tariff': 0.10  # Feed-in tariff (Scambio sul Posto)
    }
    
    # Hourly pricing for different day types
    def get_weekday_pricing():
        """Weekday (Mon-Fri) pricing structure"""
        pricing = []
        for hour in range(1, 25):  # 1-24 hours
            if hour in range(8, 20):  # 08:00-19:00
                price_buy = tariffs['F1_peak']
            elif hour in [7] or hour in range(19, 23):  # 07:00-08:00 and 19:00-23:00
                price_buy = tariffs['F2_flat']
            else:  # 23:00-07:00
                price_buy = tariffs['F3_valley']
            
            pricing.append({
                'hour': hour,
                'price_buy': price_buy,
                'price_sell': tariffs['feed_in_tariff']
            })
        return pricing
    
    def get_saturday_pricing():
        """Saturday pricing structure (no F1 peak)"""
        pricing = []
        for hour in range(1, 25):  # 1-24 hours
            if hour in range(7, 23):  # 07:00-23:00
                price_buy = tariffs['F2_flat']
            else:  # 23:00-07:00
                price_buy = tariffs['F3_valley']
            
            pricing.append({
                'hour': hour,
                'price_buy': price_buy,
                'price_sell': tariffs['feed_in_tariff']
            })
        return pricing
    
    def get_sunday_pricing():
        """Sunday/holiday pricing structure (F3 all day)"""
        pricing = []
        for hour in range(1, 25):  # 1-24 hours
            pricing.append({
                'hour': hour,
                'price_buy': tariffs['F3_valley'],
                'price_sell': tariffs['feed_in_tariff']
            })
        return pricing
    
    # Generate full year data
    yearly_data = []
    timestamps = []
    
    start_date = datetime(2024, 1, 1)
    
    for day in range(365):
        current_date = start_date + timedelta(days=day)
        day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday
        
        # Determine day type
        if day_of_week < 5:  # Monday-Friday
            day_pricing = get_weekday_pricing()
        elif day_of_week == 5:  # Saturday
            day_pricing = get_saturday_pricing()
        else:  # Sunday
            day_pricing = get_sunday_pricing()
        
        # Add to yearly data
        for hour_data in day_pricing:
            yearly_data.append(hour_data)
            timestamps.append(current_date + timedelta(hours=hour_data['hour'] - 1))
    
    # Create DataFrame
    df = pd.DataFrame(yearly_data)
    df['timestamp'] = timestamps
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_type'] = df['day_of_week'].map({
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday',
        5: 'Saturday', 6: 'Sunday'
    })
    
    # Calculate statistics
    avg_price_buy = df['price_buy'].mean()
    min_price_buy = df['price_buy'].min()
    max_price_buy = df['price_buy'].max()
    
    print(f"  ✓ Generated 8760 hours of TOU data")
    print(f"  ✓ Average buy price: €{avg_price_buy:.3f}/kWh")
    print(f"  ✓ Price range: €{min_price_buy:.3f} - €{max_price_buy:.3f}/kWh")
    print(f"  ✓ Feed-in tariff: €{tariffs['feed_in_tariff']:.3f}/kWh")
    
    return df

def save_tou_files(df):
    """
    Save TOU data to CSV files
    """
    data_dir = "project/data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save full year profile
    tou_8760_file = f"{data_dir}/tou_8760.csv"
    df[['hour', 'price_buy', 'price_sell']].to_csv(tou_8760_file, index=False)
    print(f"✓ Saved full year TOU profile: {tou_8760_file}")
    
    # Extract weekday 24-hour profile (Monday)
    weekday_data = df[df['day_of_week'] == 0].head(24)  # First Monday
    weekday_data['hour'] = range(1, 25)
    
    tou_24h_file = f"{data_dir}/tou_24h.csv"
    weekday_data[['hour', 'price_buy', 'price_sell']].to_csv(tou_24h_file, index=False)
    print(f"✓ Saved 24-hour TOU profile (weekday): {tou_24h_file}")
    
    # Create Saturday profile
    saturday_data = df[df['day_of_week'] == 5].head(24)  # First Saturday
    saturday_data['hour'] = range(1, 25)
    
    tou_saturday_file = f"{data_dir}/tou_saturday.csv"
    saturday_data[['hour', 'price_buy', 'price_sell']].to_csv(tou_saturday_file, index=False)
    print(f"✓ Saved Saturday TOU profile: {tou_saturday_file}")
    
    # Create Sunday profile
    sunday_data = df[df['day_of_week'] == 6].head(24)  # First Sunday
    sunday_data['hour'] = range(1, 25)
    
    tou_sunday_file = f"{data_dir}/tou_sunday.csv"
    sunday_data[['hour', 'price_buy', 'price_sell']].to_csv(tou_sunday_file, index=False)
    print(f"✓ Saved Sunday TOU profile: {tou_sunday_file}")
    
    return tou_8760_file, tou_24h_file, tou_saturday_file, tou_sunday_file

def create_tariff_summary(df):
    """
    Create summary of tariff structure
    """
    print("\nItalian ARERA Tariff Structure:")
    print("=" * 40)
    
    # Weekday analysis
    weekday_data = df[df['day_of_week'] < 5]
    weekday_summary = weekday_data.groupby('price_buy').size()
    
    print("Weekday (Mon-Fri) Structure:")
    for price, count in weekday_summary.items():
        if price == 0.48:
            band = "F1 (Peak)"
        elif price == 0.34:
            band = "F2 (Flat)"
        else:
            band = "F3 (Valley)"
        print(f"  {band}: €{price:.3f}/kWh - {count} hours")
    
    # Weekend analysis
    weekend_data = df[df['day_of_week'] >= 5]
    weekend_summary = weekend_data.groupby('price_buy').size()
    
    print("\nWeekend Structure:")
    for price, count in weekend_summary.items():
        if price == 0.34:
            band = "F2 (Flat)"
        else:
            band = "F3 (Valley)"
        print(f"  {band}: €{price:.3f}/kWh - {count} hours")
    
    print(f"\nFeed-in Tariff: €{df['price_sell'].iloc[0]:.3f}/kWh (all hours)")

def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("GENERATING ITALIAN ARERA TOU PRICING DATA")
    print("=" * 60)
    print("Creating F1/F2/F3 tariff structure for Italian energy market")
    print()
    
    try:
        # Generate TOU data
        tou_df = generate_italian_tou_data()
        print()
        
        # Save files
        files = save_tou_files(tou_df)
        print()
        
        # Create summary
        create_tariff_summary(tou_df)
        print()
        
        # Summary
        print("=" * 60)
        print("✅ ITALIAN TOU PRICING DATA GENERATED SUCCESSFULLY")
        print("=" * 60)
        print("Files created:")
        for file in files:
            print(f"  - {file}")
        print("\n🔍 Tariff structure matches Italian ARERA F1/F2/F3 bands")
        print("📊 Ready for energy optimization algorithms")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during TOU data generation: {str(e)}")
        return False

if __name__ == "__main__":
    main()

