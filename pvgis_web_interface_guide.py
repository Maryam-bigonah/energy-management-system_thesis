#!/usr/bin/env python3
"""
PVGIS Web Interface Guide for Turin, Italy
Step-by-step instructions to get real PVGIS data using the web interface
"""

def print_pvgis_web_guide():
    """
    Print detailed instructions for using PVGIS web interface
    """
    print("=" * 80)
    print("PVGIS WEB INTERFACE GUIDE - TURIN, ITALY")
    print("=" * 80)
    print()
    print("🌞 Since the PVGIS API is complex, here are step-by-step instructions")
    print("   to manually extract REAL PVGIS data using the web interface:")
    print()
    
    print("📋 STEP 1: Access PVGIS Web Interface")
    print("   1. Go to: https://re.jrc.ec.europa.eu/pvg_tools/en/")
    print("   2. You'll see the PVGIS interface with multiple options")
    print("   3. Select 'HOURLY DATA' from the menu")
    print()
    
    print("📋 STEP 2: Configure Location (Turin, Italy)")
    print("   Location: Turin, Italy")
    print("   Latitude: 45.0703°N")
    print("   Longitude: 7.6869°E")
    print("   (You can search for 'Turin' in the location field)")
    print()
    
    print("📋 STEP 3: Configure Data Request")
    print("   Solar radiation database: PVGIS-SARAH2")
    print("   Start year: 2005")
    print("   End year: 2023")
    print("   (This gives you 19 years of data as requested)")
    print()
    
    print("📋 STEP 4: Configure PV System")
    print("   Mounting type: Fixed")
    print("   Peak power: 120 kWp")
    print("   System loss: 14%")
    print("   Mounting position: Free-standing")
    print("   Tilt angle: 30°")
    print("   Azimuth angle: 180° (South-facing)")
    print("   PV technology: Crystalline Silicon")
    print()
    
    print("📋 STEP 5: Configure Output")
    print("   Output format: CSV")
    print("   Include: PV power output")
    print("   (Make sure to select CSV format)")
    print()
    
    print("📋 STEP 6: Download Data")
    print("   1. Click 'Calculate' button")
    print("   2. Wait for calculation to complete (may take a few minutes)")
    print("   3. Download the CSV file")
    print("   4. Save as: 'pvgis_turin_2005_2023.csv'")
    print()
    
    print("📋 STEP 7: Process Data")
    print("   After downloading, run this script to process the CSV:")
    print("   python3 process_pvgis_turin_csv.py pvgis_turin_2005_2023.csv")
    print()
    
    print("=" * 80)
    print("ALTERNATIVE: Use PVGIS Grid-Connected PV Tool")
    print("=" * 80)
    print()
    print("🌐 You can also use the 'PERFORMANCE OF GRID-CONNECTED PV' tool:")
    print("   1. Select 'PERFORMANCE OF GRID-CONNECTED PV'")
    print("   2. Configure the same parameters as above")
    print("   3. This will give you yearly and monthly data")
    print("   4. You can also get hourly data from this tool")
    print()
    
    print("📊 Expected output from PVGIS:")
    print("   - 19 years of hourly data (2005-2023)")
    print("   - Power output in W (we'll convert to kW)")
    print("   - Real solar generation data for Turin")
    print("   - Proper seasonal and daily variations")
    print()
    
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)
    print()
    print("✅ Real PVGIS data for Turin should show:")
    print("   - Annual generation: ~120,000-180,000 kWh")
    print("   - Peak generation: ~100-120 kW")
    print("   - Capacity factor: ~15-20%")
    print("   - Zero generation at night (hours 0-5, 20-23)")
    print("   - Peak generation around noon (hours 11-13)")
    print("   - Higher generation in summer months")
    print("   - Lower generation in winter months")
    print()

def create_pvgis_turin_processor():
    """
    Create a script to process downloaded PVGIS CSV files for Turin
    """
    processor_code = '''#!/usr/bin/env python3
"""
Process PVGIS CSV Data for Turin, Italy
Converts downloaded PVGIS CSV to our required format
"""

import pandas as pd
import sys
import os
import numpy as np

def process_pvgis_turin_csv(csv_file):
    """
    Process downloaded PVGIS CSV file for Turin
    """
    print(f"Processing PVGIS CSV for Turin: {csv_file}")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        print(f"Columns: {list(df.columns)}")
        print(f"Total rows: {len(df)}")
        
        # Find power column (could be P, P_dc, P_ac, etc.)
        power_col = None
        for col in df.columns:
            if 'P' in col.upper() and 'POWER' not in col.upper():
                power_col = col
                break
        
        if power_col is None:
            print("❌ No power column found")
            return False
        
        print(f"Using power column: {power_col}")
        
        # Convert from W to kW
        df['pv_generation_kw'] = df[power_col] / 1000
        
        # Handle time column
        time_col = None
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                time_col = col
                break
        
        if time_col:
            try:
                df['timestamp'] = pd.to_datetime(df[time_col])
                print(f"✅ Time column parsed: {time_col}")
            except:
                print("⚠️ Could not parse time column")
        
        # Create hour column (1-8760 for single year, or sequential for multi-year)
        if len(df) == 8760:
            # Single year data
            df['hour'] = range(1, 8761)
            print("✅ Single year data detected (8760 hours)")
        elif len(df) > 8760:
            # Multi-year data - take first year
            df = df.head(8760)
            df['hour'] = range(1, 8761)
            print(f"✅ Multi-year data detected, using first year (8760 hours)")
        else:
            # Less than a year - pad with zeros or use as is
            df['hour'] = range(1, len(df) + 1)
            print(f"⚠️ Less than full year data: {len(df)} hours")
        
        # Save processed data
        data_dir = "project/data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Save full year
        pv_8760_file = f"{data_dir}/pv_8760_real.csv"
        df[['hour', 'pv_generation_kw']].to_csv(pv_8760_file, index=False)
        print(f"✅ Saved: {pv_8760_file}")
        
        # Save 24-hour profile (summer day)
        day_180_start = 179 * 24
        day_180_end = day_180_start + 24
        
        if day_180_end <= len(df):
            profile_24h = df.iloc[day_180_start:day_180_end].copy()
            profile_24h['hour'] = range(1, 25)
            
            pv_24h_file = f"{data_dir}/pv_24h_real.csv"
            profile_24h[['hour', 'pv_generation_kw']].to_csv(pv_24h_file, index=False)
            print(f"✅ Saved: {pv_24h_file}")
        
        # Print statistics
        total_annual = df['pv_generation_kw'].sum()
        peak_generation = df['pv_generation_kw'].max()
        capacity_factor = total_annual / (120 * 8760)
        daily_average = total_annual / 365
        
        print(f"📊 Statistics for Turin:")
        print(f"   Annual generation: {total_annual:.0f} kWh")
        print(f"   Daily average: {daily_average:.1f} kWh/day")
        print(f"   Peak generation: {peak_generation:.1f} kW")
        print(f"   Capacity factor: {capacity_factor * 100:.1f}%")
        
        # Validate data quality
        night_hours = df[df['hour'].isin([1, 2, 3, 4, 5, 22, 23, 24])]['pv_generation_kw']
        if night_hours.max() < 0.1:
            print("✅ Night-time generation is realistic (near zero)")
        else:
            print("⚠️ Night-time generation may be unrealistic")
        
        peak_hours = df[df['hour'].isin([11, 12, 13])]['pv_generation_kw']
        if peak_hours.mean() > daily_average * 0.8:
            print("✅ Peak hours show realistic solar pattern")
        else:
            print("⚠️ Peak hours may not show realistic solar pattern")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing CSV: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 process_pvgis_turin_csv.py <pvgis_csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)
    
    success = process_pvgis_turin_csv(csv_file)
    if success:
        print("✅ PVGIS data for Turin processed successfully!")
        print("📁 Files created:")
        print("   - project/data/pv_8760_real.csv")
        print("   - project/data/pv_24h_real.csv")
    else:
        print("❌ Failed to process PVGIS data")
        sys.exit(1)
'''
    
    with open('process_pvgis_turin_csv.py', 'w') as f:
        f.write(processor_code)
    
    print("✅ Created: process_pvgis_turin_csv.py")
    print("   Use this script to process downloaded PVGIS CSV files for Turin")

if __name__ == "__main__":
    print_pvgis_web_guide()
    print()
    create_pvgis_turin_processor()

