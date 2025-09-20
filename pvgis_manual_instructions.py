#!/usr/bin/env python3
"""
PVGIS Manual Data Extraction Instructions
Since the API is complex, this provides instructions for manual data extraction
"""

def print_pvgis_instructions():
    """
    Print detailed instructions for manually extracting PVGIS data
    """
    print("=" * 80)
    print("PVGIS MANUAL DATA EXTRACTION INSTRUCTIONS")
    print("=" * 80)
    print()
    print("🌞 Since the PVGIS API is complex, here are step-by-step instructions")
    print("   to manually extract REAL PVGIS data for Turin, Italy:")
    print()
    
    print("📋 STEP 1: Access PVGIS")
    print("   1. Go to: https://re.jrc.ec.europa.eu/pvg_tools/en/")
    print("   2. Click on 'PVGIS' (Photovoltaic Geographical Information System)")
    print("   3. Select 'Hourly data' from the menu")
    print()
    
    print("📋 STEP 2: Configure Location")
    print("   Location: Turin, Italy")
    print("   Latitude: 45.0703°N")
    print("   Longitude: 7.6869°E")
    print("   (You can also search for 'Turin' in the location field)")
    print()
    
    print("📋 STEP 3: Configure PV System")
    print("   Peak power: 120 kWp")
    print("   System loss: 14%")
    print("   Mounting position: Free-standing")
    print("   Tilt angle: 30°")
    print("   Azimuth angle: 180° (South-facing)")
    print("   Technology: Crystalline silicon")
    print()
    
    print("📋 STEP 4: Configure Data Request")
    print("   Start year: 2020")
    print("   End year: 2020")
    print("   Output format: CSV file")
    print("   Include: P (power output)")
    print()
    
    print("📋 STEP 5: Download Data")
    print("   1. Click 'Calculate'")
    print("   2. Wait for calculation to complete")
    print("   3. Download the CSV file")
    print("   4. Save as: 'pvgis_turin_2020.csv'")
    print()
    
    print("📋 STEP 6: Process Data")
    print("   After downloading, run this script to process the CSV:")
    print("   python3 process_pvgis_csv.py pvgis_turin_2020.csv")
    print()
    
    print("=" * 80)
    print("ALTERNATIVE: Use PVGIS Web Interface")
    print("=" * 80)
    print()
    print("🌐 Direct link to PVGIS hourly data:")
    print("   https://re.jrc.ec.europa.eu/pvg_tools/en/tools.html#hourly")
    print()
    print("📊 Expected output:")
    print("   - 8760 hours of data (1 year)")
    print("   - Power output in W (we'll convert to kW)")
    print("   - Real solar generation data for Turin")
    print()
    
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)
    print()
    print("✅ Real PVGIS data should show:")
    print("   - Annual generation: ~120,000-180,000 kWh")
    print("   - Peak generation: ~100-120 kW")
    print("   - Capacity factor: ~15-20%")
    print("   - Zero generation at night (hours 0-5, 20-23)")
    print("   - Peak generation around noon (hours 11-13)")
    print()

def create_pvgis_processor():
    """
    Create a script to process downloaded PVGIS CSV files
    """
    processor_code = '''#!/usr/bin/env python3
"""
Process PVGIS CSV Data
Converts downloaded PVGIS CSV to our required format
"""

import pandas as pd
import sys
import os

def process_pvgis_csv(csv_file):
    """
    Process downloaded PVGIS CSV file
    """
    print(f"Processing PVGIS CSV: {csv_file}")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        print(f"Columns: {list(df.columns)}")
        
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
        
        # Create hour column (1-8760)
        df['hour'] = range(1, len(df) + 1)
        
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
        
        print(f"📊 Statistics:")
        print(f"   Annual generation: {total_annual:.0f} kWh")
        print(f"   Peak generation: {peak_generation:.1f} kW")
        print(f"   Capacity factor: {capacity_factor * 100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing CSV: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 process_pvgis_csv.py <pvgis_csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)
    
    success = process_pvgis_csv(csv_file)
    if success:
        print("✅ PVGIS data processed successfully!")
    else:
        print("❌ Failed to process PVGIS data")
        sys.exit(1)
'''
    
    with open('process_pvgis_csv.py', 'w') as f:
        f.write(processor_code)
    
    print("✅ Created: process_pvgis_csv.py")
    print("   Use this script to process downloaded PVGIS CSV files")

if __name__ == "__main__":
    print_pvgis_instructions()
    print()
    create_pvgis_processor()

