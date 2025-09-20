#!/usr/bin/env python3
"""
Fetch Real PVGIS Data for Turin, Italy
This script fetches actual solar generation data from PVGIS API
"""

import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta

def fetch_pvgis_data():
    """
    Fetch real PVGIS data for Turin, Italy
    Coordinates: 45.0703°N, 7.6869°E
    """
    print("🌞 Fetching REAL PVGIS data for Turin, Italy...")
    print("Location: 45.0703°N, 7.6869°E")
    print("System: 120 kWp, 30° tilt, 180° azimuth (South-facing)")
    
    # PVGIS API endpoint for hourly data
    base_url = "https://re.jrc.ec.europa.eu/api/v5_2/timeseries"
    
    # Parameters for Turin, Italy - Request hourly data
    params = {
        'lat': 45.0703,           # Latitude
        'lon': 7.6869,            # Longitude
        'peakpower': 120,         # 120 kWp system
        'loss': 14,               # 14% system losses
        'angle': 30,              # 30° tilt angle
        'aspect': 180,            # 180° azimuth (South-facing)
        'outputformat': 'json',   # JSON output
        'usehorizon': 1,          # Use horizon data
        'userhorizon': '',        # No custom horizon
        'raddatabase': 'PVGIS-SARAH2',  # Use SARAH2 database
        'startyear': 2020,        # Start year
        'endyear': 2020,          # End year (single year for hourly data)
        'pvcalculation': 1,       # PV calculation
        'pvtechchoice': 'crystSi', # Crystalline silicon
        'mountingplace': 'free',  # Free-standing
        'trackingtype': 0,        # Fixed mounting
        'optimalinclination': 0,  # Use specified inclination
        'optimalangles': 0,       # Use specified angles
        'components': 1,          # Include components
        'hourly': 1               # Request hourly data
    }
    
    try:
        print("📡 Connecting to PVGIS API...")
        response = requests.get(base_url, params=params, timeout=30)
        
        if response.status_code == 200:
            print("✅ Successfully connected to PVGIS API")
            data = response.json()
            
            # Debug: Print response structure
            print(f"📊 Response keys: {list(data.keys())}")
            if 'outputs' in data:
                print(f"📊 Outputs keys: {list(data['outputs'].keys())}")
            
            # Extract hourly data - try different possible structures
            hourly_data = None
            if 'outputs' in data and 'hourly' in data['outputs']:
                hourly_data = data['outputs']['hourly']
            elif 'hourly' in data:
                hourly_data = data['hourly']
            elif 'outputs' in data:
                # Check all possible output formats
                for key in data['outputs'].keys():
                    if 'hour' in key.lower() or 'time' in key.lower():
                        print(f"📊 Found potential hourly data in: {key}")
                        hourly_data = data['outputs'][key]
                        break
            
            if hourly_data:
                print(f"📊 Retrieved {len(hourly_data)} hours of data")
                
                # Convert to DataFrame
                df = pd.DataFrame(hourly_data)
                print(f"📊 DataFrame columns: {list(df.columns)}")
                
                # Convert time columns - try different formats
                if 'time' in df.columns:
                    try:
                        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')
                    except:
                        try:
                            df['time'] = pd.to_datetime(df['time'])
                        except:
                            print("⚠️ Could not parse time column")
                
                # Extract PV generation - try different column names
                pv_col = None
                for col in ['P', 'P_dc', 'P_ac', 'power', 'generation']:
                    if col in df.columns:
                        pv_col = col
                        break
                
                if pv_col:
                    # Convert from W to kW
                    df['pv_generation_kw'] = df[pv_col] / 1000
                    print(f"✅ Using column '{pv_col}' for PV generation")
                else:
                    print("❌ No PV generation column found")
                    return None
                
                # Calculate statistics
                total_annual = df['pv_generation_kw'].sum()
                daily_average = total_annual / 365
                peak_generation = df['pv_generation_kw'].max()
                capacity_factor = total_annual / (120 * 8760)
                
                print(f"📈 Real PVGIS Statistics:")
                print(f"   Annual generation: {total_annual:.0f} kWh")
                print(f"   Daily average: {daily_average:.1f} kWh/day")
                print(f"   Peak generation: {peak_generation:.1f} kW")
                print(f"   Capacity factor: {capacity_factor * 100:.1f}%")
                
                return df
            else:
                print("❌ No hourly data found in PVGIS response")
                return None
                
        else:
            print(f"❌ PVGIS API error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error connecting to PVGIS: {e}")
        return None
    except Exception as e:
        print(f"❌ Error processing PVGIS data: {e}")
        return None

def save_real_pv_data(df):
    """
    Save real PVGIS data to CSV files
    """
    if df is None:
        print("❌ No data to save")
        return False
    
    data_dir = "project/data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create hour column (1-8760)
    df['hour'] = range(1, len(df) + 1)
    
    # Save full year profile
    pv_8760_file = f"{data_dir}/pv_8760_real.csv"
    df[['hour', 'pv_generation_kw']].to_csv(pv_8760_file, index=False)
    print(f"✅ Saved real PVGIS data: {pv_8760_file}")
    
    # Extract 24-hour profile (summer day - day 180)
    day_180_start = 179 * 24  # 0-based indexing
    day_180_end = day_180_start + 24
    
    if day_180_end <= len(df):
        profile_24h = df.iloc[day_180_start:day_180_end].copy()
        profile_24h['hour'] = range(1, 25)  # 1-24 hours
        
        pv_24h_file = f"{data_dir}/pv_24h_real.csv"
        profile_24h[['hour', 'pv_generation_kw']].to_csv(pv_24h_file, index=False)
        print(f"✅ Saved 24-hour real PVGIS data: {pv_24h_file}")
        
        # Print 24-hour statistics
        daily_total = profile_24h['pv_generation_kw'].sum()
        peak_generation = profile_24h['pv_generation_kw'].max()
        peak_hour = profile_24h['pv_generation_kw'].idxmax() - day_180_start + 1
        
        print(f"📊 24-hour real data:")
        print(f"   Total generation: {daily_total:.1f} kWh")
        print(f"   Peak generation: {peak_generation:.1f} kW at hour {peak_hour}")
        
        return True
    else:
        print("❌ Not enough data for 24-hour profile")
        return False

def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("FETCHING REAL PVGIS DATA - TURIN, ITALY")
    print("=" * 60)
    print("Source: https://re.jrc.ec.europa.eu/pvg_tools/en/")
    print("API: PVGIS v5.2")
    print()
    
    try:
        # Fetch real PVGIS data
        pv_df = fetch_pvgis_data()
        
        if pv_df is not None:
            print()
            
            # Save files
            success = save_real_pv_data(pv_df)
            
            if success:
                print()
                print("=" * 60)
                print("✅ REAL PVGIS DATA FETCHED SUCCESSFULLY")
                print("=" * 60)
                print("Files created:")
                print("  - project/data/pv_8760_real.csv (full year)")
                print("  - project/data/pv_24h_real.csv (24 hours)")
                print("\n🔍 Data source: Real PVGIS API")
                print("📊 Location: Turin, Italy (45.0703°N, 7.6869°E)")
                print("⚡ System: 120 kWp, 30° tilt, South-facing")
                return True
            else:
                print("❌ Failed to save PVGIS data")
                return False
        else:
            print("❌ Failed to fetch PVGIS data")
            return False
            
    except Exception as e:
        print(f"❌ Error during PVGIS data fetch: {str(e)}")
        return False

if __name__ == "__main__":
    main()
