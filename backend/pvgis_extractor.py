import pandas as pd
import os
from typing import List, Dict
import json
import requests
import time

class PVDataExtractor:
    """
    Extract and process PV data for residential building community analysis
    Uses PVGIS API for real solar data from Torino, Italy
    
    Web Interface: https://re.jrc.ec.europa.eu/pvg_tools/en/
    API Documentation: https://re.jrc.ec.europa.eu/api/v5_2/
    """
    
    def __init__(self, data_path: str = "data/pv_daily.csv", use_pvgis: bool = True):
        self.data_path = data_path
        self.data = None
        self.use_pvgis = use_pvgis
        
        # Torino, Italy coordinates
        self.lat = 45.0703
        self.lon = 7.6869
        
        # PVGIS API base URL for hourly radiation data
        # Available endpoints:
        # - v5_2: Legacy API (2005-2020)
        # - v5_3: Latest API with 2023 data support
        # Available databases in v5_3:
        # - PVGIS-SARAH3: Updated satellite data
        # - PVGIS-ERA5: Reanalysis data
        self.pvgis_url = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
    
    def fetch_pvgis_data(self) -> pd.DataFrame:
        """Fetch real PV data from PVGIS API for Torino, Italy"""
        try:
            print("Fetching real PV data from PVGIS API for Torino, Italy...")
            
            # PVGIS API parameters for Torino - requesting hourly radiation data
            # Note: PVGIS 5.3 supports data from 2005-2023 (full range)
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'startyear': 2005,  # Full range: 2005-2023
                'endyear': 2023,
                'mountingplace': 'free',
                'angle': 35,       # Optimal tilt for Torino
                'aspect': 0,       # South-facing
                'outputformat': 'json',
                'raddatabase': 'PVGIS-SARAH3',  # Updated database with 2023 data
                'usehorizon': 1,
                'userhorizon': ''
            }
            
            response = requests.get(self.pvgis_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Debug: print the structure of the response
            print(f"PVGIS API response keys: {list(data.keys())}")
            if 'outputs' in data:
                print(f"Outputs keys: {list(data['outputs'].keys())}")
                if 'hourly' in data['outputs'] and len(data['outputs']['hourly']) > 0:
                    print(f"Sample hourly entry: {data['outputs']['hourly'][0]}")
                    print(f"Available fields in hourly data: {list(data['outputs']['hourly'][0].keys())}")
                    # Check a few more entries to see daytime values
                    for i in [100, 200, 300]:
                        if i < len(data['outputs']['hourly']):
                            print(f"Entry {i}: {data['outputs']['hourly'][i]}")
            
            if 'outputs' not in data:
                raise ValueError("Invalid response from PVGIS API")
            
            # Extract hourly radiation data
            if 'hourly' in data['outputs']:
                hourly_data = data['outputs']['hourly']
            else:
                raise ValueError("No hourly data found in PVGIS response")
            
            # Convert radiation to PV power using simple model
            # Assuming 1 kWp system with 14% losses and 15% efficiency
            pv_data = []
            for entry in hourly_data:
                # Extract hour from timestamp - PVGIS returns YYYYMMDD:HHMM format
                if 'time' in entry:
                    time_str = entry['time']
                    if ':' in time_str:
                        # Extract hour from HHMM format (e.g., "0010" -> 0)
                        hour_minute = time_str.split(':')[1]
                        hour = int(hour_minute[:2])  # Take first 2 digits as hour
                    else:
                        hour = len(pv_data) % 24
                else:
                    hour = len(pv_data) % 24
                
                # Get global irradiance on tilted surface
                # PVGIS returns G(i) as global irradiance on tilted surface in W/m²
                if 'G(i)' in entry:
                    irradiance = entry['G(i)']  # W/m²
                else:
                    irradiance = 0
                
                # Convert irradiance to PV power
                # 1 kWp system, 15% efficiency, 14% losses
                system_efficiency = 0.15 * (1 - 0.14)  # 12.9% effective efficiency
                pv_power_kw = (irradiance * system_efficiency) / 1000
                
                pv_data.append({
                    'hour': hour,
                    'pv_kw': max(0, pv_power_kw)  # Ensure non-negative
                })
            
            # Create daily average profile (sum all hours across the year, then average)
            daily_profile = {}
            for entry in pv_data:
                hour = entry['hour']
                if hour not in daily_profile:
                    daily_profile[hour] = []
                daily_profile[hour].append(entry['pv_kw'])
            
            print(f"Daily profile stats: {[(h, len(vals)) for h, vals in daily_profile.items()]}")
            
            # Average the values for each hour
            avg_daily_data = []
            for hour in range(24):
                if hour in daily_profile and len(daily_profile[hour]) > 0:
                    avg_power = sum(daily_profile[hour]) / len(daily_profile[hour])
                    print(f"Hour {hour}: {len(daily_profile[hour])} samples, avg power: {avg_power:.3f} kW")
                else:
                    avg_power = 0
                avg_daily_data.append({
                    'hour': hour,
                    'pv_kw': avg_power
                })
            
            df = pd.DataFrame(avg_daily_data)
            df['hour'] = df['hour'].astype(int)
            
            print(f"Successfully fetched PVGIS data: {len(df)} daily profile records")
            print(f"Total daily generation: {df['pv_kw'].sum():.2f} kWh")
            
            return df
            
        except Exception as e:
            print(f"Error fetching PVGIS data: {e}")
            print("Falling back to local CSV data...")
            return self.load_csv_data()
    
    def load_csv_data(self) -> pd.DataFrame:
        """Load PV data from local CSV file as fallback"""
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            self.data = pd.read_csv(self.data_path)
            
            # Validate required columns
            required_columns = ['hour', 'pv_kw']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            print(f"Successfully loaded CSV data: {len(self.data)} records")
            return self.data
            
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            return None
    
    def load_data(self) -> pd.DataFrame:
        """Load PV data - either from PVGIS API or local CSV"""
        if self.use_pvgis:
            self.data = self.fetch_pvgis_data()
        else:
            self.data = self.load_csv_data()
        
        return self.data
    
    def get_daily_profile(self) -> List[Dict]:
        """Get daily PV generation profile"""
        if self.data is None:
            self.load_data()
        
        if self.data is None:
            return []
        
        # Convert to list of dictionaries for API response
        profile = []
        for _, row in self.data.iterrows():
            profile.append({
                "hour": int(row['hour']),
                "pv_kw": float(row['pv_kw']),
                "timestamp": f"{int(row['hour']):02d}:00"
            })
        
        return profile
    
    def get_total_daily_generation(self) -> float:
        """Calculate total daily PV generation in kWh"""
        if self.data is None:
            self.load_data()
        
        if self.data is None:
            return 0.0
        
        # Sum all hourly generation (kW * 1 hour = kWh)
        return float(self.data['pv_kw'].sum())
    
    def get_peak_generation(self) -> Dict:
        """Get peak generation time and value"""
        if self.data is None:
            self.load_data()
        
        if self.data is None:
            return {"hour": 0, "pv_kw": 0}
        
        max_idx = self.data['pv_kw'].idxmax()
        peak_hour = int(self.data.loc[max_idx, 'hour'])
        peak_kw = float(self.data.loc[max_idx, 'pv_kw'])
        
        return {
            "hour": int(peak_hour),
            "pv_kw": float(peak_kw),
            "timestamp": f"{int(peak_hour):02d}:00"
        }
    
    def export_to_json(self, output_path: str = "data/pv_data.json"):
        """Export processed data to JSON format"""
        if self.data is None:
            self.load_data()
        
        if self.data is None:
            return False
        
        try:
            data_dict = {
                "daily_profile": self.get_daily_profile(),
                "total_daily_generation_kwh": self.get_total_daily_generation(),
                "peak_generation": self.get_peak_generation(),
                "metadata": {
                    "total_records": len(self.data),
                    "data_source": self.data_path,
                    "description": "Daily PV generation profile for residential building community"
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(data_dict, f, indent=2)
            
            print(f"Data exported to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False

def main():
    """Main function to demonstrate PV data extraction"""
    extractor = PVDataExtractor(use_pvgis=True)  # Use real PVGIS data
    
    # Load and process data
    data = extractor.load_data()
    if data is not None:
        print("\n=== PV Data Summary (Torino, Italy) ===")
        print(f"Coordinates: {extractor.lat}°N, {extractor.lon}°E")
        print(f"Total daily generation: {extractor.get_total_daily_generation():.2f} kWh")
        
        peak = extractor.get_peak_generation()
        print(f"Peak generation: {peak['pv_kw']:.2f} kW at {peak['timestamp']}")
        
        # Export to JSON
        extractor.export_to_json()
        
        print("\n=== Sample Data ===")
        print(data.head(10))
        
        # Save PVGIS data to CSV for future use
        data.to_csv("data/pvgis_torino_daily.csv", index=False)
        print(f"\nSaved PVGIS data to: data/pvgis_torino_daily.csv")

if __name__ == "__main__":
    main()

