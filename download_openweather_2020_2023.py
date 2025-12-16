"""
Download historical weather data for Turin from OpenWeather API (2020-2023).

This script efficiently downloads historical weather data while staying within
the 1,000 API call limit by using strategic sampling.

Strategy:
- For 4 years (2020-2023), we have ~1,461 days
- To stay under 1,000 calls, we'll use systematic sampling
- The script saves progress incrementally and can resume if interrupted
"""

import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
from typing import Optional, Dict, List

# Configuration
TURIN_LAT = 45.0439202
TURIN_LON = 7.639273
API_KEY = "0e17e781b901d9edb9d8d027504aec71"  # Remove { } if present
API_BASE_URL = "https://api.openweathermap.org/data/3.0/onecall/timemachine"

# Date range
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2023, 12, 31)

# Output file
OUTPUT_DIR = Path("/Users/mariabigonah/Desktop/thesis/building database")
OUTPUT_FILE = OUTPUT_DIR / "openweather_historical_turin_2020_2023.csv"
PROGRESS_FILE = OUTPUT_DIR / "openweather_download_progress_2020_2023.json"

# Rate limiting
DELAY_BETWEEN_CALLS = 1.1  # seconds


def get_unix_timestamp(dt: datetime) -> int:
    """Convert datetime to Unix timestamp."""
    return int(dt.timestamp())


def fetch_weather_data(lat: float, lon: float, dt: datetime, api_key: str, verbose: bool = False) -> Optional[Dict]:
    """
    Fetch historical weather data for a specific date using OpenWeather One Call API 3.0.
    
    Parameters
    ----------
    lat : float
        Latitude
    lon : float
        Longitude
    dt : datetime
        Date to fetch (will use noon UTC for that date)
    api_key : str
        OpenWeather API key (without curly braces)
    verbose : bool
        If True, print detailed error information
    
    Returns
    -------
    dict or None
        Weather data if successful, None if error
    """
    # Remove any curly braces from API key if accidentally included
    api_key_clean = api_key.strip('{}')
    
    # Use noon UTC for the requested date to get a representative timestamp
    dt_utc = dt.replace(hour=12, minute=0, second=0, microsecond=0)
    unix_timestamp = get_unix_timestamp(dt_utc)
    
    url = f"{API_BASE_URL}?lat={lat}&lon={lon}&dt={unix_timestamp}&appid={api_key_clean}"
    
    try:
        response = requests.get(url, timeout=30)
        data = response.json()
        
        # Check for API errors
        if 'cod' in data and data['cod'] != 200:
            error_msg = data.get('message', 'Unknown error')
            if verbose:
                print(f"  API Error {data.get('cod', response.status_code)}: {error_msg}")
                print(f"  Full response: {data}")
            return None
        
        # Check if response has hourly data
        if 'hourly' not in data:
            if verbose:
                print(f"  Warning: No 'hourly' key in response. Keys: {list(data.keys())}")
                if len(str(data)) < 500:
                    print(f"  Response: {data}")
            return None
        
        return data
        
    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"  Request error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"  Response status: {e.response.status_code}")
                try:
                    print(f"  Response: {e.response.json()}")
                except:
                    print(f"  Response text: {e.response.text[:500]}")
        return None
    except Exception as e:
        if verbose:
            print(f"  Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        return None


def parse_weather_data(weather_json: Dict, target_date: datetime) -> pd.DataFrame:
    """
    Parse OpenWeather API response into a DataFrame with hourly data.
    
    Parameters
    ----------
    weather_json : dict
        JSON response from OpenWeather API
    target_date : datetime
        The date this data corresponds to
    
    Returns
    -------
    pd.DataFrame
        DataFrame with hourly weather data
    """
    if 'hourly' not in weather_json:
        return pd.DataFrame()
    
    records = []
    for hour_data in weather_json['hourly']:
        # Convert Unix timestamp to datetime
        dt = datetime.fromtimestamp(hour_data['dt'])
        
        # Extract relevant weather parameters
        record = {
            'time': dt,
            'temp': hour_data.get('temp', np.nan),
            'pressure': hour_data.get('pressure', np.nan),
            'humidity': hour_data.get('humidity', np.nan),
            'wind_speed': hour_data.get('wind_speed', np.nan),
            'clouds': hour_data.get('clouds', np.nan),
        }
        
        # Add weather condition if available
        if 'weather' in hour_data and len(hour_data['weather']) > 0:
            record['weather_main'] = hour_data['weather'][0].get('main', '')
            record['weather_description'] = hour_data['weather'][0].get('description', '')
        else:
            record['weather_main'] = ''
            record['weather_description'] = ''
        
        records.append(record)
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    df = df.set_index('time').sort_index()
    return df


def load_progress() -> Dict:
    """Load download progress from file."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {'downloaded_dates': [], 'failed_dates': [], 'total_calls': 0}
    return {'downloaded_dates': [], 'failed_dates': [], 'total_calls': 0}


def save_progress(progress: Dict):
    """Save download progress to file."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2, default=str)


def generate_date_sampling_strategy(start: datetime, end: datetime, max_calls: int = 1000) -> List[datetime]:
    """
    Generate a list of dates to sample, staying within the API call limit.
    
    Strategy: Use systematic sampling with even spacing to maximize coverage.
    
    Parameters
    ----------
    start : datetime
        Start date
    end : datetime
        End date
    max_calls : int
        Maximum number of API calls allowed
    
    Returns
    -------
    list of datetime
        Dates to sample
    """
    all_dates = []
    current = start
    while current <= end:
        all_dates.append(current)
        current += timedelta(days=1)
    
    total_days = len(all_dates)
    
    if total_days <= max_calls:
        # We can sample every day
        return all_dates
    
    # We need to skip some days
    # Strategy: Use systematic sampling with even spacing
    # Calculate step size to get close to max_calls samples
    step = total_days / max_calls
    
    sampled_dates = []
    i = 0.0
    
    while i < total_days and len(sampled_dates) < max_calls:
        idx = int(round(i))
        if idx < len(all_dates):
            sampled_dates.append(all_dates[idx])
        i += step
    
    # Always include the first and last dates if not already included
    if all_dates[0] not in sampled_dates and len(sampled_dates) < max_calls:
        sampled_dates.insert(0, all_dates[0])
    if all_dates[-1] not in sampled_dates and len(sampled_dates) < max_calls:
        sampled_dates.append(all_dates[-1])
    
    # Remove duplicates and sort
    sampled_dates = sorted(list(set(sampled_dates)))
    
    # Trim to max_calls if needed
    return sampled_dates[:max_calls]


def main():
    """Main function to download historical weather data."""
    print("=" * 80)
    print("OpenWeather Historical Data Downloader for Turin (2020-2023)")
    print("=" * 80)
    print()
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing progress
    progress = load_progress()
    downloaded_dates = set(progress.get('downloaded_dates', []))
    failed_dates = set(progress.get('failed_dates', []))
    total_calls = progress.get('total_calls', 0)
    
    print(f"Progress loaded:")
    print(f"  - Already downloaded: {len(downloaded_dates)} dates")
    print(f"  - Previously failed: {len(failed_dates)} dates")
    print(f"  - Total API calls made: {total_calls}")
    print()
    
    # Warn if we're at or near the limit
    if total_calls >= 1000:
        print("⚠️  WARNING: You have already used all 1,000 API calls today!")
        print("   The daily limit resets at midnight UTC.")
        print("   Please wait until the limit resets before running again.")
        print()
        return
    elif total_calls >= 900:
        print(f"⚠️  WARNING: You have used {total_calls}/1000 calls today.")
        print("   Only a few calls remaining. Consider waiting for limit reset.")
        print()
    
    # Generate date sampling strategy
    print("Generating date sampling strategy...")
    dates_to_sample = generate_date_sampling_strategy(START_DATE, END_DATE, max_calls=1000)
    print(f"  Total dates to sample: {len(dates_to_sample)}")
    print(f"  Date range: {dates_to_sample[0].date()} to {dates_to_sample[-1].date()}")
    print()
    
    # Filter out already downloaded dates
    dates_to_download = [d for d in dates_to_sample 
                        if d.date() not in downloaded_dates and d.date() not in failed_dates]
    
    remaining_calls = 1000 - total_calls
    if len(dates_to_download) > remaining_calls:
        print(f"Warning: {len(dates_to_download)} dates remaining but only {remaining_calls} calls left.")
        print(f"Will download first {remaining_calls} dates.")
        dates_to_download = dates_to_download[:remaining_calls]
    
    print(f"Dates to download: {len(dates_to_download)}")
    print()
    
    if len(dates_to_download) == 0:
        print("All dates already downloaded!")
        # Still create CSV from existing data
        if OUTPUT_FILE.exists():
            print(f"CSV file already exists: {OUTPUT_FILE}")
            return
    
    # Load existing data if available
    all_data = []
    if OUTPUT_FILE.exists():
        print(f"Loading existing data from {OUTPUT_FILE.name}...")
        try:
            existing_df = pd.read_csv(OUTPUT_FILE, parse_dates=['time'], index_col='time')
            all_data.append(existing_df)
            print(f"  Loaded {len(existing_df)} existing records")
        except Exception as e:
            print(f"  Warning: Could not load existing file: {e}")
            print("  Starting fresh...")
    
    print()
    print("Starting download...")
    print(f"Rate limiting: {DELAY_BETWEEN_CALLS} seconds between calls")
    print()
    
    successful_downloads = 0
    failed_downloads = 0
    
    for i, date in enumerate(dates_to_download, 1):
        # Double-check: Never exceed 1000 calls
        if total_calls >= 1000:
            print(f"\n⚠️  Reached 1,000 API call limit. Stopping.")
            break
        
        # Extra safety: Check remaining calls
        remaining = 1000 - total_calls
        if remaining <= 0:
            print(f"\n⚠️  No API calls remaining. Stopping.")
            break
        
        print(f"[{i}/{len(dates_to_download)}] Fetching data for {date.date()}... (Calls remaining: {remaining})", end=' ')
        
        # Fetch data (verbose on first call to diagnose issues)
        verbose = (i == 1 and total_calls == 0)
        weather_json = fetch_weather_data(TURIN_LAT, TURIN_LON, date, API_KEY, verbose=verbose)
        total_calls += 1
        
        if weather_json is None:
            print("FAILED")
            failed_dates.add(date.date())
            failed_downloads += 1
        else:
            # Parse and store
            df = parse_weather_data(weather_json, date)
            if len(df) > 0:
                all_data.append(df)
                downloaded_dates.add(date.date())
                successful_downloads += 1
                print(f"OK ({len(df)} hours)")
            else:
                print("FAILED (no data)")
                failed_dates.add(date.date())
                failed_downloads += 1
        
        # Save progress periodically (every 10 downloads)
        if i % 10 == 0:
            progress = {
                'downloaded_dates': sorted([str(d) for d in downloaded_dates]),
                'failed_dates': sorted([str(d) for d in failed_dates]),
                'total_calls': total_calls
            }
            save_progress(progress)
            
            # Also save data incrementally
            if all_data:
                combined_df = pd.concat(all_data).sort_index()
                combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                combined_df.to_csv(OUTPUT_FILE)
        
        # Rate limiting
        if i < len(dates_to_download):
            time.sleep(DELAY_BETWEEN_CALLS)
    
    # Final save
    print()
    print("Saving final data...")
    if all_data:
        combined_df = pd.concat(all_data).sort_index()
        # Remove duplicates (keep first occurrence)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        combined_df.to_csv(OUTPUT_FILE)
        print(f"  ✓ Saved {len(combined_df)} records to {OUTPUT_FILE}")
    else:
        print("  ⚠️  No data to save")
    
    # Save final progress
    progress = {
        'downloaded_dates': sorted([str(d) for d in downloaded_dates]),
        'failed_dates': sorted([str(d) for d in failed_dates]),
        'total_calls': total_calls
    }
    save_progress(progress)
    
    print()
    print("=" * 80)
    print("Download Summary")
    print("=" * 80)
    print(f"  Successful downloads: {successful_downloads}")
    print(f"  Failed downloads: {failed_downloads}")
    print(f"  Total API calls made: {total_calls}")
    print(f"  Remaining calls: {1000 - total_calls}")
    print(f"  Output file: {OUTPUT_FILE}")
    print()
    
    if OUTPUT_FILE.exists():
        df = pd.read_csv(OUTPUT_FILE, parse_dates=['time'], index_col='time')
        print(f"Final dataset:")
        print(f"  Total records: {len(df)}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()

