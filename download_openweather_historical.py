"""
Download historical weather data for Turin from OpenWeather API (2022-2024).

This script efficiently downloads historical weather data while staying within
the 1,000 API call limit by using strategic sampling.

Strategy:
- For 3 years (2022-2024), we have ~1,095 days
- To stay under 1,000 calls, we'll sample every day but skip some strategically
- Alternative: sample every other day (548 calls) or use weekly sampling
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
import sys


# Configuration
TURIN_LAT = 45.0439202
TURIN_LON = 7.639273
API_KEY = "0e17e781b901d9edb9d8d027504aec71"
API_BASE_URL = "https://api.openweathermap.org/data/3.0/onecall/timemachine"

# Date range
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2024, 12, 31)

# Output file
OUTPUT_DIR = Path("/Users/mariabigonah/Desktop/thesis/building database")
OUTPUT_FILE = OUTPUT_DIR / "openweather_historical_turin_2022_2024.csv"
PROGRESS_FILE = OUTPUT_DIR / "openweather_download_progress.json"

# Rate limiting (be conservative to avoid hitting limits)
CALLS_PER_MINUTE = 60  # OpenWeather free tier allows 60 calls/minute
DELAY_BETWEEN_CALLS = 1.1  # seconds (slightly more than 1 second)


def get_unix_timestamp(dt: datetime) -> int:
    """Convert datetime to Unix timestamp."""
    return int(dt.timestamp())


def fetch_weather_data(lat: float, lon: float, dt: datetime, api_key: str) -> Optional[Dict]:
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
        OpenWeather API key
    
    Returns
    -------
    dict or None
        Weather data if successful, None if error
    """
    # Use noon UTC for the requested date to get a representative timestamp
    dt_utc = dt.replace(hour=12, minute=0, second=0, microsecond=0)
    unix_timestamp = get_unix_timestamp(dt_utc)
    
    url = f"{API_BASE_URL}?lat={lat}&lon={lon}&dt={unix_timestamp}&appid={api_key}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching data for {dt.date()}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response status: {e.response.status_code}")
            print(f"  Response text: {e.response.text[:200]}")
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
    
    Strategy: Use systematic sampling to maximize coverage while staying under limit.
    For 3 years (~1,096 days) and 1,000 calls, we'll sample approximately every day
    but skip about 96 days evenly distributed.
    
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
    # Sample approximately every (total_days / max_calls) days
    # This gives us even temporal coverage
    
    # Calculate step size to get close to max_calls samples
    step = total_days / max_calls
    
    sampled_dates = []
    i = 0.0
    
    while i < total_days and len(sampled_dates) < max_calls:
        idx = int(round(i))
        if idx < len(all_dates):
            sampled_dates.append(all_dates[idx])
        i += step
    
    # Always include the last date if not already included
    if all_dates[-1] not in sampled_dates and len(sampled_dates) < max_calls:
        sampled_dates.append(all_dates[-1])
    
    # Remove duplicates and sort
    sampled_dates = sorted(list(set(sampled_dates)))
    
    # Trim to max_calls if needed
    return sampled_dates[:max_calls]


def main():
    """Main function to download historical weather data."""
    print("=" * 80)
    print("OpenWeather Historical Data Downloader for Turin (2022-2024)")
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
        if total_calls >= 1000:
            print(f"\n⚠️  Reached 1,000 API call limit. Stopping.")
            break
        
        print(f"[{i}/{len(dates_to_download)}] Fetching data for {date.date()}...", end=' ')
        
        # Fetch data
        weather_json = fetch_weather_data(TURIN_LAT, TURIN_LON, date, API_KEY)
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

