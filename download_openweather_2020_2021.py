"""
Download historical weather data for Turin from OpenWeather API (2020-2021).
Stays within 1,000 API call limit.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time

# Configuration
LAT = 45.0439202
LON = 7.639273
API_KEY = "0e17e781b901d9edb9d8d027504aec71"

START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2021, 12, 31)
MAX_CALLS = 900  # Safety margin under 1000
SLEEP_SEC = 1.2

OUT_DIR = Path("/Users/mariabigonah/Desktop/thesis/building database")
OUT_CSV = OUT_DIR / "openweather_historical_turin_2020_2021.csv"
BASE_URL = "https://api.openweathermap.org/data/3.0/onecall/timemachine"


def iter_days(start, end):
    """Generate all days in the range."""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def fetch_day(day, verbose=False):
    """Fetch weather data for a specific day."""
    # Use noon UTC for the day
    noon = day.replace(hour=12, minute=0, second=0, microsecond=0)
    ts = int(noon.timestamp())
    url = f"{BASE_URL}?lat={LAT}&lon={LON}&dt={ts}&appid={API_KEY}"
    
    try:
        r = requests.get(url, timeout=30)
        data = r.json()
    except Exception as e:
        if verbose:
            print(f"  Request error: {e}")
        return None

    # Check for API errors
    if isinstance(data, dict) and data.get("cod") not in (None, 200):
        if verbose:
            print(f"  API error: {data}")
        return None

    hourly = data.get("hourly")
    if not hourly:
        if verbose:
            print(f"  No hourly data, keys: {list(data.keys())}")
        return None

    rows = []
    for h in hourly:
        t = datetime.utcfromtimestamp(h["dt"])
        rows.append({
            "time": t,
            "temp": h.get("temp", np.nan),
            "pressure": h.get("pressure", np.nan),
            "humidity": h.get("humidity", np.nan),
            "wind_speed": h.get("wind_speed", np.nan),
            "clouds": h.get("clouds", np.nan),
            "weather_main": (h.get("weather") or [{}])[0].get("main", ""),
            "weather_description": (h.get("weather") or [{}])[0].get("description", ""),
        })
    return rows


def main():
    """Main download function."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    days = list(iter_days(START_DATE, END_DATE))
    total_days = len(days)
    
    print("=" * 80)
    print("OpenWeather Historical Data Downloader for Turin (2020-2021)")
    print("=" * 80)
    print(f"Total days to download: {total_days}")
    print(f"Maximum API calls: {MAX_CALLS}")
    print(f"Output file: {OUT_CSV}")
    print()
    
    if total_days > MAX_CALLS:
        print(f"⚠️  Warning: {total_days} days requested but only {MAX_CALLS} calls available.")
        print(f"   Will download first {MAX_CALLS} days only.")
        days = days[:MAX_CALLS]
    
    all_rows = []
    calls = 0
    
    for i, day in enumerate(days, 1):
        if calls >= MAX_CALLS:
            print(f"\n⚠️  Reached MAX_CALLS={MAX_CALLS}, stopping.")
            break
            
        print(f"[{i}/{len(days)}] {day.date()} ...", end=" ", flush=True)
        
        rows = fetch_day(day, verbose=(i == 1))
        calls += 1
        
        if not rows:
            print("FAILED")
        else:
            all_rows.extend(rows)
            print(f"OK ({len(rows)} hours)")
        
        if i < len(days):
            time.sleep(SLEEP_SEC)
    
    if not all_rows:
        print("\n⚠️  No data collected. Check your API subscription and key.")
        return
    
    # Create DataFrame and save
    df = pd.DataFrame(all_rows)
    df = df.sort_values("time").drop_duplicates(subset=["time"])
    df.to_csv(OUT_CSV, index=False)
    
    print()
    print("=" * 80)
    print("Download Complete!")
    print("=" * 80)
    print(f"Total rows saved: {len(df)}")
    print(f"Time range: {df['time'].min()} → {df['time'].max()}")
    print(f"Output file: {OUT_CSV}")
    print(f"Total API calls made: {calls}")


if __name__ == "__main__":
    main()

