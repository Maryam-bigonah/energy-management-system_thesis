"""
Enhanced Data Loading Utilities for Torino Building
Handles 20 apartments with 4 family archetypes
"""

import pandas as pd
import numpy as np


def load_pvgis_data(filepath_or_url):
    """
    Load PV data from PVGIS output for Torino building
    
    Parameters:
    - filepath_or_url: path to PVGIS CSV file or URL
    
    Returns:
    - DataFrame with 'pv' column and hourly datetime index
    """
    df = pd.read_csv(filepath_or_url)
    
    # Handle different datetime column names
    datetime_cols = ['datetime', 'time', 'timestamp', 'Date', 'Time']
    datetime_col = None
    
    for col in datetime_cols:
        if col in df.columns:
            datetime_col = col
            break
    
    if datetime_col:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df.set_index(datetime_col, inplace=True)
    else:
        # Try first column as datetime
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)
    
    # Ensure hourly frequency
    df = df.resample('H').sum()
    
    # Find PV column (common names from PVGIS)
    pv_cols = [col for col in df.columns if any(
        term in col.lower() for term in ['p', 'pv', 'power', 'generation']
    )]
    
    if pv_cols:
        df['pv'] = df[pv_cols].sum(axis=1)
    else:
        # If no PV column found, use first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df['pv'] = df[numeric_cols[0]]
        else:
            raise ValueError("Could not find PV generation column in PVGIS data")
    
    return df[['pv']]


def load_lpg_data(filepath):
    """
    Load load profile data from Load Profile Generator for 20 apartments
    
    Torino building: 20 apartments, 4 archetypes
    - Archetype 1: Couple working (5 apartments)
    - Archetype 2: Family with one child (5 apartments)
    - Archetype 3: One-working couple (5 apartments)
    - Archetype 4: Retired (5 apartments)
    
    Parameters:
    - filepath: path to Load Profile Generator CSV file
    
    Returns:
    - DataFrame with 'load' column (total building load) and hourly datetime index
    """
    df = pd.read_csv(filepath)
    
    # Handle different datetime column names
    datetime_cols = ['datetime', 'time', 'timestamp', 'Date', 'Time']
    datetime_col = None
    
    for col in datetime_cols:
        if col in df.columns:
            datetime_col = col
            break
    
    if datetime_col:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df.set_index(datetime_col, inplace=True)
    else:
        # Try first column as datetime
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)
    
    # Ensure hourly frequency
    df = df.resample('H').sum()
    
    # Find load/power columns (LPG might have separate columns per apartment)
    load_cols = [col for col in df.columns if any(
        term in col.lower() for term in ['load', 'power', 'demand', 'consumption']
    )]
    
    if load_cols:
        # Sum all apartment loads to get total building load
        df['load'] = df[load_cols].sum(axis=1)
    elif 'Load' in df.columns:
        df.rename(columns={'Load': 'load'}, inplace=True)
    else:
        # If columns are named by apartment number (e.g., "Apartment_1", "Apt_01")
        apt_cols = [col for col in df.columns if any(
            term in col.lower() for term in ['apt', 'apartment', 'unit']
        )]
        if apt_cols:
            df['load'] = df[apt_cols].sum(axis=1)
        else:
            # Use first numeric column as total load
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df['load'] = df[numeric_cols].sum(axis=1)
            else:
                raise ValueError("Could not find load columns in LPG data")
    
    return df[['load']]


def combine_data(pv_df, load_df):
    """
    Combine PV and Load data into single DataFrame
    
    Parameters:
    - pv_df: DataFrame with 'pv' column and datetime index
    - load_df: DataFrame with 'load' column and datetime index
    
    Returns:
    - Combined DataFrame with both 'pv' and 'load' columns, hourly frequency
    """
    # Merge on index (datetime)
    df = pd.merge(pv_df, load_df, left_index=True, right_index=True, how='inner')
    
    # Fill any missing values
    df = df.ffill().bfill()
    
    # Ensure sorted by datetime
    df = df.sort_index()
    
    # Ensure hourly frequency (no duplicates, no gaps)
    df = df.asfreq('H', method='ffill')
    
    return df


def create_archetype_load_profiles(dates, archetype='all'):
    """
    Create load profiles for different family archetypes
    
    Parameters:
    - dates: DatetimeIndex with hourly frequency
    - archetype: 'couple_working', 'family_one_child', 'one_working', 'retired', or 'all'
    
    Returns:
    - Dictionary with load profiles per archetype (or single profile if archetype specified)
    """
    np.random.seed(42)
    n = len(dates)
    
    profiles = {}
    
    # Archetype 1: Couple working (both work, peaks morning/evening)
    load_couple_working = (
        2.0 +  # Base load per apartment
        1.5 * np.sin(2 * np.pi * (dates.hour - 7) / 24) +  # Morning peak (7-9 AM)
        1.5 * np.sin(2 * np.pi * (dates.hour - 18) / 24) +  # Evening peak (6-8 PM)
        0.3 * np.where(dates.dayofweek < 5, 1.0, 0.7) +  # Lower on weekends
        np.random.normal(0, 0.2, n)
    )
    load_couple_working = np.maximum(load_couple_working, 0)
    profiles['couple_working'] = load_couple_working * 5  # 5 apartments
    
    # Archetype 2: Family with one child (higher consumption, earlier peaks)
    load_family_one_child = (
        2.5 +  # Higher base (family)
        1.8 * np.sin(2 * np.pi * (dates.hour - 7) / 24) +  # Morning peak
        2.0 * np.sin(2 * np.pi * (dates.hour - 17) / 24) +  # Earlier evening peak
        0.3 * np.where(dates.dayofweek < 5, 1.0, 0.9) +  # Slightly lower on weekends
        np.random.normal(0, 0.25, n)
    )
    load_family_one_child = np.maximum(load_family_one_child, 0)
    profiles['family_one_child'] = load_family_one_child * 5  # 5 apartments
    
    # Archetype 3: One-working couple (one works, moderate consumption)
    load_one_working = (
        2.0 +
        1.2 * np.sin(2 * np.pi * (dates.hour - 7) / 24) +
        1.3 * np.sin(2 * np.pi * (dates.hour - 18) / 24) +
        0.3 * np.where(dates.dayofweek < 5, 1.0, 0.85) +
        np.random.normal(0, 0.2, n)
    )
    load_one_working = np.maximum(load_one_working, 0)
    profiles['one_working'] = load_one_working * 5  # 5 apartments
    
    # Archetype 4: Retired (lower consumption, more uniform throughout day)
    load_retired = (
        1.8 +
        0.8 * np.sin(2 * np.pi * (dates.hour - 9) / 24) +  # Later morning
        0.8 * np.sin(2 * np.pi * (dates.hour - 19) / 24) +  # Later evening
        0.2 * np.where(dates.dayofweek < 5, 1.0, 0.95) +  # Similar weekend/weekday
        np.random.normal(0, 0.15, n)
    )
    load_retired = np.maximum(load_retired, 0)
    profiles['retired'] = load_retired * 5  # 5 apartments
    
    if archetype == 'all':
        # Sum all archetypes for total building load
        total_load = sum(profiles.values())
        return total_load
    elif archetype in profiles:
        return profiles[archetype]
    else:
        raise ValueError(f"Unknown archetype: {archetype}")


def create_sample_torino_data(start_date='2023-01-01', end_date='2023-12-31'):
    """
    Create sample data for Torino building with 20 apartments, 4 archetypes
    
    Parameters:
    - start_date: start date string (format: 'YYYY-MM-DD')
    - end_date: end date string (format: 'YYYY-MM-DD')
    
    Returns:
    - DataFrame with 'load' and 'pv' columns, hourly datetime index
    """
    dates = pd.date_range(start_date, end_date, freq='h')
    
    # Create load from 4 archetypes (20 apartments total)
    print("Generating load profiles for 4 archetypes (20 apartments)...")
    load = create_archetype_load_profiles(dates, archetype='all')
    
    # Add seasonal variation (higher in winter for heating)
    seasonal_multiplier = 1.0 + 0.25 * np.sin(2 * np.pi * (dates.dayofyear - 81) / 365)
    load = load * seasonal_multiplier
    
    # PV generation for Torino (latitude ~45°N)
    # Seasonal variation (peak in summer, lower in winter)
    pv_seasonal = 30 * np.sin(2 * np.pi * (dates.dayofyear - 81) / 365)
    pv_seasonal = np.maximum(pv_seasonal, 0)
    
    # Daily solar pattern (generation during daylight hours)
    pv_hourly = np.zeros(len(dates))
    day_hours = (dates.hour >= 6) & (dates.hour <= 19)
    pv_hourly[day_hours] = 25 * np.sin(np.pi * (dates.hour[day_hours] - 6) / 13)
    pv_hourly = np.maximum(pv_hourly, 0)
    
    # Add random variation
    pv_random = np.random.normal(0, 2, len(dates))
    pv = pv_seasonal + pv_hourly + pv_random
    pv = np.maximum(pv, 0)
    
    df = pd.DataFrame({
        'load': load,
        'pv': pv
    }, index=dates)
    
    return df

