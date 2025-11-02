"""
Data Loading Utilities
Helper functions to load data from PVGIS and Load Profile Generator
"""

import pandas as pd
import numpy as np


def load_pvgis_data(filepath_or_url):
    """
    Load PV data from PVGIS output
    
    Expected format: CSV with datetime and PV generation columns
    Adjust column names and parsing based on your PVGIS export format
    """
    # Example: adjust based on your PVGIS output format
    df = pd.read_csv(filepath_or_url)
    
    # If datetime column exists, set as index
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    
    # Ensure hourly frequency
    df = df.resample('H').sum()  # or .mean() depending on your data
    
    # Rename PV column if needed (adjust based on your column name)
    if 'P' in df.columns:
        df.rename(columns={'P': 'pv'}, inplace=True)
    elif 'PV' in df.columns:
        df.rename(columns={'PV': 'pv'}, inplace=True)
    
    return df[['pv']]


def load_lpg_data(filepath):
    """
    Load load profile data from Load Profile Generator
    
    Expected format: CSV with datetime and load columns
    Adjust column names and parsing based on your LPG export format
    """
    df = pd.read_csv(filepath)
    
    # If datetime column exists, set as index
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    
    # Ensure hourly frequency
    df = df.resample('H').sum()
    
    # Rename load column if needed (adjust based on your column name)
    # For 20 apartments, sum all apartment loads if they're separate columns
    load_cols = [col for col in df.columns if 'load' in col.lower() or 'power' in col.lower()]
    if load_cols:
        df['load'] = df[load_cols].sum(axis=1)
    elif 'Load' in df.columns:
        df.rename(columns={'Load': 'load'}, inplace=True)
    
    return df[['load']]


def combine_data(pv_df, load_df):
    """
    Combine PV and Load data into single DataFrame
    
    Parameters:
    - pv_df: DataFrame with 'pv' column and datetime index
    - load_df: DataFrame with 'load' column and datetime index
    
    Returns:
    - Combined DataFrame with both 'pv' and 'load' columns
    """
    # Merge on index (datetime)
    df = pd.merge(pv_df, load_df, left_index=True, right_index=True, how='inner')
    
    # Fill any missing values (forward fill then backward fill)
    df = df.ffill().bfill()
    
    # Ensure sorted by datetime
    df = df.sort_index()
    
    return df


def create_sample_torino_data(start_date='2023-01-01', end_date='2023-12-31'):
    """
    Create sample data for Torino building with 20 apartments
    This is a placeholder - replace with actual data loading
    
    Parameters:
    - start_date: start date for data generation
    - end_date: end date for data generation
    
    Returns:
    - DataFrame with 'load' and 'pv' columns, hourly frequency
    """
    dates = pd.date_range(start_date, end_date, freq='H')
    n = len(dates)
    
    # Load profile: 20 apartments, 4 archetypes
    # Archetype 1: Couple working (5 apartments)
    # Archetype 2: Family with one child (5 apartments)
    # Archetype 3: One-working couple (5 apartments)
    # Archetype 4: Retired (5 apartments)
    
    np.random.seed(42)
    
    # Base load pattern (Torino climate)
    base_load = 40  # kW base load
    
    # Hourly pattern (higher during morning/evening)
    hourly_pattern = (
        15 * np.sin(2 * np.pi * (dates.hour - 6) / 24) +  # Morning peak
        10 * np.sin(2 * np.pi * (dates.hour - 18) / 24)   # Evening peak
    )
    hourly_pattern[hourly_pattern < 0] = 0
    
    # Weekly pattern (lower on weekends)
    weekly_pattern = np.where(dates.dayofweek >= 5, 0.8, 1.0)
    
    # Seasonal pattern (higher in winter, lower in summer)
    seasonal_pattern = 1.0 + 0.3 * np.sin(2 * np.pi * (dates.dayofyear - 81) / 365)
    
    # Random variation
    random_variation = np.random.normal(0, 5, n)
    
    # Total load
    load = (base_load + hourly_pattern) * weekly_pattern * seasonal_pattern + random_variation
    load[load < 0] = 0
    
    # PV generation (Torino latitude ~45°)
    # Seasonal variation (peak in summer)
    pv_seasonal = 25 * np.sin(2 * np.pi * (dates.dayofyear - 81) / 365)
    pv_seasonal[pv_seasonal < 0] = 0
    
    # Hourly pattern (solar generation during day, 0 at night)
    pv_hourly = np.zeros(n)
    day_hours = (dates.hour >= 6) & (dates.hour <= 18)
    pv_hourly[day_hours] = 20 * np.sin(np.pi * (dates.hour[day_hours] - 6) / 12)
    
    # Random variation
    pv_random = np.random.normal(0, 2, n)
    
    # Total PV
    pv = pv_seasonal + pv_hourly + pv_random
    pv[pv < 0] = 0
    
    df = pd.DataFrame({
        'load': load,
        'pv': pv
    }, index=dates)
    
    return df

