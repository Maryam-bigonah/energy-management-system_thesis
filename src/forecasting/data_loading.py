"""
Utilities to load the user's specific PV and weather datasets and transform
them into the formats expected by the forecasting models.

IMPORTANT:
    * These helpers ONLY read and aggregate the user's data.
    * They do NOT generate any synthetic samples.
    * All resampling is simple aggregation (mean) to 1-hour resolution.

Datasets used (absolute paths as provided by the user):
    - /Users/mariabigonah/Desktop/thesis/building database/Timeseries_45.044_7.639_SA3_40deg_2deg_2005_2023.csv
    - /Users/mariabigonah/Desktop/thesis/building database/openweather_historical.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _read_pvgis_timeseries(filepath: Path) -> pd.DataFrame:
    """
    Read the PVGIS SARAH3 timeseries file provided by the user and return
    the raw 10-minute resolution data as a DataFrame.

    The file has several metadata lines at the top followed by a header line:
        time,Gb(i),Gd(i),Gr(i),H_sun,T2m,WS10m,Int
    and then 10-minute time steps such as: 20050101:0010, ...
    """
    filepath = Path(filepath)

    # Find the header line programmatically (starts with 'time,')
    with filepath.open("r") as f:
        skiprows = 0
        for line in f:
            if line.startswith("time,"):
                break
            skiprows += 1

    df = pd.read_csv(filepath, skiprows=skiprows)
    # Parse the PVGIS time format YYYYMMDD:HHMM into a proper datetime index
    def _parse_pvgis_time(s: str) -> pd.Timestamp:
        date_part, time_part = s.split(":")
        return pd.to_datetime(date_part + time_part, format="%Y%m%d%H%M")

    df["time"] = df["time"].astype(str).map(_parse_pvgis_time)
    df = df.set_index("time").sort_index()
    return df


def load_pvgis_weather_hourly(filepath: Path) -> pd.DataFrame:
    """
    Load the user's PVGIS SARAH3 timeseries file and aggregate it to
    1-hour resolution weather features for PV forecasting.

    Columns in the result:
        - irr_direct  <- Gb(i)  (direct irradiance on tilted plane)
        - irr_diffuse <- Gd(i)  (diffuse irradiance on tilted plane)
        - temp_amb    <- T2m    (ambient temperature)

    Aggregation:
        - 10-minute values are averaged over each hour using resample('1H').mean().
          This does not create synthetic data; it only aggregates the existing
          measurements to the required 1-hour resolution.
    """
    df = _read_pvgis_timeseries(filepath)

    # Rename to internal feature names
    rename_map = {
        "Gb(i)": "irr_direct",
        "Gd(i)": "irr_diffuse",
        "T2m": "temp_amb",
    }
    for src, dst in rename_map.items():
        if src not in df.columns:
            raise ValueError(f"Expected column '{src}' in PVGIS file but did not find it.")
    df = df.rename(columns=rename_map)

    # Keep only the columns we actually use in the forecasting model
    weather_10min = df[["irr_direct", "irr_diffuse", "temp_amb"]]

    # Aggregate to hourly resolution (mean over 10-minute steps within each hour)
    weather_hourly = weather_10min.resample("1H").mean()
    return weather_hourly


def load_openweather_hourly(filepath: Path) -> pd.DataFrame:
    """
    Load the user's OpenWeather historical file and return a 1-hour
    resolution time series.

    Original columns:
        time,pressure,humidity,wind_speed,clouds,temp

    The returned DataFrame keeps these columns and uses 'time' as a
    DatetimeIndex. If the file already has 1-hour resolution (as in the
    user's dataset), no further resampling is applied.
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath, parse_dates=["time"])
    df = df.set_index("time").sort_index()
    return df


def merge_pv_weather_sources(
    pv_power: pd.Series,
    pvgis_hourly: pd.DataFrame,
    openweather_hourly: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merge measured PV power with PVGIS weather (and optionally OpenWeather)
    into a single hourly DataFrame suitable for the PV forecaster.

    Parameters
    ----------
    pv_power : pd.Series
        Historical PV AC power measurements at 1-hour resolution with a
        DatetimeIndex.
    pvgis_hourly : pd.DataFrame
        Output of load_pvgis_weather_hourly. Must have the same hourly index
        (or be alignable by time).
    openweather_hourly : pd.DataFrame, optional
        Output of load_openweather_hourly. If provided, it is joined on the
        same hourly index and all of its columns are added as extra features.
        No synthetic values are generated; only an inner join is performed.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least:
            - 'pv_power'
            - 'temp_amb'
            - 'irr_direct'
            - 'irr_diffuse'
        plus any additional OpenWeather fields if provided.
    """
    if not isinstance(pv_power.index, pd.DatetimeIndex):
        raise ValueError("pv_power must use a DatetimeIndex.")

    pv_df = pv_power.sort_index().to_frame(name="pv_power")

    # Inner join to ensure we only keep timestamps where all data sources exist.
    df = pv_df.join(pvgis_hourly, how="inner")

    if openweather_hourly is not None:
        openweather_hourly = openweather_hourly.sort_index()
        df = df.join(openweather_hourly, how="inner", rsuffix="_ow")

    return df


__all__ = [
    "load_pvgis_weather_hourly",
    "load_openweather_hourly",
    "merge_pv_weather_sources",
]


