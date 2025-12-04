# Short-Term PV Forecasting Module

Implements the MPC-ready, 1-hour-resolution PV forecasting workflow requested by the user. The module is 100% data-driven and only consumes the datasets you provide (historical PV, historical weather, and 24-hour-ahead NWP forecasts). No synthetic data are generated inside the package.

## Structure

- `src/forecasting/pv_forecaster.py`: Core implementation (feature engineering, recursive forecasting, evaluation, optional visualization).
- `requirements.txt`: Python dependencies (`pandas`, `numpy`, `scikit-learn`, `matplotlib`).

## Basic Usage

```python
import pandas as pd
from forecasting import (
    RecursivePVForecaster,
    ShiftableDevice,
    compute_shiftable_load_profiles,
    forecast_non_shiftable_load_seasonal_naive,
    forecast_pv_timeseries,
)

# === 1) PV PRODUCTION FORECAST (DATA-DRIVEN, RECURSIVE) ===

# history_df: hourly historical dataframe with columns
# ["pv_power", "temp_amb", "irr_direct", "irr_diffuse"]
# and a DateTimeIndex.
# weather_fcst_df: 24-row dataframe with the same weather columns and timestamps
# covering the next 24 hours.

static_features = {"tilt_deg": 40, "azimuth_deg": 2, "capacity_kw": 15}

pv_forecast, metrics, forecaster = forecast_pv_timeseries(
    history_df=history_df,
    weather_forecast_df=weather_fcst_df,
    static_features=static_features,
    lag_hours=24,
)

# You can also call evaluate_forecast(y_true, y_pred) and plot_forecast(...)
# once realized PV values for the forecast horizon are available.


# === 2) NON-SHIFTABLE LOAD FORECAST (STRICT WEEKLY SEASONAL NAIVE) ===

# non_shiftable_load_hist: hourly non-shiftable demand (DatetimeIndex, 1-hour freq)
# Assume it already excludes shiftable devices.

non_shiftable_forecast = forecast_non_shiftable_load_seasonal_naive(
    load_history=non_shiftable_load_hist,
    # If omitted, forecast_start defaults to last history timestamp + 1 hour.
    horizon_hours=24,
    season_hours=168,  # fixed weekly period
)


# === 3) SHIFTABLE LOADS (NO FORECASTING, ONLY DETERMINISTIC PROFILES) ===

# Example for a single device (e.g., dishwasher):
# dishwasher_profile_15min: pandas Series with 15-minute resolution for one full cycle
# dishwasher_earliest_start and dishwasher_latest_end: pd.Timestamp

dishwasher = ShiftableDevice(
    name="Dishwasher",
    power_profile_15min_kw=dishwasher_profile_15min,
    earliest_start=dishwasher_earliest_start,
    latest_end=dishwasher_latest_end,
)

devices = {"dishwasher": dishwasher}

# Convert 15-minute cycle profiles to 1-hour resolution (no scheduling/forecasting)
shiftable_hourly_profiles = compute_shiftable_load_profiles(devices, freq="1H")

# `shiftable_hourly_profiles` and the corresponding allowed windows in `devices`
# can then be passed directly to the MPC/optimization layer as deterministic inputs.
```

Use `forecasting.evaluate_forecast` for MAE/RMSE/nRMSE/R² reporting and `forecasting.plot_forecast` for optional visualization once actuals are available.

## Temporal Feature Consistency

**Historical training data**  
The `RecursivePVForecaster._prepare_supervised_frame()` function calls `_add_time_features(history_df)`. This computes the full set of temporal features (hour-of-day as sin/cos, day-of-week, `is_weekend`, month, and season) on the historical PV–weather time series before building the supervised learning frame. These features are therefore available for every training sample.

**24-hour NWP forecast horizon**  
During prediction, `RecursivePVForecaster._build_single_feature_row()` constructs a rolling buffer that includes each future timestamp from `weather_forecast_df` and then calls the same `_add_time_features` helper on this buffer. As a result, the identical temporal features (hour-of-day, day-of-week, `is_weekend`, month, season) are recomputed for every forecast time step using the forecast timestamps.

Using the same `_add_time_features` function in both paths guarantees that the historical training data and the 24-hour NWP forecast share an identical temporal feature space, which is exactly what the specification requires.

