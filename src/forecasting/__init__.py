"""Forecasting package for MPC-ready energy applications."""

from .pv_forecaster import (
    RecursivePVForecaster,
    ShiftableDevice,
    classify_device_name,
    classify_devices,
    compute_shiftable_load_profiles,
    evaluate_forecast,
    forecast_non_shiftable_load_seasonal_naive,
    forecast_pv_timeseries,
    plot_forecast,
    time_series_train_val_test_split,
)
from .data_loading import (
    load_openweather_hourly,
    load_pvgis_weather_hourly,
    merge_pv_weather_sources,
)

__all__ = [
    "RecursivePVForecaster",
    "classify_device_name",
    "classify_devices",
    "ShiftableDevice",
    "compute_shiftable_load_profiles",
    "load_openweather_hourly",
    "load_pvgis_weather_hourly",
    "merge_pv_weather_sources",
    "evaluate_forecast",
    "forecast_non_shiftable_load_seasonal_naive",
    "forecast_pv_timeseries",
    "plot_forecast",
    "time_series_train_val_test_split",
]

