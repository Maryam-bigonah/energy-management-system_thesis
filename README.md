# Short-Term PV Forecasting Module

Implements the MPC-ready, 1-hour-resolution PV forecasting workflow requested by the user. The module is 100% data-driven and only consumes the datasets you provide (historical PV, historical weather, and 24-hour-ahead NWP forecasts). No synthetic data are generated inside the package.

## Structure

- `src/forecasting/pv_forecaster.py`: Core implementation (feature engineering, recursive forecasting, evaluation, optional visualization).
- `requirements.txt`: Python dependencies (`pandas`, `numpy`, `scikit-learn`, `matplotlib`).

## Basic Usage

```python
import pandas as pd
from forecasting import RecursivePVForecaster

# history_df: hourly historical dataframe with columns
# ["pv_power", "temp_amb", "irr_direct", "irr_diffuse"]
# and a DateTimeIndex.
# weather_fcst_df: 24-row dataframe with the same weather columns and timestamps
# covering the next 24 hours.

forecaster = RecursivePVForecaster(
    lag_hours=24,  # adjust if you want a different look-back window
    static_features={"tilt_deg": 25, "azimuth_deg": 180, "capacity_kw": 15},
)

metrics = forecaster.fit(history_df)
pv_forecast = forecaster.forecast(history_df, weather_fcst_df)
```

Use `forecasting.evaluate_forecast` for MAE/RMSE/nRMSE/R² reporting and `forecasting.plot_forecast` for optional visualization once actuals are available.

