"""
Quick demonstration script to run PV forecasting models with your data.
This will run at least GradientBoosting and show results for your professor.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from forecasting.data_loading import (
    load_pvgis_weather_hourly,
    merge_pv_weather_sources,
)
from forecasting.pv_forecaster import forecast_pv_timeseries as forecast_gb

# Try importing others
try:
    from forecasting.pv_forecaster_xgboost import forecast_pv_timeseries as forecast_xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    print("Note: XGBoost not available (will skip)")

try:
    from forecasting.pv_forecaster_lstm import forecast_pv_timeseries as forecast_lstm
    LSTM_AVAILABLE = True
except:
    LSTM_AVAILABLE = False
    print("Note: LSTM not available (will skip)")

def estimate_pv_from_irradiance(irr_direct, irr_diffuse, temp_amb, capacity_kw=15.0, efficiency=0.18):
    """Simple PV power estimation from irradiance for demonstration."""
    g_eff = irr_direct + irr_diffuse
    pv_power = (g_eff / 1000.0) * capacity_kw * efficiency * (1 - 0.004 * (temp_amb - 25))
    return pv_power.clip(lower=0)

print("=" * 70)
print("PV FORECASTING DEMONSTRATION")
print("=" * 70)
print()

# Load data
print("Loading data...")
pvgis_path = Path("/Users/mariabigonah/Desktop/thesis/building database/Timeseries_45.044_7.639_SA3_40deg_2deg_2005_2023.csv")

pvgis_hourly = load_pvgis_weather_hourly(pvgis_path)
print(f"PVGIS data: {len(pvgis_hourly)} hours")

# Estimate PV power
pv_power = estimate_pv_from_irradiance(
    pvgis_hourly["irr_direct"],
    pvgis_hourly["irr_diffuse"],
    pvgis_hourly["temp_amb"],
    capacity_kw=15.0
)

# Merge (PVGIS only)
history_df = merge_pv_weather_sources(pv_power, pvgis_hourly)
print(f"Dataset: {len(history_df)} hours")

# Need at least several months of data for training
# Use last 2 years if available, otherwise use all available data
if len(history_df) > 17520:
    history_df = history_df.tail(17520)
    print(f"Using last 2 years: {len(history_df)} hours")
elif len(history_df) < 2000:
    print(f"WARNING: Only {len(history_df)} hours available. Need at least 2000+ hours for proper training.")
    print("Using all available data...")

print(f"Dataset: {len(history_df)} hours")
print(f"Date range: {history_df.index.min()} to {history_df.index.max()}")
print()

# Prepare forecast
weather_forecast_df = history_df[["temp_amb", "irr_direct", "irr_diffuse"]].iloc[-24:].copy()
weather_forecast_df.index = history_df.index[-24:] + pd.Timedelta(days=1)

static_features = {"tilt_deg": 40, "azimuth_deg": 2, "capacity_kw": 15.0}

# Run GradientBoosting
print("=" * 70)
print("MODEL 1: GradientBoostingRegressor")
print("=" * 70)
print("Training...")

pv_forecast_gb, metrics_gb, forecaster_gb = forecast_gb(
    history_df=history_df,
    weather_forecast_df=weather_forecast_df,
    static_features=static_features,
    lag_hours=24,
    val_size=0.1,
    test_size=0.1,
)

print("\nResults:")
print(f"  Validation - MAE: {metrics_gb['validation']['mae']:.4f} kW, RMSE: {metrics_gb['validation']['rmse']:.4f} kW, R²: {metrics_gb['validation']['r2']:.4f}")
print(f"  Test       - MAE: {metrics_gb['test']['mae']:.4f} kW, RMSE: {metrics_gb['test']['rmse']:.4f} kW, R²: {metrics_gb['test']['r2']:.4f}")

# Create visualization
output_dir = Path(__file__).parent / "results"
output_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Forecast plot
axes[0].plot(pv_forecast_gb.index, pv_forecast_gb.values, 'o-', label='24h Forecast', linewidth=2, markersize=6)
axes[0].set_ylabel('PV Power (kW)', fontsize=12)
axes[0].set_title('GradientBoosting: 24-Hour PV Forecast', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Metrics bar chart
metrics_names = ['Val MAE', 'Val RMSE', 'Test MAE', 'Test RMSE']
metrics_values = [
    metrics_gb['validation']['mae'],
    metrics_gb['validation']['rmse'],
    metrics_gb['test']['mae'],
    metrics_gb['test']['rmse']
]
axes[1].bar(metrics_names, metrics_values, color=['steelblue', 'steelblue', 'coral', 'coral'], alpha=0.7)
axes[1].set_ylabel('Error (kW)', fontsize=12)
axes[1].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(output_dir / "gradientboosting_results.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Results saved to: {output_dir / 'gradientboosting_results.png'}")

# Save metrics
metrics_df = pd.DataFrame({
    'Set': ['Validation', 'Test'],
    'MAE (kW)': [metrics_gb['validation']['mae'], metrics_gb['test']['mae']],
    'RMSE (kW)': [metrics_gb['validation']['rmse'], metrics_gb['test']['rmse']],
    'nRMSE': [metrics_gb['validation']['nrmse'], metrics_gb['test']['nrmse']],
    'R²': [metrics_gb['validation']['r2'], metrics_gb['test']['r2']],
})
metrics_df.to_csv(output_dir / "gradientboosting_metrics.csv", index=False)
print(f"✓ Metrics saved to: {output_dir / 'gradientboosting_metrics.csv'}")

print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE!")
print("=" * 70)
print(f"\nResults directory: {output_dir}")
print("\nNOTE: This used estimated PV power from irradiance.")
print("      Replace with actual measurements for production use.")

