"""
Comprehensive script to run all three PV forecasting models (GradientBoosting, XGBoost, LSTM)
with the user's actual data and generate comparison results.

This script:
1. Loads PVGIS data only
2. Estimates PV power from irradiance (for demonstration - replace with actual measurements)
3. Trains all three models
4. Generates forecasts and evaluation metrics
5. Creates comparison visualizations
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from forecasting.data_loading import (
    load_pvgis_weather_hourly,
    merge_pv_weather_sources,
)
from forecasting.pv_forecaster import forecast_pv_timeseries as forecast_gb

# Try importing XGBoost and LSTM, handle gracefully if not available
try:
    from forecasting.pv_forecaster_xgboost import forecast_pv_timeseries as forecast_xgb
    XGBOOST_AVAILABLE = True
except ImportError as e:
    print(f"Warning: XGBoost not available: {e}")
    XGBOOST_AVAILABLE = False
    forecast_xgb = None

try:
    from forecasting.pv_forecaster_lstm import forecast_pv_timeseries as forecast_lstm
    LSTM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LSTM/TensorFlow not available: {e}")
    LSTM_AVAILABLE = False
    forecast_lstm = None


def estimate_pv_from_irradiance(
    irr_direct: pd.Series,
    irr_diffuse: pd.Series,
    temp_amb: pd.Series,
    capacity_kw: float = 15.0,
    efficiency: float = 0.18,
    temp_coeff: float = -0.004,
    temp_ref: float = 25.0,
) -> pd.Series:
    """
    Estimate PV power from irradiance data for demonstration purposes.
    
    NOTE: This is a simplified model for demonstration. Replace with actual
    PV power measurements when available.
    
    Formula: P = (G_eff / 1000) * capacity * efficiency * (1 + temp_coeff * (T - T_ref))
    where G_eff = G_direct + G_diffuse (in W/m²)
    """
    # Convert irradiance to effective (direct + diffuse)
    g_eff = irr_direct + irr_diffuse
    
    # Simple PV model: power depends on irradiance and temperature
    pv_power = (g_eff / 1000.0) * capacity_kw * efficiency * (
        1 + temp_coeff * (temp_amb - temp_ref)
    )
    
    # Ensure non-negative
    pv_power = pv_power.clip(lower=0)
    
    return pv_power


def main():
    print("=" * 80)
    print("PV Forecasting Model Comparison")
    print("Running GradientBoosting, XGBoost, and LSTM models")
    print("=" * 80)
    print()
    
    # Data paths
    pvgis_path = Path("/Users/mariabigonah/Desktop/thesis/building database/Timeseries_45.044_7.639_SA3_40deg_2deg_2005_2023.csv")
    
    print("Step 1: Loading weather data...")
    print(f"  - PVGIS file: {pvgis_path.name}")
    
    try:
        # Load weather data (PVGIS only)
        pvgis_hourly = load_pvgis_weather_hourly(pvgis_path)
        print(f"  ✓ Loaded PVGIS data: {len(pvgis_hourly)} hours")
        print(f"    Date range: {pvgis_hourly.index.min()} to {pvgis_hourly.index.max()}")
        
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return
    
    print()
    print("Step 2: Estimating PV power from irradiance (for demonstration)...")
    print("  NOTE: Replace this with actual PV power measurements when available!")
    
    # Estimate PV power from irradiance
    pv_power = estimate_pv_from_irradiance(
        irr_direct=pvgis_hourly["irr_direct"],
        irr_diffuse=pvgis_hourly["irr_diffuse"],
        temp_amb=pvgis_hourly["temp_amb"],
        capacity_kw=15.0,  # Assume 15 kW system
    )
    
    print(f"  ✓ Estimated PV power: {len(pv_power)} hours")
    print(f"    Mean: {pv_power.mean():.2f} kW, Max: {pv_power.max():.2f} kW")
    
    print()
    print("Step 3: Merging data sources...")
    
    # Merge data (PVGIS only)
    history_df = merge_pv_weather_sources(
        pv_power=pv_power,
        pvgis_hourly=pvgis_hourly,
    )
    
    print(f"  ✓ Merged dataset: {len(history_df)} hours")
    print(f"    Columns: {list(history_df.columns)}")
    
    # Use a subset for faster training (last 2 years)
    if len(history_df) > 17520:  # More than 2 years
        print()
        print("  Using last 2 years of data for faster training...")
        history_df = history_df.tail(17520)
        print(f"  ✓ Subset: {len(history_df)} hours")
    
    # Prepare forecast weather (use last 24 hours as "forecast")
    weather_forecast_df = history_df[["temp_amb", "irr_direct", "irr_diffuse"]].iloc[-24:].copy()
    weather_forecast_df.index = history_df.index[-24:] + pd.Timedelta(days=1)
    
    # Static features
    static_features = {
        "tilt_deg": 40,
        "azimuth_deg": 2,
        "capacity_kw": 15.0,
    }
    
    print()
    print("=" * 80)
    print("MODEL 1: GradientBoostingRegressor")
    print("=" * 80)
    
    try:
        print("Training GradientBoosting model...")
        pv_forecast_gb, metrics_gb, forecaster_gb = forecast_gb(
            history_df=history_df,
            weather_forecast_df=weather_forecast_df,
            static_features=static_features,
            lag_hours=24,
            val_size=0.1,
            test_size=0.1,
        )
        
        print("  ✓ Training complete!")
        print(f"  Validation metrics:")
        print(f"    MAE:  {metrics_gb['validation']['mae']:.4f} kW")
        print(f"    RMSE: {metrics_gb['validation']['rmse']:.4f} kW")
        print(f"    nRMSE: {metrics_gb['validation']['nrmse']:.4f}")
        print(f"    R²:   {metrics_gb['validation']['r2']:.4f}")
        print(f"  Test metrics:")
        print(f"    MAE:  {metrics_gb['test']['mae']:.4f} kW")
        print(f"    RMSE: {metrics_gb['test']['rmse']:.4f} kW")
        print(f"    nRMSE: {metrics_gb['test']['nrmse']:.4f}")
        print(f"    R²:   {metrics_gb['test']['r2']:.4f}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        metrics_gb = None
        pv_forecast_gb = None
    
    print()
    print("=" * 80)
    print("MODEL 2: XGBoost")
    print("=" * 80)
    
    if not XGBOOST_AVAILABLE:
        print("  ⚠ XGBoost not available (missing dependencies). Skipping...")
        metrics_xgb = None
        pv_forecast_xgb = None
    else:
        try:
            print("Training XGBoost model...")
            pv_forecast_xgb, metrics_xgb, forecaster_xgb = forecast_xgb(
                history_df=history_df,
                weather_forecast_df=weather_forecast_df,
                static_features=static_features,
                lag_hours=24,
                val_size=0.1,
                test_size=0.1,
            )
            
            print("  ✓ Training complete!")
            print(f"  Validation metrics:")
            print(f"    MAE:  {metrics_xgb['validation']['mae']:.4f} kW")
            print(f"    RMSE: {metrics_xgb['validation']['rmse']:.4f} kW")
            print(f"    nRMSE: {metrics_xgb['validation']['nrmse']:.4f}")
            print(f"    R²:   {metrics_xgb['validation']['r2']:.4f}")
            print(f"  Test metrics:")
            print(f"    MAE:  {metrics_xgb['test']['mae']:.4f} kW")
            print(f"    RMSE: {metrics_xgb['test']['rmse']:.4f} kW")
            print(f"    nRMSE: {metrics_xgb['test']['nrmse']:.4f}")
            print(f"    R²:   {metrics_xgb['test']['r2']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            metrics_xgb = None
            pv_forecast_xgb = None
    
    print()
    print("=" * 80)
    print("MODEL 3: LSTM")
    print("=" * 80)
    
    if not LSTM_AVAILABLE:
        print("  ⚠ LSTM/TensorFlow not available (missing dependencies). Skipping...")
        metrics_lstm = None
        pv_forecast_lstm = None
    else:
        try:
            print("Training LSTM model (this may take a few minutes)...")
            pv_forecast_lstm, metrics_lstm, forecaster_lstm = forecast_lstm(
                history_df=history_df,
                weather_forecast_df=weather_forecast_df,
                static_features=static_features,
                lag_hours=24,
                val_size=0.1,
                test_size=0.1,
                epochs=30,  # Reduced for faster demo
                batch_size=32,
                verbose=0,  # Suppress training output
            )
            
            print("  ✓ Training complete!")
            print(f"  Validation metrics:")
            print(f"    MAE:  {metrics_lstm['validation']['mae']:.4f} kW")
            print(f"    RMSE: {metrics_lstm['validation']['rmse']:.4f} kW")
            print(f"    nRMSE: {metrics_lstm['validation']['nrmse']:.4f}")
            print(f"    R²:   {metrics_lstm['validation']['r2']:.4f}")
            print(f"  Test metrics:")
            print(f"    MAE:  {metrics_lstm['test']['mae']:.4f} kW")
            print(f"    RMSE: {metrics_lstm['test']['rmse']:.4f} kW")
            print(f"    nRMSE: {metrics_lstm['test']['nrmse']:.4f}")
            print(f"    R²:   {metrics_lstm['test']['r2']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            metrics_lstm = None
            pv_forecast_lstm = None
    
    print()
    print("=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    # Create comparison table
    available_models = []
    model_names = []
    
    if metrics_gb is not None:
        available_models.append(metrics_gb)
        model_names.append("GradientBoosting")
    if metrics_xgb is not None:
        available_models.append(metrics_xgb)
        model_names.append("XGBoost")
    if metrics_lstm is not None:
        available_models.append(metrics_lstm)
        model_names.append("LSTM")
    
    # Define output directory early so it's always available
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    if len(available_models) > 0:
        comparison_data = {
            "Model": model_names,
            "Val MAE (kW)": [m['validation']['mae'] for m in available_models],
            "Val RMSE (kW)": [m['validation']['rmse'] for m in available_models],
            "Val R²": [m['validation']['r2'] for m in available_models],
            "Test MAE (kW)": [m['test']['mae'] for m in available_models],
            "Test RMSE (kW)": [m['test']['rmse'] for m in available_models],
            "Test R²": [m['test']['r2'] for m in available_models],
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        print()
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
        print(f"\n  ✓ Comparison saved to: {output_dir / 'model_comparison.csv'}")
        
        # Create visualization
        print()
        print("Generating comparison visualization...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Test set metrics comparison
        models = []
        test_mae = []
        test_rmse = []
        test_r2 = []
        forecasts = []
        
        if metrics_gb is not None:
            models.append("GB")
            test_mae.append(metrics_gb['test']['mae'])
            test_rmse.append(metrics_gb['test']['rmse'])
            test_r2.append(metrics_gb['test']['r2'])
            if pv_forecast_gb is not None:
                forecasts.append(("GradientBoosting", pv_forecast_gb))
        
        if metrics_xgb is not None:
            models.append("XGB")
            test_mae.append(metrics_xgb['test']['mae'])
            test_rmse.append(metrics_xgb['test']['rmse'])
            test_r2.append(metrics_xgb['test']['r2'])
            if pv_forecast_xgb is not None:
                forecasts.append(("XGBoost", pv_forecast_xgb))
        
        if metrics_lstm is not None:
            models.append("LSTM")
            test_mae.append(metrics_lstm['test']['mae'])
            test_rmse.append(metrics_lstm['test']['rmse'])
            test_r2.append(metrics_lstm['test']['r2'])
            if pv_forecast_lstm is not None:
                forecasts.append(("LSTM", pv_forecast_lstm))
        
        colors = ['steelblue', 'coral', 'green'][:len(models)]
        
        if len(models) > 0:
            axes[0, 0].bar(models, test_mae, color=colors, alpha=0.7)
            axes[0, 0].set_ylabel('MAE (kW)', fontsize=12)
            axes[0, 0].set_title('Test Set: Mean Absolute Error', fontsize=13, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            axes[0, 1].bar(models, test_rmse, color=colors, alpha=0.7)
            axes[0, 1].set_ylabel('RMSE (kW)', fontsize=12)
            axes[0, 1].set_title('Test Set: Root Mean Squared Error', fontsize=13, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            axes[1, 0].bar(models, test_r2, color=colors, alpha=0.7)
            axes[1, 0].set_ylabel('R²', fontsize=12)
            axes[1, 0].set_title('Test Set: Coefficient of Determination', fontsize=13, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            axes[1, 0].set_ylim([0, 1])
            
            # Forecast comparison
            if len(forecasts) > 0:
                markers = ['o', 's', '^']
                for idx, (name, forecast) in enumerate(forecasts):
                    axes[1, 1].plot(forecast.index, forecast.values, 
                                   marker=markers[idx % len(markers)], linestyle='-',
                                   label=name, linewidth=2, markersize=4)
                axes[1, 1].set_xlabel('Time', fontsize=12)
                axes[1, 1].set_ylabel('PV Power (kW)', fontsize=12)
                axes[1, 1].set_title('24-Hour Forecast Comparison', fontsize=13, fontweight='bold')
                axes[1, 1].legend(fontsize=10)
                axes[1, 1].grid(True, alpha=0.3)
                plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        fig.savefig(output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        print(f"  ✓ Visualization saved to: {output_dir / 'model_comparison.png'}")
        plt.close()
    
    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nNOTE: This demonstration used estimated PV power from irradiance.")
    print("      Replace with actual PV power measurements for production use.")


if __name__ == "__main__":
    main()

