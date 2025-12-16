"""
Main forecasting pipeline for load and PV forecasting.

This script:
1. Loads the master dataset
2. Splits into train (2022) and test (2023)
3. Trains baseline models (weekly seasonal naïve for load, persistence for PV)
4. Trains Random Forest models for load and PV
5. Generates 24-hour ahead forecasts on test set
6. Evaluates and saves results
7. Creates visualization plots
"""

import sys
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from forecasting.load_forecaster import (
    weekly_seasonal_naive_forecast,
    prepare_load_features,
    LoadForecaster,
    evaluate_load_forecasts,
)
from forecasting.pv_forecaster_rf import (
    persistence_forecast,
    prepare_pv_features,
    PVForecaster,
    evaluate_pv_forecasts,
)


# Configuration
MASTER_DATASET_PATH = project_root / "outputs" / "MASTER_20_APARTMENTS_2022_2023.csv"
OUTPUT_DIR = project_root / "outputs" / "forecasting_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create subdirectories
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_master_dataset(filepath: Path) -> pd.DataFrame:
    """Load and prepare the master dataset."""
    print(f"Loading master dataset from {filepath}...")
    df = pd.read_csv(filepath, parse_dates=["time"], index_col="time")
    df = df.sort_index()
    print(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train (2022) and test (2023)."""
    train = df[df.index.year == 2022].copy()
    test = df[df.index.year == 2023].copy()
    
    print(f"Train set: {len(train)} rows ({train.index.min()} to {train.index.max()})")
    print(f"Test set: {len(test)} rows ({test.index.min()} to {test.index.max()})")
    
    return train, test


def generate_baseline_forecasts(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate baseline forecasts for load and PV.
    
    Returns
    -------
    load_baseline : pd.DataFrame
        Baseline load forecasts with columns [time, L_true, L_hat_h1, ..., L_hat_h24]
    pv_baseline : pd.DataFrame
        Baseline PV forecasts with columns [time, PV_true, PV_hat_h1, ..., PV_hat_h24]
    """
    print("\n" + "="*60)
    print("Generating baseline forecasts...")
    print("="*60)
    
    # Load baseline
    print("\nGenerating weekly seasonal naïve forecasts for load...")
    load_baseline_rows = []
    
    # Generate forecasts for each hour in test set (using rolling window)
    for t in test.index:
        # Get history up to (but not including) time t
        history_end = t - pd.Timedelta(hours=1)
        history = pd.concat([train, test.loc[:history_end]]) if history_end in test.index else train
        
        if len(history) < 168:  # Need at least 1 week of history
            continue
        
        # Generate 24h forecast
        forecast = weekly_seasonal_naive_forecast(
            history["total_load"],
            forecast_start=t,
            horizon_hours=24,
        )
        
        # Store results
        row = {
            "time": t,
            "L_true": test.loc[t, "total_load"],
        }
        for h in range(1, 25):
            if h <= len(forecast):
                row[f"L_hat_h{h}"] = forecast.iloc[h-1]
            else:
                row[f"L_hat_h{h}"] = np.nan
        
        load_baseline_rows.append(row)
    
    load_baseline = pd.DataFrame(load_baseline_rows)
    load_baseline = load_baseline.set_index("time")
    print(f"Generated {len(load_baseline)} baseline load forecasts")
    
    # PV baseline (persistence)
    print("\nGenerating persistence forecasts for PV...")
    pv_baseline_rows = []
    
    for t in test.index:
        # Get history up to (but not including) time t
        history_end = t - pd.Timedelta(hours=1)
        history = pd.concat([train, test.loc[:history_end]]) if history_end in test.index else train
        
        if len(history) == 0:
            continue
        
        # Generate 24h forecast (persistence: all values = last observed)
        forecast = persistence_forecast(
            history["PV_true"],
            forecast_start=t,
            horizon_hours=24,
        )
        
        # Store results
        row = {
            "time": t,
            "PV_true": test.loc[t, "PV_true"],
        }
        for h in range(1, 25):
            if h <= len(forecast):
                row[f"PV_hat_h{h}"] = forecast.iloc[h-1]
            else:
                row[f"PV_hat_h{h}"] = np.nan
        
        pv_baseline_rows.append(row)
    
    pv_baseline = pd.DataFrame(pv_baseline_rows)
    pv_baseline = pv_baseline.set_index("time")
    print(f"Generated {len(pv_baseline)} baseline PV forecasts")
    
    return load_baseline, pv_baseline


def train_and_forecast_load(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, LoadForecaster]:
    """
    Train Random Forest model for load and generate forecasts.
    
    Returns
    -------
    load_forecasts : pd.DataFrame
        Load forecasts with columns [time, L_true, L_hat_h1, ..., L_hat_h24]
    load_metrics : pd.DataFrame
        Evaluation metrics for each horizon
    model : LoadForecaster
        Trained model
    """
    print("\n" + "="*60)
    print("Training Load Forecasting Model (Random Forest)...")
    print("="*60)
    
    # Prepare features
    print("\nPreparing load features...")
    X_train = prepare_load_features(train)
    y_train = train["total_load"]
    
    # Remove rows with NaN
    valid_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    X_train_clean = X_train[valid_mask]
    y_train_clean = y_train[valid_mask]
    
    print(f"Training samples: {len(X_train_clean)}")
    
    # Train model
    print("\nTraining Random Forest models (24 horizons)...")
    forecaster = LoadForecaster(
        n_estimators=100,
        max_depth=None,
        random_state=42,
    )
    
    train_metrics = forecaster.fit(X_train_clean, y_train_clean)
    print("\nTraining metrics (sample):")
    for h in [1, 6, 12, 24]:
        if h in train_metrics:
            print(f"  Horizon {h:2d}h: MAE={train_metrics[h]['mae']:.3f} kWh/h, "
                  f"RMSE={train_metrics[h]['rmse']:.3f} kWh/h")
    
    # Save model
    model_path = MODELS_DIR / "load_forecaster.pkl"
    forecaster.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Generate forecasts on test set
    print("\nGenerating forecasts on test set...")
    print("Preparing features for entire dataset (this may take a moment)...")
    
    # Prepare features once for entire dataset (more efficient)
    full_data = pd.concat([train, test]).sort_index()
    X_full = prepare_load_features(full_data)
    
    # Filter test set to only hours where we have valid features
    test_with_features = test.loc[test.index.intersection(X_full.index)]
    print(f"Generating forecasts for {len(test_with_features)} test hours...")
    
    load_forecast_rows = []
    for i, t in enumerate(test_with_features.index):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(test_with_features)} hours")
        
        # Get features at time t
        if t not in X_full.index:
            continue
        
        X_t = X_full.loc[[t]]
        
        # Check for NaN
        if X_t.isna().any().any():
            continue
        
        # Generate forecast
        forecast_df = forecaster.predict(X_t, forecast_start=t)
        
        # Store results
        row = {
            "time": t,
            "L_true": test_with_features.loc[t, "total_load"],
        }
        for h in range(1, 25):
            col = f"L_hat_h{h}"
            if col in forecast_df.columns and len(forecast_df) > 0:
                row[col] = forecast_df.iloc[0][col]
            else:
                row[col] = np.nan
        
        load_forecast_rows.append(row)
    
    load_forecasts = pd.DataFrame(load_forecast_rows)
    load_forecasts = load_forecasts.set_index("time")
    print(f"Generated {len(load_forecasts)} load forecasts")
    
    # Evaluate
    print("\nEvaluating forecasts...")
    y_true = test["total_load"]
    load_metrics = evaluate_load_forecasts(y_true, load_forecasts)
    
    print("\nTest metrics (sample):")
    for h in [1, 6, 12, 24]:
        row = load_metrics[load_metrics["horizon"] == h]
        if len(row) > 0:
            print(f"  Horizon {h:2d}h: MAE={row['mae'].values[0]:.3f} kWh/h, "
                  f"RMSE={row['rmse'].values[0]:.3f} kWh/h, "
                  f"nRMSE={row['nrmse'].values[0]:.3f}")
    
    return load_forecasts, load_metrics, forecaster


def train_and_forecast_pv(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, PVForecaster]:
    """
    Train Random Forest model for PV and generate forecasts.
    
    Returns
    -------
    pv_forecasts : pd.DataFrame
        PV forecasts with columns [time, PV_true, PV_hat_h1, ..., PV_hat_h24]
    pv_metrics : pd.DataFrame
        Evaluation metrics for each horizon
    model : PVForecaster
        Trained model
    """
    print("\n" + "="*60)
    print("Training PV Forecasting Model (Random Forest)...")
    print("="*60)
    
    # Prepare features
    print("\nPreparing PV features...")
    X_train = prepare_pv_features(train, include_lag=True)
    y_train = train["PV_true"]
    
    # Remove rows with NaN
    valid_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    X_train_clean = X_train[valid_mask]
    y_train_clean = y_train[valid_mask]
    
    print(f"Training samples: {len(X_train_clean)}")
    print(f"Features: {list(X_train_clean.columns)}")
    
    # Train model
    print("\nTraining Random Forest models (24 horizons)...")
    forecaster = PVForecaster(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        include_lag=True,
    )
    
    train_metrics = forecaster.fit(X_train_clean, y_train_clean)
    print("\nTraining metrics (sample):")
    for h in [1, 6, 12, 24]:
        if h in train_metrics:
            print(f"  Horizon {h:2d}h: MAE={train_metrics[h]['mae']:.3f} kW, "
                  f"RMSE={train_metrics[h]['rmse']:.3f} kW")
    
    # Save model
    model_path = MODELS_DIR / "pv_forecaster.pkl"
    forecaster.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Generate forecasts on test set
    print("\nGenerating forecasts on test set...")
    print("Preparing features for entire dataset (this may take a moment)...")
    
    # Prepare features once for entire dataset (more efficient)
    full_data = pd.concat([train, test]).sort_index()
    X_full = prepare_pv_features(full_data, include_lag=True)
    
    # Filter test set to only hours where we have valid features
    test_with_features = test.loc[test.index.intersection(X_full.index)]
    print(f"Generating forecasts for {len(test_with_features)} test hours...")
    
    pv_forecast_rows = []
    for i, t in enumerate(test_with_features.index):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(test_with_features)} hours")
        
        # Get features at time t
        if t not in X_full.index:
            continue
        
        X_t = X_full.loc[[t]]
        
        # Check for NaN
        if X_t.isna().any().any():
            continue
        
        # Generate forecast
        forecast_df = forecaster.predict(X_t, forecast_start=t)
        
        # Store results
        row = {
            "time": t,
            "PV_true": test_with_features.loc[t, "PV_true"],
        }
        for h in range(1, 25):
            col = f"PV_hat_h{h}"
            if col in forecast_df.columns and len(forecast_df) > 0:
                row[col] = forecast_df.iloc[0][col]
            else:
                row[col] = np.nan
        
        pv_forecast_rows.append(row)
    
    pv_forecasts = pd.DataFrame(pv_forecast_rows)
    pv_forecasts = pv_forecasts.set_index("time")
    print(f"Generated {len(pv_forecasts)} PV forecasts")
    
    # Evaluate
    print("\nEvaluating forecasts...")
    y_true = test["PV_true"]
    pv_metrics = evaluate_pv_forecasts(y_true, pv_forecasts)
    
    print("\nTest metrics (sample):")
    for h in [1, 6, 12, 24]:
        row = pv_metrics[pv_metrics["horizon"] == h]
        if len(row) > 0:
            print(f"  Horizon {h:2d}h: MAE={row['mae'].values[0]:.3f} kW, "
                  f"RMSE={row['rmse'].values[0]:.3f} kW, "
                  f"nRMSE={row['nrmse'].values[0]:.3f}")
    
    return pv_forecasts, pv_metrics, forecaster


def create_forecast_plots(
    load_forecasts: pd.DataFrame,
    pv_forecasts: pd.DataFrame,
    test: pd.DataFrame,
):
    """Create visualization plots for summer and winter weeks."""
    print("\n" + "="*60)
    print("Creating forecast visualization plots...")
    print("="*60)
    
    # Select summer and winter weeks
    summer_start = pd.Timestamp("2023-07-10 00:00:00")
    winter_start = pd.Timestamp("2023-01-10 00:00:00")
    
    week_days = 7
    week_hours = week_days * 24
    
    # Summer week
    if summer_start in test.index:
        summer_end = summer_start + pd.Timedelta(hours=week_hours-1)
        summer_mask = (test.index >= summer_start) & (test.index <= summer_end)
        summer_data = test[summer_mask]
        summer_load = load_forecasts.loc[summer_data.index, ["L_true", "L_hat_h1", "L_hat_h6", "L_hat_h12", "L_hat_h24"]]
        summer_pv = pv_forecasts.loc[summer_data.index, ["PV_true", "PV_hat_h1", "PV_hat_h6", "PV_hat_h12", "PV_hat_h24"]]
        
        # Load plot
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(summer_load.index, summer_load["L_true"], label="True", linewidth=2, color="black")
        ax.plot(summer_load.index, summer_load["L_hat_h1"], label="h=1h", alpha=0.7, linestyle="--")
        ax.plot(summer_load.index, summer_load["L_hat_h6"], label="h=6h", alpha=0.7, linestyle="--")
        ax.plot(summer_load.index, summer_load["L_hat_h12"], label="h=12h", alpha=0.7, linestyle="--")
        ax.plot(summer_load.index, summer_load["L_hat_h24"], label="h=24h", alpha=0.7, linestyle="--")
        ax.set_xlabel("Time")
        ax.set_ylabel("Load (kWh/h)")
        ax.set_title("Load Forecasts - Summer Week (July 2023)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "load_forecast_summer_week.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # PV plot
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(summer_pv.index, summer_pv["PV_true"], label="True", linewidth=2, color="black")
        ax.plot(summer_pv.index, summer_pv["PV_hat_h1"], label="h=1h", alpha=0.7, linestyle="--")
        ax.plot(summer_pv.index, summer_pv["PV_hat_h6"], label="h=6h", alpha=0.7, linestyle="--")
        ax.plot(summer_pv.index, summer_pv["PV_hat_h12"], label="h=12h", alpha=0.7, linestyle="--")
        ax.plot(summer_pv.index, summer_pv["PV_hat_h24"], label="h=24h", alpha=0.7, linestyle="--")
        ax.set_xlabel("Time")
        ax.set_ylabel("PV Power (kW)")
        ax.set_title("PV Forecasts - Summer Week (July 2023)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "pv_forecast_summer_week.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print("Created summer week plots")
    
    # Winter week
    if winter_start in test.index:
        winter_end = winter_start + pd.Timedelta(hours=week_hours-1)
        winter_mask = (test.index >= winter_start) & (test.index <= winter_end)
        winter_data = test[winter_mask]
        winter_load = load_forecasts.loc[winter_data.index, ["L_true", "L_hat_h1", "L_hat_h6", "L_hat_h12", "L_hat_h24"]]
        winter_pv = pv_forecasts.loc[winter_data.index, ["PV_true", "PV_hat_h1", "PV_hat_h6", "PV_hat_h12", "PV_hat_h24"]]
        
        # Load plot
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(winter_load.index, winter_load["L_true"], label="True", linewidth=2, color="black")
        ax.plot(winter_load.index, winter_load["L_hat_h1"], label="h=1h", alpha=0.7, linestyle="--")
        ax.plot(winter_load.index, winter_load["L_hat_h6"], label="h=6h", alpha=0.7, linestyle="--")
        ax.plot(winter_load.index, winter_load["L_hat_h12"], label="h=12h", alpha=0.7, linestyle="--")
        ax.plot(winter_load.index, winter_load["L_hat_h24"], label="h=24h", alpha=0.7, linestyle="--")
        ax.set_xlabel("Time")
        ax.set_ylabel("Load (kWh/h)")
        ax.set_title("Load Forecasts - Winter Week (January 2023)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "load_forecast_winter_week.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # PV plot
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(winter_pv.index, winter_pv["PV_true"], label="True", linewidth=2, color="black")
        ax.plot(winter_pv.index, winter_pv["PV_hat_h1"], label="h=1h", alpha=0.7, linestyle="--")
        ax.plot(winter_pv.index, winter_pv["PV_hat_h6"], label="h=6h", alpha=0.7, linestyle="--")
        ax.plot(winter_pv.index, winter_pv["PV_hat_h12"], label="h=12h", alpha=0.7, linestyle="--")
        ax.plot(winter_pv.index, winter_pv["PV_hat_h24"], label="h=24h", alpha=0.7, linestyle="--")
        ax.set_xlabel("Time")
        ax.set_ylabel("PV Power (kW)")
        ax.set_title("PV Forecasts - Winter Week (January 2023)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "pv_forecast_winter_week.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print("Created winter week plots")


def main():
    """Main pipeline execution."""
    print("="*60)
    print("FORECASTING PIPELINE")
    print("="*60)
    print(f"Start time: {datetime.now()}")
    
    # Load data
    df = load_master_dataset(MASTER_DATASET_PATH)
    
    # Split train/test
    train, test = split_train_test(df)
    
    # Generate baseline forecasts
    load_baseline, pv_baseline = generate_baseline_forecasts(train, test)
    
    # Save baseline forecasts
    load_baseline.to_csv(OUTPUT_DIR / "load_forecasts_baseline.csv")
    pv_baseline.to_csv(OUTPUT_DIR / "pv_forecasts_baseline.csv")
    print(f"\nBaseline forecasts saved to {OUTPUT_DIR}")
    
    # Train and forecast load
    load_forecasts, load_metrics, load_model = train_and_forecast_load(train, test)
    
    # Train and forecast PV
    pv_forecasts, pv_metrics, pv_model = train_and_forecast_pv(train, test)
    
    # Save results
    load_forecasts.to_csv(OUTPUT_DIR / "load_forecasts_rf.csv")
    pv_forecasts.to_csv(OUTPUT_DIR / "pv_forecasts_rf.csv")
    load_metrics.to_csv(OUTPUT_DIR / "load_metrics.csv", index=False)
    pv_metrics.to_csv(OUTPUT_DIR / "pv_metrics.csv", index=False)
    print(f"\nForecasts and metrics saved to {OUTPUT_DIR}")
    
    # Create plots
    create_forecast_plots(load_forecasts, pv_forecasts, test)
    print(f"\nPlots saved to {FIGURES_DIR}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"End time: {datetime.now()}")


if __name__ == "__main__":
    main()
