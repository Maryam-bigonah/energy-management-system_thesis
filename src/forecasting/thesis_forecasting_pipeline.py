"""
Thesis Forecasting Pipeline - 24-hour ahead predictions for Load and PV

Implements:
1. Baseline models (seasonal naïve for load, persistence for PV)
2. Random Forest models (load: lags + calendar, PV: PVGIS + weather + calendar)
3. Direct multi-step forecasting (24 separate models per target)
4. Train on 2022, test on 2023
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load master dataset and prepare for forecasting.
    
    Preprocessing:
    - Parse time index
    - Ensure proper data types
    - Sort by time
    """
    print("=" * 70)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("=" * 70)
    
    df = pd.read_csv(csv_path, parse_dates=['time'])
    df = df.set_index('time').sort_index()
    
    # Ensure numeric columns are numeric
    numeric_cols = ['total_load', 'PV_true', 'temp', 'humidity', 'wind_speed', 'clouds',
                    'Gb', 'Gd', 'Gr', 'H_sun', 'T2m', 'WS10m', 'hour', 'dow', 
                    'is_weekend', 'month', 'season']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove any rows with missing critical values
    initial_len = len(df)
    df = df.dropna(subset=['total_load', 'PV_true'])
    removed = initial_len - len(df)
    if removed > 0:
        print(f"  Removed {removed} rows with missing load or PV values")
    
    print(f"  Loaded {len(df)} rows")
    print(f"  Time range: {df.index.min()} to {df.index.max()}")
    print(f"  Columns: {len(df.columns)}")
    
    return df


def create_lag_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Create lag features for load forecasting:
    - L(t-1): previous hour
    - L(t-24): same hour yesterday
    - L(t-168): same hour last week
    """
    df = df.copy()
    
    # Lag 1 hour
    df[f'{target_col}_lag1'] = df[target_col].shift(1)
    
    # Lag 24 hours (same hour yesterday)
    df[f'{target_col}_lag24'] = df[target_col].shift(24)
    
    # Lag 168 hours (same hour last week)
    df[f'{target_col}_lag168'] = df[target_col].shift(168)
    
    return df


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train (2022) and test (2023).
    """
    print("\n" + "=" * 70)
    print("STEP 2: TRAIN/TEST SPLIT")
    print("=" * 70)
    
    train = df[df.index.year == 2022].copy()
    test = df[df.index.year == 2023].copy()
    
    print(f"  Training set: {len(train)} rows ({train.index.min()} to {train.index.max()})")
    print(f"  Test set: {len(test)} rows ({test.index.min()} to {test.index.max()})")
    
    return train, test


def baseline_load_seasonal_naive(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """
    Baseline: Weekly seasonal naïve for load forecasting.
    L_hat(t+h) = L(t+h-168)
    """
    result = df.copy()
    
    for h in range(1, horizon + 1):
        result[f'load_hat_h{h}'] = result['total_load'].shift(168 - h)
    
    return result


def baseline_pv_persistence(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """
    Baseline: Persistence for PV forecasting.
    PV_hat(t+h) = PV(t)
    """
    result = df.copy()
    
    for h in range(1, horizon + 1):
        result[f'pv_hat_h{h}'] = result['PV_true']
    
    return result


def train_load_forecaster(train: pd.DataFrame) -> Dict[int, RandomForestRegressor]:
    """
    Train 24 separate Random Forest models for load forecasting (one per horizon).
    
    Features:
    - Lags: L(t-1), L(t-24), L(t-168)
    - Calendar: hour, dow, is_weekend, month, season
    """
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING LOAD FORECASTING MODELS")
    print("=" * 70)
    
    # Create lag features
    train_with_lags = create_lag_features(train, 'total_load')
    
    # Feature set for load forecasting
    load_features = [
        'total_load_lag1', 'total_load_lag24', 'total_load_lag168',
        'hour', 'dow', 'is_weekend', 'month', 'season'
    ]
    
    # Remove rows with NaN from lags (first 168 hours)
    train_clean = train_with_lags.dropna(subset=load_features + ['total_load'])
    
    X_train = train_clean[load_features]
    models = {}
    
    print(f"  Training 24 models (one per horizon)...")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {load_features}")
    
    for h in range(1, 25):
        # Create target: load at t+h
        y_train = train_clean['total_load'].shift(-h).dropna()
        X_train_h = X_train.loc[y_train.index]
        
        if len(X_train_h) == 0:
            continue
        
        # Train Random Forest
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_h, y_train)
        models[h] = model
        
        if h % 6 == 0:
            print(f"    Trained model for horizon h={h}")
    
    print(f"  ✓ Trained {len(models)} load forecasting models")
    return models


def train_pv_forecaster(train: pd.DataFrame) -> Dict[int, RandomForestRegressor]:
    """
    Train 24 separate Random Forest models for PV forecasting (one per horizon).
    
    Features:
    - PVGIS: Gb, Gd, Gr, H_sun, T2m, WS10m
    - Weather: clouds, temp, humidity, wind_speed
    - Calendar: hour, month, season
    - Optional lag: PV(t-1)
    """
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING PV FORECASTING MODELS")
    print("=" * 70)
    
    # Create lag feature
    train_with_lag = train.copy()
    train_with_lag['PV_true_lag1'] = train_with_lag['PV_true'].shift(1)
    
    # Feature set for PV forecasting
    pv_features = [
        'Gb', 'Gd', 'Gr', 'H_sun', 'T2m', 'WS10m',  # PVGIS
        'clouds', 'temp', 'humidity', 'wind_speed',  # OpenWeather
        'hour', 'month', 'season',  # Calendar
        'PV_true_lag1'  # Optional lag
    ]
    
    # Remove rows with NaN
    train_clean = train_with_lag.dropna(subset=pv_features + ['PV_true'])
    
    X_train = train_clean[pv_features]
    models = {}
    
    print(f"  Training 24 models (one per horizon)...")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {len(pv_features)} features from PVGIS + OpenWeather + calendar")
    
    for h in range(1, 25):
        # Create target: PV at t+h
        y_train = train_clean['PV_true'].shift(-h).dropna()
        X_train_h = X_train.loc[y_train.index]
        
        if len(X_train_h) == 0:
            continue
        
        # Train Random Forest
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_h, y_train)
        models[h] = model
        
        if h % 6 == 0:
            print(f"    Trained model for horizon h={h}")
    
    print(f"  ✓ Trained {len(models)} PV forecasting models")
    return models


def generate_forecasts(
    test: pd.DataFrame,
    load_models: Dict[int, RandomForestRegressor],
    pv_models: Dict[int, RandomForestRegressor],
    load_baseline: pd.DataFrame,
    pv_baseline: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate 24-hour ahead forecasts for both load and PV on test set.
    """
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING FORECASTS")
    print("=" * 70)
    
    # Prepare test data with lags
    test_load = create_lag_features(test, 'total_load')
    test_pv = test.copy()
    test_pv['PV_true_lag1'] = test_pv['PV_true'].shift(1)
    
    # Load features
    load_features = [
        'total_load_lag1', 'total_load_lag24', 'total_load_lag168',
        'hour', 'dow', 'is_weekend', 'month', 'season'
    ]
    
    # PV features
    pv_features = [
        'Gb', 'Gd', 'Gr', 'H_sun', 'T2m', 'WS10m',
        'clouds', 'temp', 'humidity', 'wind_speed',
        'hour', 'month', 'season', 'PV_true_lag1'
    ]
    
    # Initialize forecast DataFrames
    load_forecasts = test[['total_load']].copy()
    pv_forecasts = test[['PV_true']].copy()
    
    # Generate forecasts for each horizon
    print("  Generating load forecasts...")
    for h in range(1, 25):
        if h not in load_models:
            continue
        
        # Get features for this horizon
        test_clean = test_load.dropna(subset=load_features)
        if len(test_clean) == 0:
            continue
        
        X_test = test_clean[load_features]
        predictions = load_models[h].predict(X_test)
        
        # Align predictions with test index
        load_forecasts.loc[X_test.index, f'load_hat_h{h}'] = predictions
        
        if h % 6 == 0:
            print(f"    Generated forecasts for horizon h={h}")
    
    print("  Generating PV forecasts...")
    for h in range(1, 25):
        if h not in pv_models:
            continue
        
        # Get features for this horizon
        test_clean = test_pv.dropna(subset=pv_features)
        if len(test_clean) == 0:
            continue
        
        X_test = test_clean[pv_features]
        predictions = pv_models[h].predict(X_test)
        
        # Align predictions with test index
        pv_forecasts.loc[X_test.index, f'pv_hat_h{h}'] = predictions
        
        if h % 6 == 0:
            print(f"    Generated forecasts for horizon h={h}")
    
    # Add baseline forecasts
    print("  Adding baseline forecasts...")
    load_baseline_test = baseline_load_seasonal_naive(test)
    pv_baseline_test = baseline_pv_persistence(test)
    
    for h in range(1, 25):
        if f'load_hat_h{h}' in load_baseline_test.columns:
            load_forecasts[f'load_baseline_h{h}'] = load_baseline_test[f'load_hat_h{h}']
        if f'pv_hat_h{h}' in pv_baseline_test.columns:
            pv_forecasts[f'pv_baseline_h{h}'] = pv_baseline_test[f'pv_hat_h{h}']
    
    print(f"  ✓ Generated forecasts for {len(load_forecasts)} test samples")
    
    return load_forecasts, pv_forecasts


def evaluate_forecasts(
    load_forecasts: pd.DataFrame,
    pv_forecasts: pd.DataFrame
) -> Dict:
    """
    Evaluate forecasts using MAE, RMSE, and nRMSE.
    """
    print("\n" + "=" * 70)
    print("STEP 6: EVALUATION")
    print("=" * 70)
    
    metrics = {}
    
    # Evaluate load forecasts
    print("\n  Load Forecasting Metrics:")
    print("  " + "-" * 60)
    print(f"  {'Horizon':<8} {'Model MAE':<12} {'Model RMSE':<12} {'Baseline MAE':<14} {'Baseline RMSE':<14}")
    print("  " + "-" * 60)
    
    load_metrics = {}
    for h in range(1, 25):
        if f'load_hat_h{h}' not in load_forecasts.columns:
            continue
        
        # Model metrics
        y_true = load_forecasts['total_load'].dropna()
        y_pred = load_forecasts[f'load_hat_h{h}'].dropna()
        common_idx = y_true.index.intersection(y_pred.index)
        
        if len(common_idx) == 0:
            continue
        
        y_true_h = y_true.loc[common_idx]
        y_pred_h = y_pred.loc[common_idx]
        
        mae = mean_absolute_error(y_true_h, y_pred_h)
        rmse = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
        nrmse = rmse / (y_true_h.max() - y_true_h.min() + 1e-10)
        
        # Baseline metrics
        if f'load_baseline_h{h}' in load_forecasts.columns:
            y_baseline = load_forecasts[f'load_baseline_h{h}'].loc[common_idx]
            baseline_mae = mean_absolute_error(y_true_h, y_baseline)
            baseline_rmse = np.sqrt(mean_squared_error(y_true_h, y_baseline))
        else:
            baseline_mae = np.nan
            baseline_rmse = np.nan
        
        load_metrics[h] = {
            'mae': mae,
            'rmse': rmse,
            'nrmse': nrmse,
            'baseline_mae': baseline_mae,
            'baseline_rmse': baseline_rmse
        }
        
        if h % 6 == 0 or h <= 3:
            print(f"  h={h:<6} {mae:>10.4f}   {rmse:>10.4f}   {baseline_mae:>12.4f}     {baseline_rmse:>12.4f}")
    
    metrics['load'] = load_metrics
    
    # Evaluate PV forecasts
    print("\n  PV Forecasting Metrics:")
    print("  " + "-" * 60)
    print(f"  {'Horizon':<8} {'Model MAE':<12} {'Model RMSE':<12} {'Baseline MAE':<14} {'Baseline RMSE':<14}")
    print("  " + "-" * 60)
    
    pv_metrics = {}
    for h in range(1, 25):
        if f'pv_hat_h{h}' not in pv_forecasts.columns:
            continue
        
        # Model metrics
        y_true = pv_forecasts['PV_true'].dropna()
        y_pred = pv_forecasts[f'pv_hat_h{h}'].dropna()
        common_idx = y_true.index.intersection(y_pred.index)
        
        if len(common_idx) == 0:
            continue
        
        y_true_h = y_true.loc[common_idx]
        y_pred_h = y_pred.loc[common_idx]
        
        mae = mean_absolute_error(y_true_h, y_pred_h)
        rmse = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
        nrmse = rmse / (y_true_h.max() - y_true_h.min() + 1e-10) if (y_true_h.max() - y_true_h.min()) > 0 else np.nan
        
        # Baseline metrics
        if f'pv_baseline_h{h}' in pv_forecasts.columns:
            y_baseline = pv_forecasts[f'pv_baseline_h{h}'].loc[common_idx]
            baseline_mae = mean_absolute_error(y_true_h, y_baseline)
            baseline_rmse = np.sqrt(mean_squared_error(y_true_h, y_baseline))
        else:
            baseline_mae = np.nan
            baseline_rmse = np.nan
        
        pv_metrics[h] = {
            'mae': mae,
            'rmse': rmse,
            'nrmse': nrmse,
            'baseline_mae': baseline_mae,
            'baseline_rmse': baseline_rmse
        }
        
        if h % 6 == 0 or h <= 3:
            print(f"  h={h:<6} {mae:>10.4f}   {rmse:>10.4f}   {baseline_mae:>12.4f}     {baseline_rmse:>12.4f}")
    
    metrics['pv'] = pv_metrics
    
    return metrics


def save_results(
    load_forecasts: pd.DataFrame,
    pv_forecasts: pd.DataFrame,
    metrics: Dict,
    output_dir: str
):
    """
    Save forecast files and metrics.
    """
    print("\n" + "=" * 70)
    print("STEP 7: SAVING RESULTS")
    print("=" * 70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save load forecasts
    load_file = output_path / 'load_forecasts_2023.csv'
    load_forecasts.to_csv(load_file)
    print(f"  ✓ Load forecasts saved: {load_file}")
    
    # Save PV forecasts
    pv_file = output_path / 'pv_forecasts_2023.csv'
    pv_forecasts.to_csv(pv_file)
    print(f"  ✓ PV forecasts saved: {pv_file}")
    
    # Save metrics summary
    metrics_summary = []
    for target in ['load', 'pv']:
        for h, m in metrics[target].items():
            metrics_summary.append({
                'target': target,
                'horizon': h,
                'mae': m['mae'],
                'rmse': m['rmse'],
                'nrmse': m['nrmse'],
                'baseline_mae': m['baseline_mae'],
                'baseline_rmse': m['baseline_rmse']
            })
    
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_file = output_path / 'forecast_metrics_summary.csv'
    metrics_df.to_csv(metrics_file, index=False)
    print(f"  ✓ Metrics summary saved: {metrics_file}")
    
    print(f"\n  All results saved to: {output_path}")


def run_forecasting_pipeline(
    csv_path: str,
    output_dir: str = '/Users/mariabigonah/Desktop/thesis/code/outputs/forecasts'
):
    """
    Run the complete forecasting pipeline.
    """
    print("\n" + "=" * 70)
    print("THESIS FORECASTING PIPELINE")
    print("24-Hour Ahead Load and PV Forecasting")
    print("=" * 70)
    
    # Step 1: Load and preprocess
    df = load_and_prepare_data(csv_path)
    
    # Step 2: Split train/test
    train, test = split_train_test(df)
    
    # Step 3: Train load forecaster
    load_models = train_load_forecaster(train)
    
    # Step 4: Train PV forecaster
    pv_models = train_pv_forecaster(train)
    
    # Step 5: Generate forecasts
    load_forecasts, pv_forecasts = generate_forecasts(
        test, load_models, pv_models,
        baseline_load_seasonal_naive(test),
        baseline_pv_persistence(test)
    )
    
    # Step 6: Evaluate
    metrics = evaluate_forecasts(load_forecasts, pv_forecasts)
    
    # Step 7: Save results
    save_results(load_forecasts, pv_forecasts, metrics, output_dir)
    
    print("\n" + "=" * 70)
    print("FORECASTING PIPELINE COMPLETE")
    print("=" * 70)
    
    return load_forecasts, pv_forecasts, metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run thesis forecasting pipeline')
    parser.add_argument('--input', type=str,
                       default='/Users/mariabigonah/Desktop/thesis/code/outputs/MASTER_20_APARTMENTS_2022_2023.csv',
                       help='Path to master dataset CSV')
    parser.add_argument('--output', type=str,
                       default='/Users/mariabigonah/Desktop/thesis/code/outputs/forecasts',
                       help='Output directory for forecasts')
    
    args = parser.parse_args()
    run_forecasting_pipeline(args.input, args.output)

