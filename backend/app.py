"""
Flask backend API for Energy Management System Dashboard
Provides endpoints for data visualization, feature analysis, and forecasting results.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from forecasting.data_loading import (
    load_pvgis_weather_hourly,
    merge_pv_weather_sources,
)
from forecasting.pv_forecaster import (
    forecast_pv_timeseries as forecast_gb,
    forecast_non_shiftable_load_seasonal_naive,
    classify_devices,
    ShiftableDevice,
    compute_shiftable_load_profiles,
)

# Try importing other forecasters (handle gracefully if dependencies missing)
try:
    from forecasting.pv_forecaster_xgboost import forecast_pv_timeseries as forecast_xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    forecast_xgb = None
    print(f"Note: XGBoost not available: {e}")

try:
    from forecasting.pv_forecaster_lstm import forecast_pv_timeseries as forecast_lstm
    LSTM_AVAILABLE = True
except (ImportError, Exception) as e:
    LSTM_AVAILABLE = False
    forecast_lstm = None
    print(f"Note: LSTM/TensorFlow not available: {e}")

# Get project root directory
project_root = Path(__file__).parent.parent
frontend_path = project_root / "frontend"

app = Flask(__name__, 
            static_folder=str(frontend_path),
            static_url_path='',
            template_folder=str(frontend_path))
CORS(app)

# Data paths
DATA_DIR = Path("/Users/mariabigonah/Desktop/thesis/building database")
PVGIS_PATH = DATA_DIR / "Timeseries_45.044_7.639_SA3_40deg_2deg_2005_2023.csv"
DEVICE_PROFILES_PATH = Path("/Users/mariabigonah/Desktop/thesis/CHR54 Retired Couple, no work/Results/DeviceProfiles.HH1.Electricity.csv")
DEVICE_DURATION_PATH = Path("/Users/mariabigonah/Desktop/thesis/CHR54 Retired Couple, no work/Reports/DeviceDurationCurves.Electricity.csv")

# Cache for loaded data
_data_cache = {}


def estimate_pv_from_irradiance(irr_direct, irr_diffuse, temp_amb, capacity_kw=15.0, efficiency=0.18):
    """Estimate PV power from irradiance for demonstration."""
    g_eff = irr_direct + irr_diffuse
    pv_power = (g_eff / 1000.0) * capacity_kw * efficiency * (1 - 0.004 * (temp_amb - 25.0))
    return pv_power.clip(lower=0)


def load_device_data():
    """Load and process device load profiles."""
    if 'device_data' in _data_cache:
        return _data_cache['device_data']
    
    print("Loading device data...")
    
    # Load device profiles (15-minute resolution)
    try:
        device_profiles_df = pd.read_csv(DEVICE_PROFILES_PATH, sep=';', low_memory=False)
        
        # Parse time column
        if 'Time' in device_profiles_df.columns:
            device_profiles_df['Time'] = pd.to_datetime(device_profiles_df['Time'], errors='coerce')
            device_profiles_df = device_profiles_df.dropna(subset=['Time'])
            device_profiles_df = device_profiles_df.set_index('Time')
        
        # Remove metadata columns
        device_cols = [col for col in device_profiles_df.columns 
                      if col not in ['Electricity.Timestep', 'Time'] 
                      and not col.startswith('Unnamed')]
        
        # Get device names (remove [kWh] suffix)
        device_names = [col.replace(' [kWh]', '') for col in device_cols]
        
        # Classify devices
        device_classification = classify_devices(device_names)
        
        # Separate shiftable and non-shiftable
        shiftable_devices = device_classification[device_classification['category'] == 'shiftable']['device_name'].tolist()
        non_shiftable_devices = device_classification[device_classification['category'] == 'non_shiftable']['device_name'].tolist()
        
        # Map back to original column names
        shiftable_cols = [col for col, name in zip(device_cols, device_names) if name in shiftable_devices]
        non_shiftable_cols = [col for col, name in zip(device_cols, device_names) if name in non_shiftable_devices]
        
        # Aggregate non-shiftable load (sum all non-shiftable devices)
        if non_shiftable_cols:
            non_shiftable_load = device_profiles_df[non_shiftable_cols].sum(axis=1)
            # Convert from kWh to kW (assuming 15-minute intervals)
            non_shiftable_load = non_shiftable_load * 4  # Convert kWh/15min to kW
        else:
            non_shiftable_load = pd.Series(dtype=float)
        
        # Get shiftable device profiles
        shiftable_profiles = {}
        for col, name in zip(shiftable_cols, shiftable_devices):
            if col in device_profiles_df.columns:
                profile = device_profiles_df[col].copy()
                profile = profile * 4  # Convert kWh/15min to kW
                shiftable_profiles[name] = profile
        
        # Resample to hourly
        if len(non_shiftable_load) > 0:
            non_shiftable_hourly = non_shiftable_load.resample('1h').mean()
        else:
            non_shiftable_hourly = pd.Series(dtype=float)
        
        shiftable_hourly = {}
        for name, profile in shiftable_profiles.items():
            shiftable_hourly[name] = profile.resample('1h').mean()
        
        device_data = {
            'non_shiftable_load': non_shiftable_hourly,
            'shiftable_profiles': shiftable_hourly,
            'classification': device_classification,
            'shiftable_devices': shiftable_devices,
            'non_shiftable_devices': non_shiftable_devices,
        }
        
    except Exception as e:
        print(f"Error loading device data: {e}")
        device_data = {
            'non_shiftable_load': pd.Series(dtype=float),
            'shiftable_profiles': {},
            'classification': pd.DataFrame(),
            'shiftable_devices': [],
            'non_shiftable_devices': [],
        }
    
    _data_cache['device_data'] = device_data
    return device_data


def load_all_data():
    """Load and cache all data sources (PVGIS only)."""
    if 'history_df' in _data_cache:
        return _data_cache['history_df'], _data_cache['pvgis']
    
    print("Loading data...")
    pvgis_hourly = load_pvgis_weather_hourly(PVGIS_PATH)
    
    # Estimate PV power
    pv_power = estimate_pv_from_irradiance(
        pvgis_hourly["irr_direct"],
        pvgis_hourly["irr_diffuse"],
        pvgis_hourly["temp_amb"],
    )
    
    # Merge data (PVGIS only, no OpenWeather)
    history_df = merge_pv_weather_sources(
        pv_power=pv_power,
        pvgis_hourly=pvgis_hourly,
    )
    
    # Use last 2 years for faster processing
    if len(history_df) > 17520:
        history_df = history_df.tail(17520)
    
    _data_cache['history_df'] = history_df
    _data_cache['pvgis'] = pvgis_hourly
    
    return history_df, pvgis_hourly


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


@app.route('/')
def index():
    """Serve the main dashboard page."""
    return send_from_directory(str(frontend_path), 'index.html')


@app.route('/api/data/overview')
def data_overview():
    """Get overview statistics for all databases (PVGIS and device loads)."""
    history_df, pvgis = load_all_data()
    device_data = load_device_data()
    
    # Prepare device load statistics
    non_shiftable_load = device_data['non_shiftable_load']
    shiftable_profiles = device_data['shiftable_profiles']
    
    overview = {
        'pvgis': {
            'name': 'PVGIS SARAH3',
            'rows': len(pvgis),
            'columns': list(pvgis.columns),
            'date_range': {
                'start': pvgis.index.min().isoformat(),
                'end': pvgis.index.max().isoformat(),
            },
            'statistics': pvgis.describe().to_dict(),
        },
        'merged': {
            'name': 'Merged Dataset (PVGIS)',
            'rows': len(history_df),
            'columns': list(history_df.columns),
            'date_range': {
                'start': history_df.index.min().isoformat(),
                'end': history_df.index.max().isoformat(),
            },
            'statistics': history_df.describe().to_dict(),
        },
        'non_shiftable_load': {
            'name': 'Non-Shiftable Load Profile',
            'rows': len(non_shiftable_load),
            'columns': ['non_shiftable_load_kw'],
            'date_range': {
                'start': non_shiftable_load.index.min().isoformat() if len(non_shiftable_load) > 0 else None,
                'end': non_shiftable_load.index.max().isoformat() if len(non_shiftable_load) > 0 else None,
            },
            'statistics': non_shiftable_load.describe().to_dict() if len(non_shiftable_load) > 0 else {},
            'device_count': len(device_data['non_shiftable_devices']),
            'devices': device_data['non_shiftable_devices'][:10],  # First 10 devices
        },
        'shiftable_load': {
            'name': 'Shiftable Load Profiles',
            'rows': len(list(shiftable_profiles.values())[0]) if shiftable_profiles else 0,
            'columns': list(shiftable_profiles.keys()),
            'date_range': {
                'start': list(shiftable_profiles.values())[0].index.min().isoformat() if shiftable_profiles else None,
                'end': list(shiftable_profiles.values())[0].index.max().isoformat() if shiftable_profiles else None,
            },
            'statistics': {name: profile.describe().to_dict() for name, profile in list(shiftable_profiles.items())[:5]} if shiftable_profiles else {},
            'device_count': len(device_data['shiftable_devices']),
            'devices': device_data['shiftable_devices'],
        },
    }
    
    return jsonify(overview)


@app.route('/api/data/features')
def get_features():
    """Get detailed feature information for all data sources."""
    history_df, pvgis = load_all_data()
    device_data = load_device_data()
    
    features = {
        'pvgis': {},
        'merged': {},
        'non_shiftable_load': {},
        'shiftable_load': {},
    }
    
    # PVGIS features
    for col in pvgis.columns:
        if pd.api.types.is_numeric_dtype(pvgis[col]):
            features['pvgis'][col] = {
                'dtype': str(pvgis[col].dtype),
                'mean': float(pvgis[col].mean()),
                'std': float(pvgis[col].std()),
                'min': float(pvgis[col].min()),
                'max': float(pvgis[col].max()),
                'median': float(pvgis[col].median()),
                'missing': int(pvgis[col].isna().sum()),
                'missing_pct': float(pvgis[col].isna().sum() / len(pvgis) * 100),
            }
    
    # Merged dataset features
    for col in history_df.columns:
        if pd.api.types.is_numeric_dtype(history_df[col]):
            features['merged'][col] = {
                'dtype': str(history_df[col].dtype),
                'mean': float(history_df[col].mean()),
                'std': float(history_df[col].std()),
                'min': float(history_df[col].min()),
                'max': float(history_df[col].max()),
                'median': float(history_df[col].median()),
                'missing': int(history_df[col].isna().sum()),
                'missing_pct': float(history_df[col].isna().sum() / len(history_df) * 100),
            }
    
    # Non-shiftable load features
    non_shiftable_load = device_data['non_shiftable_load']
    if len(non_shiftable_load) > 0:
        features['non_shiftable_load']['non_shiftable_load_kw'] = {
            'dtype': str(non_shiftable_load.dtype),
            'mean': float(non_shiftable_load.mean()),
            'std': float(non_shiftable_load.std()),
            'min': float(non_shiftable_load.min()),
            'max': float(non_shiftable_load.max()),
            'median': float(non_shiftable_load.median()),
            'missing': int(non_shiftable_load.isna().sum()),
            'missing_pct': float(non_shiftable_load.isna().sum() / len(non_shiftable_load) * 100),
            'description': 'Aggregated load from all non-shiftable devices',
            'device_count': len(device_data['non_shiftable_devices']),
        }
    
    # Shiftable load features (individual devices)
    shiftable_profiles = device_data['shiftable_profiles']
    for device_name, profile in shiftable_profiles.items():
        if len(profile) > 0:
            features['shiftable_load'][device_name] = {
                'dtype': str(profile.dtype),
                'mean': float(profile.mean()),
                'std': float(profile.std()),
                'min': float(profile.min()),
                'max': float(profile.max()),
                'median': float(profile.median()),
                'missing': int(profile.isna().sum()),
                'missing_pct': float(profile.isna().sum() / len(profile) * 100),
                'description': 'Shiftable device power profile',
            }
    
    return jsonify(features)


@app.route('/api/data/covariance')
def get_covariance():
    """Get covariance matrix and visualization."""
    history_df, _ = load_all_data()
    
    # Select numeric columns only
    numeric_cols = history_df.select_dtypes(include=[np.number]).columns
    cov_matrix = history_df[numeric_cols].cov()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cov_matrix, annot=True, fmt='.4f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Feature Covariance Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    img_base64 = fig_to_base64(fig)
    
    return jsonify({
        'covariance_matrix': cov_matrix.to_dict(),
        'image': img_base64,
        'columns': list(cov_matrix.columns),
    })


@app.route('/api/data/visualization')
def data_visualization():
    """Generate database visualization figures."""
    history_df, pvgis = load_all_data()
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. PV Power time series
    ax1 = fig.add_subplot(gs[0, :])
    if 'pv_power' in history_df.columns:
        sample_data = history_df['pv_power'].tail(1000)  # Last 1000 hours
        ax1.plot(sample_data.index, sample_data.values, linewidth=1, alpha=0.7)
        ax1.set_title('PV Power Time Series (Last 1000 Hours)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('PV Power (kW)')
        ax1.grid(True, alpha=0.3)
    
    # 2. Temperature distribution
    ax2 = fig.add_subplot(gs[1, 0])
    if 'temp_amb' in history_df.columns:
        ax2.hist(history_df['temp_amb'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        ax2.set_title('Ambient Temperature Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Irradiance distribution
    ax3 = fig.add_subplot(gs[1, 1])
    if 'irr_direct' in history_df.columns:
        ax3.hist(history_df['irr_direct'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax3.set_title('Direct Irradiance Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Irradiance (W/m²)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Feature correlation (subset)
    ax4 = fig.add_subplot(gs[2, 0])
    numeric_cols = history_df.select_dtypes(include=[np.number]).columns[:5]  # First 5 numeric
    if len(numeric_cols) > 1:
        corr_subset = history_df[numeric_cols].corr()
        im = ax4.imshow(corr_subset, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(corr_subset.columns)))
        ax4.set_yticks(range(len(corr_subset.columns)))
        ax4.set_xticklabels(corr_subset.columns, rotation=45, ha='right')
        ax4.set_yticklabels(corr_subset.columns)
        ax4.set_title('Feature Correlation (Subset)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax4)
    
    # 5. Daily pattern
    ax5 = fig.add_subplot(gs[2, 1])
    if 'pv_power' in history_df.columns:
        history_df['hour'] = history_df.index.hour
        daily_pattern = history_df.groupby('hour')['pv_power'].mean()
        ax5.plot(daily_pattern.index, daily_pattern.values, marker='o', linewidth=2)
        ax5.set_title('Average Daily PV Power Pattern', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Hour of Day')
        ax5.set_ylabel('Average PV Power (kW)')
        ax5.grid(True, alpha=0.3)
        ax5.set_xticks(range(0, 24, 2))
    
    img_base64 = fig_to_base64(fig)
    
    return jsonify({'image': img_base64})


@app.route('/api/forecast/run')
def run_forecasts():
    """Run all available forecasting models and return results."""
    history_df, _ = load_all_data()
    
    # Prepare forecast weather (use last 24 hours as "forecast")
    weather_forecast_df = history_df[["temp_amb", "irr_direct", "irr_diffuse"]].iloc[-24:].copy()
    weather_forecast_df.index = history_df.index[-24:] + pd.Timedelta(days=1)
    
    static_features = {
        "tilt_deg": 40,
        "azimuth_deg": 2,
        "capacity_kw": 15.0,
    }
    
    results = {}
    
    # GradientBoosting
    try:
        print("Running GradientBoosting forecast...")
        pv_forecast_gb, metrics_gb, _ = forecast_gb(
            history_df=history_df,
            weather_forecast_df=weather_forecast_df,
            static_features=static_features,
            lag_hours=24,
            val_size=0.1,
            test_size=0.1,
        )
        results['gradientboosting'] = {
            'forecast': pv_forecast_gb.to_dict(),
            'metrics': metrics_gb,
            'available': True,
        }
    except Exception as e:
        results['gradientboosting'] = {
            'available': False,
            'error': str(e),
        }
    
    # XGBoost
    if XGBOOST_AVAILABLE and forecast_xgb:
        try:
            print("Running XGBoost forecast...")
            pv_forecast_xgb, metrics_xgb, _ = forecast_xgb(
                history_df=history_df,
                weather_forecast_df=weather_forecast_df,
                static_features=static_features,
                lag_hours=24,
                val_size=0.1,
                test_size=0.1,
            )
            results['xgboost'] = {
                'forecast': pv_forecast_xgb.to_dict(),
                'metrics': metrics_xgb,
                'available': True,
            }
        except Exception as e:
            results['xgboost'] = {
                'available': False,
                'error': str(e),
            }
    else:
        results['xgboost'] = {'available': False, 'error': 'XGBoost not installed'}
    
    # LSTM
    if LSTM_AVAILABLE and forecast_lstm:
        try:
            print("Running LSTM forecast...")
            pv_forecast_lstm, metrics_lstm, _ = forecast_lstm(
                history_df=history_df,
                weather_forecast_df=weather_forecast_df,
                static_features=static_features,
                lag_hours=24,
                val_size=0.1,
                test_size=0.1,
                epochs=20,  # Reduced for faster response
                batch_size=32,
                verbose=0,
            )
            results['lstm'] = {
                'forecast': pv_forecast_lstm.to_dict(),
                'metrics': metrics_lstm,
                'available': True,
            }
        except Exception as e:
            results['lstm'] = {
                'available': False,
                'error': str(e),
            }
    else:
        results['lstm'] = {'available': False, 'error': 'TensorFlow not installed'}
    
    return jsonify(results)


@app.route('/api/forecast/individual/<model_name>')
def forecast_individual(model_name):
    """Get individual forecast results and visualization for a specific model."""
    history_df, _ = load_all_data()
    weather_forecast_df = history_df[["temp_amb", "irr_direct", "irr_diffuse"]].iloc[-24:].copy()
    weather_forecast_df.index = history_df.index[-24:] + pd.Timedelta(days=1)
    static_features = {"tilt_deg": 40, "azimuth_deg": 2, "capacity_kw": 15.0}
    
    model_name_lower = model_name.lower()
    forecast_func = None
    model_display_name = ""
    
    if model_name_lower == "gradientboosting":
        forecast_func = forecast_gb
        model_display_name = "GradientBoosting"
    elif model_name_lower == "xgboost" and XGBOOST_AVAILABLE and forecast_xgb:
        forecast_func = forecast_xgb
        model_display_name = "XGBoost"
    elif model_name_lower == "lstm" and LSTM_AVAILABLE and forecast_lstm:
        forecast_func = forecast_lstm
        model_display_name = "LSTM"
    else:
        return jsonify({'error': f'Model {model_name} not available'}), 404
    
    try:
        # Run forecast
        if model_name_lower == "lstm":
            pv_forecast, metrics, _ = forecast_func(
                history_df=history_df, weather_forecast_df=weather_forecast_df,
                static_features=static_features, lag_hours=24, val_size=0.1, test_size=0.1,
                epochs=20, batch_size=32, verbose=0,
            )
        else:
            pv_forecast, metrics, _ = forecast_func(
                history_df=history_df, weather_forecast_df=weather_forecast_df,
                static_features=static_features, lag_hours=24, val_size=0.1, test_size=0.1,
            )
        
        # Create individual visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Forecast time series
        ax1 = axes[0, 0]
        ax1.plot(pv_forecast.index, pv_forecast.values, marker='o', linewidth=2, 
                markersize=6, color='steelblue', label='Forecast')
        ax1.set_title(f'{model_display_name}: 24-Hour PV Forecast', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('PV Power (kW)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Validation metrics
        ax2 = axes[0, 1]
        val_metrics = ['MAE', 'RMSE']
        val_values = [metrics['validation']['mae'], metrics['validation']['rmse']]
        ax2.bar(val_metrics, val_values, color='coral', alpha=0.7)
        ax2.set_title('Validation Set Metrics', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Error (kW)')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Test metrics
        ax3 = axes[1, 0]
        test_metrics = ['MAE', 'RMSE']
        test_values = [metrics['test']['mae'], metrics['test']['rmse']]
        ax3.bar(test_metrics, test_values, color='green', alpha=0.7)
        ax3.set_title('Test Set Metrics', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Error (kW)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. R² comparison
        ax4 = axes[1, 1]
        r2_data = {
            'Validation': metrics['validation']['r2'],
            'Test': metrics['test']['r2']
        }
        ax4.bar(r2_data.keys(), r2_data.values(), color=['coral', 'green'], alpha=0.7)
        ax4.set_title('Coefficient of Determination (R²)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('R²')
        ax4.set_ylim([0, 1])
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        img_base64 = fig_to_base64(fig)
        
        # Convert forecast to dict with string keys for JSON serialization
        forecast_dict = {str(k): float(v) for k, v in pv_forecast.items()}
        
        # Ensure metrics are JSON-serializable (convert any Timestamps or other non-serializable types)
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {str(k): make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
                return str(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        serializable_metrics = make_json_serializable(metrics)
        
        return jsonify({
            'model': model_display_name,
            'forecast': forecast_dict,
            'metrics': serializable_metrics,
            'image': img_base64,
            'available': True,
        })
    except Exception as e:
        return jsonify({
            'model': model_display_name,
            'available': False,
            'error': str(e),
        }), 500


@app.route('/api/forecast/visualization')
def forecast_visualization():
    """Generate visualization comparing all forecasting models."""
    # Run forecasts first
    history_df, _ = load_all_data()
    weather_forecast_df = history_df[["temp_amb", "irr_direct", "irr_diffuse"]].iloc[-24:].copy()
    weather_forecast_df.index = history_df.index[-24:] + pd.Timedelta(days=1)
    static_features = {"tilt_deg": 40, "azimuth_deg": 2, "capacity_kw": 15.0}
    
    forecasts = {}
    metrics_dict = {}
    
    # Collect forecasts
    try:
        pv_forecast_gb, metrics_gb, _ = forecast_gb(
            history_df=history_df, weather_forecast_df=weather_forecast_df,
            static_features=static_features, lag_hours=24, val_size=0.1, test_size=0.1,
        )
        forecasts['GradientBoosting'] = pv_forecast_gb
        metrics_dict['GradientBoosting'] = metrics_gb
    except:
        pass
    
    if XGBOOST_AVAILABLE and forecast_xgb:
        try:
            pv_forecast_xgb, metrics_xgb, _ = forecast_xgb(
                history_df=history_df, weather_forecast_df=weather_forecast_df,
                static_features=static_features, lag_hours=24, val_size=0.1, test_size=0.1,
            )
            forecasts['XGBoost'] = pv_forecast_xgb
            metrics_dict['XGBoost'] = metrics_xgb
        except:
            pass
    
    if LSTM_AVAILABLE and forecast_lstm:
        try:
            pv_forecast_lstm, metrics_lstm, _ = forecast_lstm(
                history_df=history_df, weather_forecast_df=weather_forecast_df,
                static_features=static_features, lag_hours=24, val_size=0.1, test_size=0.1,
                epochs=20, batch_size=32, verbose=0,
            )
            forecasts['LSTM'] = pv_forecast_lstm
            metrics_dict['LSTM'] = metrics_lstm
        except:
            pass
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Forecast comparison
    ax1 = axes[0, 0]
    colors = ['steelblue', 'coral', 'green']
    markers = ['o', 's', '^']
    for idx, (name, forecast) in enumerate(forecasts.items()):
        ax1.plot(forecast.index, forecast.values, marker=markers[idx], 
                linestyle='-', label=name, linewidth=2, markersize=6, color=colors[idx])
    ax1.set_title('24-Hour PV Forecast Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('PV Power (kW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Test MAE comparison
    ax2 = axes[0, 1]
    models = list(metrics_dict.keys())
    test_mae = [metrics_dict[m]['test']['mae'] for m in models]
    ax2.bar(models, test_mae, color=colors[:len(models)], alpha=0.7)
    ax2.set_title('Test Set: Mean Absolute Error', fontsize=14, fontweight='bold')
    ax2.set_ylabel('MAE (kW)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Test RMSE comparison
    ax3 = axes[1, 0]
    test_rmse = [metrics_dict[m]['test']['rmse'] for m in models]
    ax3.bar(models, test_rmse, color=colors[:len(models)], alpha=0.7)
    ax3.set_title('Test Set: Root Mean Squared Error', fontsize=14, fontweight='bold')
    ax3.set_ylabel('RMSE (kW)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Test R² comparison
    ax4 = axes[1, 1]
    test_r2 = [metrics_dict[m]['test']['r2'] for m in models]
    ax4.bar(models, test_r2, color=colors[:len(models)], alpha=0.7)
    ax4.set_title('Test Set: Coefficient of Determination', fontsize=14, fontweight='bold')
    ax4.set_ylabel('R²')
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    img_base64 = fig_to_base64(fig)
    
    # Ensure metrics are JSON-serializable
    def make_json_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    serializable_metrics = make_json_serializable(metrics_dict)
    
    return jsonify({'image': img_base64, 'metrics': serializable_metrics})


@app.route('/api/forecast/non_shiftable')
def forecast_non_shiftable():
    """Generate non-shiftable load forecast using seasonal naive method."""
    device_data = load_device_data()
    non_shiftable_load = device_data['non_shiftable_load']
    
    if len(non_shiftable_load) == 0:
        return jsonify({
            'available': False,
            'error': 'No non-shiftable load data available'
        }), 404
    
    try:
        # Use last 2 weeks of data for forecast (need at least 168 hours)
        if len(non_shiftable_load) < 168:
            return jsonify({
                'available': False,
                'error': f'Insufficient data: need at least 168 hours, got {len(non_shiftable_load)}'
            }), 400
        
        # Get recent history for visualization
        recent_history = non_shiftable_load.tail(336)  # Last 2 weeks
        
        # Generate 24-hour forecast
        forecast = forecast_non_shiftable_load_seasonal_naive(
            load_history=non_shiftable_load,
            horizon_hours=24,
            season_hours=168,  # Weekly pattern
        )
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # 1. Historical data and forecast
        ax1 = axes[0]
        # Show last week of history
        history_week = recent_history.tail(168)
        ax1.plot(history_week.index, history_week.values, 
                linewidth=2, color='steelblue', label='Historical Load (Last Week)', alpha=0.7)
        ax1.plot(forecast.index, forecast.values, 
                marker='o', linewidth=2, markersize=6, color='coral', 
                label='24-Hour Forecast (Seasonal Naive)', linestyle='--')
        ax1.set_title('Non-Shiftable Load: Historical vs Forecast', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Load (kW)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Forecast details
        ax2 = axes[1]
        ax2.plot(forecast.index, forecast.values, marker='o', linewidth=2, 
                markersize=8, color='coral', label='24-Hour Forecast')
        ax2.fill_between(forecast.index, forecast.values, alpha=0.3, color='coral')
        ax2.set_title('24-Hour Non-Shiftable Load Forecast (Seasonal Naive)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Load (kW)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        img_base64 = fig_to_base64(fig)
        
        # Calculate forecast statistics
        forecast_stats = {
            'mean': float(forecast.mean()),
            'std': float(forecast.std()),
            'min': float(forecast.min()),
            'max': float(forecast.max()),
            'total': float(forecast.sum()),
        }
        
        # Convert forecast to dict with string keys
        forecast_dict = {str(k): float(v) for k, v in forecast.items()}
        
        return jsonify({
            'available': True,
            'forecast': forecast_dict,
            'statistics': forecast_stats,
            'method': 'Seasonal Naive (Weekly Pattern)',
            'description': 'Forecast rule: forecast_load[t] = realized_load[t - 168 hours]',
            'image': img_base64,
        })
    except Exception as e:
        return jsonify({
            'available': False,
            'error': str(e),
        }), 500


if __name__ == '__main__':
    print("Starting Energy Management System Dashboard Backend...")
    print("Access the dashboard at: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)

