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
    load_openweather_hourly,
    merge_pv_weather_sources,
)
from forecasting.pv_forecaster import forecast_pv_timeseries as forecast_gb

# Try importing other forecasters
try:
    from forecasting.pv_forecaster_xgboost import forecast_pv_timeseries as forecast_xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    forecast_xgb = None

try:
    from forecasting.pv_forecaster_lstm import forecast_pv_timeseries as forecast_lstm
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    forecast_lstm = None

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
OPENWEATHER_PATH = DATA_DIR / "openweather_historical.csv"

# Cache for loaded data
_data_cache = {}


def estimate_pv_from_irradiance(irr_direct, irr_diffuse, temp_amb, capacity_kw=15.0, efficiency=0.18):
    """Estimate PV power from irradiance for demonstration."""
    g_eff = irr_direct + irr_diffuse
    pv_power = (g_eff / 1000.0) * capacity_kw * efficiency * (1 - 0.004 * (temp_amb - 25.0))
    return pv_power.clip(lower=0)


def load_all_data():
    """Load and cache all data sources."""
    if 'history_df' in _data_cache:
        return _data_cache['history_df'], _data_cache['pvgis'], _data_cache['openweather']
    
    print("Loading data...")
    pvgis_hourly = load_pvgis_weather_hourly(PVGIS_PATH)
    openweather_hourly = load_openweather_hourly(OPENWEATHER_PATH)
    
    # Estimate PV power
    pv_power = estimate_pv_from_irradiance(
        pvgis_hourly["irr_direct"],
        pvgis_hourly["irr_diffuse"],
        pvgis_hourly["temp_amb"],
    )
    
    # Merge data
    history_df = merge_pv_weather_sources(
        pv_power=pv_power,
        pvgis_hourly=pvgis_hourly,
        openweather_hourly=openweather_hourly,
    )
    
    # Use last 2 years for faster processing
    if len(history_df) > 17520:
        history_df = history_df.tail(17520)
    
    _data_cache['history_df'] = history_df
    _data_cache['pvgis'] = pvgis_hourly
    _data_cache['openweather'] = openweather_hourly
    
    return history_df, pvgis_hourly, openweather_hourly


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
    """Get overview statistics for all databases."""
    history_df, pvgis, openweather = load_all_data()
    
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
        'openweather': {
            'name': 'OpenWeather Historical',
            'rows': len(openweather),
            'columns': list(openweather.columns) if len(openweather) > 0 else [],
            'date_range': {
                'start': openweather.index.min().isoformat() if len(openweather) > 0 else None,
                'end': openweather.index.max().isoformat() if len(openweather) > 0 else None,
            },
            'statistics': openweather.describe().to_dict() if len(openweather) > 0 else {},
        },
        'merged': {
            'name': 'Merged Dataset',
            'rows': len(history_df),
            'columns': list(history_df.columns),
            'date_range': {
                'start': history_df.index.min().isoformat(),
                'end': history_df.index.max().isoformat(),
            },
            'statistics': history_df.describe().to_dict(),
        },
    }
    
    return jsonify(overview)


@app.route('/api/data/features')
def get_features():
    """Get detailed feature information."""
    history_df, _, _ = load_all_data()
    
    features = {}
    for col in history_df.columns:
        if pd.api.types.is_numeric_dtype(history_df[col]):
            features[col] = {
                'dtype': str(history_df[col].dtype),
                'mean': float(history_df[col].mean()),
                'std': float(history_df[col].std()),
                'min': float(history_df[col].min()),
                'max': float(history_df[col].max()),
                'median': float(history_df[col].median()),
                'missing': int(history_df[col].isna().sum()),
                'missing_pct': float(history_df[col].isna().sum() / len(history_df) * 100),
            }
    
    return jsonify(features)


@app.route('/api/data/covariance')
def get_covariance():
    """Get covariance matrix and visualization."""
    history_df, _, _ = load_all_data()
    
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
    history_df, pvgis, openweather = load_all_data()
    
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
    history_df, _, _ = load_all_data()
    
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


@app.route('/api/forecast/visualization')
def forecast_visualization():
    """Generate visualization comparing all forecasting models."""
    # Run forecasts first
    history_df, _, _ = load_all_data()
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
    
    return jsonify({'image': img_base64, 'metrics': metrics_dict})


if __name__ == '__main__':
    print("Starting Energy Management System Dashboard Backend...")
    print("Access the dashboard at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

