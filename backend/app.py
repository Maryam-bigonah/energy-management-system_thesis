"""
Flask Backend API for Energy Forecasting Visualization
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import sys

# Add parent directory to path to import LSTM model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    from lstm_energy_forecast import EnergyForecastLSTM
    from data_loader import create_sample_torino_data, combine_data, load_pvgis_data, load_lpg_data
    from shared_battery_model import simulate_shared_battery_torino
    from tariffs_model import TariffsModel
    from energy_economic_analysis import run_complete_analysis
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Make sure all required modules are in the parent directory")

app = Flask(__name__)
CORS(app)

# Global variables
model = None
df = None
df_master = None  # Master dataset with all apartments
battery_results = None  # Battery simulation results
economic_results = None  # Economic analysis results
model_trained = False

# Data storage (in production, use a database)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_trained': model_trained
    })


@app.route('/api/data/load-sample', methods=['POST'])
def load_sample_data():
    """Load sample Torino data"""
    global df
    
    try:
        data = request.get_json() or {}
        start_date = data.get('start_date', '2023-01-01')
        end_date = data.get('end_date', '2023-12-31')
        
        df = create_sample_torino_data(start_date=start_date, end_date=end_date)
        
        return jsonify({
            'success': True,
            'message': f'Sample data loaded: {len(df)} records',
            'date_range': {
                'start': df.index.min().isoformat(),
                'end': df.index.max().isoformat()
            },
            'stats': {
                'load_mean': float(df['load'].mean()),
                'load_std': float(df['load'].std()),
                'pv_mean': float(df['pv'].mean()),
                'pv_std': float(df['pv'].std())
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/data/upload', methods=['POST'])
def upload_data():
    """Upload CSV data files"""
    global df
    
    try:
        if 'pv_file' not in request.files or 'load_file' not in request.files:
            return jsonify({'success': False, 'error': 'Both PV and Load files required'}), 400
        
        pv_file = request.files['pv_file']
        load_file = request.files['load_file']
        
        # Save temporarily
        pv_path = os.path.join(DATA_DIR, 'temp_pv.csv')
        load_path = os.path.join(DATA_DIR, 'temp_load.csv')
        
        os.makedirs(DATA_DIR, exist_ok=True)
        pv_file.save(pv_path)
        load_file.save(load_path)
        
        # Load and combine
        pv_df = load_pvgis_data(pv_path)
        load_df = load_lpg_data(load_path)
        df = combine_data(pv_df, load_df)
        
        return jsonify({
            'success': True,
            'message': f'Data loaded: {len(df)} records',
            'date_range': {
                'start': df.index.min().isoformat(),
                'end': df.index.max().isoformat()
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/data/historical', methods=['GET'])
def get_historical_data():
    """Get historical data for visualization"""
    global df, df_master
    
    # Use master dataset if available, otherwise fall back to simple df
    data_source = df_master if df_master is not None else df
    
    if data_source is None or len(data_source) == 0:
        return jsonify({'success': False, 'error': 'No data loaded'}), 400
    
    try:
        # Get date range from query params
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        data_df = data_source.copy()
        
        if start_date:
            data_df = data_df[data_df.index >= start_date]
        if end_date:
            data_df = data_df[data_df.index <= end_date]
        
        # Limit to last 1000 points for performance, or specific range
        limit = int(request.args.get('limit', 1000))
        if len(data_df) > limit:
            data_df = data_df.iloc[-limit:]
        
        # Build response based on available columns
        apt_cols = [col for col in data_df.columns if col.startswith('apartment_')]
        apt_cols.sort()
        
        data = {
            'timestamps': [ts.isoformat() for ts in data_df.index],
        }
        
        # Add PV
        if 'pv_1kw' in data_df.columns:
            data['pv'] = data_df['pv_1kw'].values.tolist()
        elif 'pv' in data_df.columns:
            data['pv'] = data_df['pv'].values.tolist()
        
        # Add apartment loads
        if apt_cols:
            data['apartments'] = {}
            for col in apt_cols:
                data['apartments'][col] = data_df[col].values.tolist()
            data['total_load'] = data_df[apt_cols].sum(axis=1).values.tolist()
        elif 'load' in data_df.columns:
            data['load'] = data_df['load'].values.tolist()
        
        # Add calendar features if available
        if 'hour' in data_df.columns:
            data['hour'] = data_df['hour'].values.tolist()
        if 'dayofweek' in data_df.columns:
            data['dayofweek'] = data_df['dayofweek'].values.tolist()
        if 'month' in data_df.columns:
            data['month'] = data_df['month'].values.tolist()
        if 'season' in data_df.columns:
            data['season'] = data_df['season'].values.tolist()
        
        return jsonify({
            'success': True,
            'data': data,
            'count': len(data_df),
            'has_apartments': len(apt_cols) > 0,
            'has_calendar': 'hour' in data_df.columns
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/model/train', methods=['POST'])
def train_model():
    """Train the LSTM model"""
    global model, df, model_trained
    
    if df is None or len(df) == 0:
        return jsonify({'success': False, 'error': 'No data loaded. Please load data first.'}), 400
    
    try:
        data = request.get_json() or {}
        epochs = int(data.get('epochs', 50))
        batch_size = int(data.get('batch_size', 32))
        validation_split = float(data.get('validation_split', 0.2))
        
        # Initialize and train model
        model = EnergyForecastLSTM(lookback_hours=24, forecast_hours=1)
        
        history = model.train(
            df,
            target_cols=['load', 'pv'],
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size
        )
        
        model_trained = True
        
        # Extract training history
        history_dict = {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'mae': [float(x) for x in history.history['mae']],
            'val_mae': [float(x) for x in history.history['val_mae']]
        }
        
        # Evaluate on validation set
        split_idx = int(len(df) * (1 - validation_split))
        df_val = df.iloc[split_idx:]
        metrics = model.evaluate(df_val)
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'history': history_dict,
            'metrics': {
                'load': {k: float(v) for k, v in metrics['load'].items()},
                'pv': {k: float(v) for k, v in metrics['pv'].items()}
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/model/predict', methods=['POST'])
def predict():
    """Make predictions"""
    global model, df
    
    if model is None or not model_trained:
        return jsonify({'success': False, 'error': 'Model not trained. Please train the model first.'}), 400
    
    if df is None or len(df) == 0:
        return jsonify({'success': False, 'error': 'No data available'}), 400
    
    try:
        data = request.get_json() or {}
        hours = int(data.get('hours', 168))  # Default: 1 week
        
        # Get last N hours for prediction
        df_pred = df.iloc[-min(hours, len(df)):]
        
        # Make predictions
        predictions = model.predict(df_pred)
        
        # Get actual values for comparison
        actual_df = df.iloc[-len(predictions):]
        
        # Format response
        result = {
            'timestamps': [ts.isoformat() for ts in predictions.index],
            'predictions': {
                'load': predictions['load_pred'].values.tolist(),
                'pv': predictions['pv_pred'].values.tolist()
            },
            'actual': {
                'load': actual_df['load'].values.tolist(),
                'pv': actual_df['pv'].values.tolist()
            }
        }
        
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/model/metrics', methods=['GET'])
def get_metrics():
    """Get model performance metrics"""
    global model, df
    
    if model is None or not model_trained:
        return jsonify({'success': False, 'error': 'Model not trained'}), 400
    
    if df is None:
        return jsonify({'success': False, 'error': 'No data available'}), 400
    
    try:
        # Use last 20% for evaluation
        split_idx = int(len(df) * 0.8)
        df_test = df.iloc[split_idx:]
        
        metrics = model.evaluate(df_test)
        
        return jsonify({
            'success': True,
            'metrics': {
                'load': {k: float(v) for k, v in metrics['load'].items()},
                'pv': {k: float(v) for k, v in metrics['pv'].items()}
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/model/forecast-next', methods=['GET'])
def forecast_next_hour():
    """Forecast next hour"""
    global model, df
    
    if model is None or not model_trained:
        return jsonify({'success': False, 'error': 'Model not trained'}), 400
    
    if df is None or len(df) < 24:
        return jsonify({'success': False, 'error': 'Not enough data (need at least 24 hours)'}), 400
    
    try:
        # Get last 24 hours
        df_last_24h = df.iloc[-24:]
        predictions = model.predict(df_last_24h)
        
        # Get the last prediction (next hour)
        next_hour = {
            'timestamp': predictions.index[-1].isoformat(),
            'load_pred': float(predictions['load_pred'].iloc[-1]),
            'pv_pred': float(predictions['pv_pred'].iloc[-1])
        }
        
        return jsonify({
            'success': True,
            'forecast': next_hour
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/data/master-dataset', methods=['GET'])
def get_master_dataset():
    """Get complete master dataset with all features"""
    global df_master
    
    if df_master is None or len(df_master) == 0:
        return jsonify({'success': False, 'error': 'Master dataset not loaded'}), 400
    
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = int(request.args.get('limit', 1000))
        
        data_df = df_master.copy()
        
        if start_date:
            data_df = data_df[data_df.index >= start_date]
        if end_date:
            data_df = data_df[data_df.index <= end_date]
        
        if len(data_df) > limit:
            data_df = data_df.iloc[-limit:]
        
        # Get all column names
        apt_cols = [col for col in data_df.columns if col.startswith('apartment_')]
        apt_cols.sort()
        calendar_cols = ['hour', 'dayofweek', 'month', 'is_weekend', 'season']
        other_cols = [col for col in data_df.columns if col not in apt_cols and col not in calendar_cols]
        
        # Build comprehensive response
        result = {
            'timestamps': [ts.isoformat() for ts in data_df.index],
            'apartments': {},
            'pv': data_df['pv_1kw'].values.tolist() if 'pv_1kw' in data_df.columns else [],
            'total_load': data_df[apt_cols].sum(axis=1).values.tolist() if apt_cols else [],
            'calendar': {}
        }
        
        # Add apartment data
        for col in apt_cols:
            result['apartments'][col] = data_df[col].values.tolist()
        
        # Add calendar features
        for col in calendar_cols:
            if col in data_df.columns:
                result['calendar'][col] = data_df[col].values.tolist()
        
        # Add other columns (battery, grid, etc.)
        for col in other_cols:
            if col in data_df.columns and data_df[col].dtype in ['float64', 'int64']:
                result[col] = data_df[col].values.tolist()
        
        return jsonify({
            'success': True,
            'data': result,
            'metadata': {
                'count': len(data_df),
                'n_apartments': len(apt_cols),
                'date_range': {
                    'start': data_df.index[0].isoformat(),
                    'end': data_df.index[-1].isoformat()
                },
                'columns': {
                    'apartments': apt_cols,
                    'calendar': [c for c in calendar_cols if c in data_df.columns],
                    'other': other_cols
                }
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/battery/simulate', methods=['POST'])
def simulate_battery():
    """Run battery simulation on master dataset"""
    global df_master, battery_results
    
    if df_master is None or len(df_master) == 0:
        return jsonify({'success': False, 'error': 'Master dataset not loaded'}), 400
    
    try:
        data = request.get_json() or {}
        battery_capacity = float(data.get('capacity_kwh', 20.0))
        allocation_method = data.get('allocation_method', 'energy_share')
        initial_soc = float(data.get('initial_soc', 0.5))
        
        results, model = simulate_shared_battery_torino(
            df_master=df_master,
            battery_capacity_kwh=battery_capacity,
            allocation_method=allocation_method,
            initial_soc=initial_soc
        )
        
        battery_results = results
        
        # Convert to JSON format
        limit = int(data.get('limit', 1000))
        if len(results) > limit:
            results = results.iloc[-limit:]
        
        response_data = {
            'timestamps': [ts.isoformat() for ts in results.index],
            'battery_soc': results['battery_soc'].values.tolist(),
            'battery_charge_total': results['battery_charge_total'].values.tolist(),
            'battery_discharge_total': results['battery_discharge_total'].values.tolist(),
            'building_pv': results['building_pv'].values.tolist() if 'building_pv' in results.columns else [],
            'building_total_load': results['building_total_load'].values.tolist() if 'building_total_load' in results.columns else [],
            'grid_import': results['grid_import'].values.tolist() if 'grid_import' in results.columns else [],
            'grid_export': results['grid_export'].values.tolist() if 'grid_export' in results.columns else [],
            'unit_allocation': {}
        }
        
        # Add unit-level allocation
        apt_cols = [col for col in results.columns if 'apartment' in col and 'battery_charge' in col]
        for col in apt_cols:
            unit_name = col.replace('_battery_charge', '')
            response_data['unit_allocation'][unit_name] = {
                'charge': results[col].values.tolist(),
                'discharge': results[unit_name + '_battery_discharge'].values.tolist() if unit_name + '_battery_discharge' in results.columns else []
            }
        
        summary = model.get_allocation_summary(results) if hasattr(model, 'get_allocation_summary') else {}
        
        return jsonify({
            'success': True,
            'data': response_data,
            'summary': summary,
            'count': len(results)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/battery/data', methods=['GET'])
def get_battery_data():
    """Get battery simulation results"""
    global battery_results
    
    if battery_results is None or len(battery_results) == 0:
        return jsonify({'success': False, 'error': 'No battery simulation results available'}), 400
    
    try:
        limit = int(request.args.get('limit', 1000))
        data_df = battery_results.copy()
        
        if len(data_df) > limit:
            data_df = data_df.iloc[-limit:]
        
        # Similar format as simulate endpoint
        response_data = {
            'timestamps': [ts.isoformat() for ts in data_df.index],
            'battery_soc': data_df['battery_soc'].values.tolist(),
            'battery_charge_total': data_df['battery_charge_total'].values.tolist(),
            'battery_discharge_total': data_df['battery_discharge_total'].values.tolist(),
            'grid_import': data_df['grid_import'].values.tolist() if 'grid_import' in data_df.columns else [],
            'grid_export': data_df['grid_export'].values.tolist() if 'grid_export' in data_df.columns else []
        }
        
        return jsonify({
            'success': True,
            'data': response_data,
            'count': len(data_df)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/economic/analyze', methods=['POST'])
def analyze_economic():
    """Run economic analysis with tariffs"""
    global battery_results, economic_results
    
    if battery_results is None or len(battery_results) == 0:
        return jsonify({'success': False, 'error': 'Need battery simulation results first'}), 400
    
    try:
        data = request.get_json() or {}
        tariffs_csv = data.get('tariffs_csv')
        fit_csv = data.get('fit_csv')
        
        if not tariffs_csv and not fit_csv:
            return jsonify({'success': False, 'error': 'Need tariffs_csv or fit_csv'}), 400
        
        tariffs = TariffsModel(
            tariffs_csv=tariffs_csv if tariffs_csv else None,
            fit_csv=fit_csv if fit_csv else None
        )
        
        econ_results, summary = tariffs.calculate_cost_revenue(
            battery_results,
            grid_import_col='grid_import',
            grid_export_col='grid_export'
        )
        
        economic_results = econ_results
        
        limit = int(data.get('limit', 1000))
        if len(econ_results) > limit:
            econ_results = econ_results.iloc[-limit:]
        
        response_data = {
            'timestamps': [ts.isoformat() for ts in econ_results.index],
            'grid_import_cost': econ_results['grid_import_cost'].values.tolist() if 'grid_import_cost' in econ_results.columns else [],
            'grid_export_revenue': econ_results['grid_export_revenue'].values.tolist() if 'grid_export_revenue' in econ_results.columns else [],
            'grid_import': econ_results['grid_import'].values.tolist() if 'grid_import' in econ_results.columns else [],
            'grid_export': econ_results['grid_export'].values.tolist() if 'grid_export' in econ_results.columns else []
        }
        
        return jsonify({
            'success': True,
            'data': response_data,
            'summary': summary,
            'count': len(econ_results)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/data/summary', methods=['GET'])
def get_data_summary():
    """Get summary statistics of all data"""
    global df_master, battery_results, economic_results
    
    summary = {}
    
    if df_master is not None and len(df_master) > 0:
        apt_cols = [col for col in df_master.columns if col.startswith('apartment_')]
        summary['master_dataset'] = {
            'count': len(df_master),
            'date_range': {
                'start': df_master.index[0].isoformat(),
                'end': df_master.index[-1].isoformat()
            },
            'n_apartments': len(apt_cols),
            'total_load_mean': float(df_master[apt_cols].sum(axis=1).mean()) if apt_cols else None,
            'pv_mean': float(df_master['pv_1kw'].mean()) if 'pv_1kw' in df_master.columns else None
        }
    
    if battery_results is not None and len(battery_results) > 0:
        summary['battery'] = {
            'count': len(battery_results),
            'avg_soc': float(battery_results['battery_soc'].mean()),
            'total_charge': float(battery_results['battery_charge_total'].sum()),
            'total_discharge': float(battery_results['battery_discharge_total'].sum()),
            'total_grid_import': float(battery_results['grid_import'].sum()) if 'grid_import' in battery_results.columns else 0,
            'total_grid_export': float(battery_results['grid_export'].sum()) if 'grid_export' in battery_results.columns else 0
        }
    
    if economic_results is not None and len(economic_results) > 0:
        summary['economic'] = {
            'total_import_cost': float(economic_results['grid_import_cost'].sum()) if 'grid_import_cost' in economic_results.columns else 0,
            'total_export_revenue': float(economic_results['grid_export_revenue'].sum()) if 'grid_export_revenue' in economic_results.columns else 0,
            'net_cost': float(summary.get('economic', {}).get('total_import_cost', 0) - summary.get('economic', {}).get('total_export_revenue', 0))
        }
    
    return jsonify({
        'success': True,
        'summary': summary
    })


if __name__ == '__main__':
    # Initialize with sample data
    print("Initializing with sample data...")
    df = create_sample_torino_data('2023-01-01', '2023-12-31')
    print(f"Loaded {len(df)} records")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

