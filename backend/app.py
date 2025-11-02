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
except ImportError as e:
    print(f"Warning: Could not import LSTM modules: {e}")
    print("Make sure lstm_energy_forecast.py and data_loader.py are in the parent directory")

app = Flask(__name__)
CORS(app)

# Global variables
model = None
df = None
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
    global df
    
    if df is None or len(df) == 0:
        return jsonify({'success': False, 'error': 'No data loaded'}), 400
    
    try:
        # Get date range from query params
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        data_df = df.copy()
        
        if start_date:
            data_df = data_df[data_df.index >= start_date]
        if end_date:
            data_df = data_df[data_df.index <= end_date]
        
        # Limit to last 1000 points for performance, or specific range
        limit = int(request.args.get('limit', 1000))
        if len(data_df) > limit:
            data_df = data_df.iloc[-limit:]
        
        # Convert to JSON format
        data = {
            'timestamps': [ts.isoformat() for ts in data_df.index],
            'load': df['load'].values.tolist(),
            'pv': df['pv'].values.tolist()
        }
        
        return jsonify({
            'success': True,
            'data': data,
            'count': len(data_df)
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


if __name__ == '__main__':
    # Initialize with sample data
    print("Initializing with sample data...")
    df = create_sample_torino_data('2023-01-01', '2023-12-31')
    print(f"Loaded {len(df)} records")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

