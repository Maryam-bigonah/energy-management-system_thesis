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
    from build_master_dataset_final import build_master_dataset
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
def health_check():
    """Health check endpoint"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'message': 'Backend is running'
    })


@app.route('/api/data/load-master-csv', methods=['POST'])
def load_master_csv():
    """Load master dataset from CSV file"""
    global df_master
    
    try:
        data = request.get_json() or {}
        csv_path = data.get('csv_path', 'data/master_dataset_2024.csv')
        
        # Convert relative to absolute path
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(parent_dir, csv_path)
        
        if not os.path.exists(csv_path):
            return jsonify({
                'success': False, 
                'error': f'CSV file not found: {csv_path}'
            }), 400
        
        # Load CSV
        df_master = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        return jsonify({
            'success': True,
            'message': f'Master dataset loaded: {len(df_master)} records',
            'date_range': {
                'start': df_master.index.min().isoformat(),
                'end': df_master.index.max().isoformat()
            },
            'columns': list(df_master.columns),
            'n_apartments': len([col for col in df_master.columns if col.startswith('apartment_')])
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/data/family-consumption', methods=['GET'])
def get_family_consumption():
    """Get consumption data by family type"""
    global df_master
    
    if df_master is None or len(df_master) == 0:
        return jsonify({'success': False, 'error': 'Master dataset not loaded'}), 400
    
    try:
        # Family type mapping
        family_types = {
            'couple_working': [f'apartment_{i:02d}' for i in range(1, 6)],
            'family_one_child': [f'apartment_{i:02d}' for i in range(6, 11)],
            'one_working': [f'apartment_{i:02d}' for i in range(11, 16)],
            'retired': [f'apartment_{i:02d}' for i in range(16, 21)]
        }
        
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
        
        result = {
            'timestamps': [ts.isoformat() for ts in data_df.index],
            'families': {},
            'pv': data_df['pv_1kw'].values.tolist() if 'pv_1kw' in data_df.columns else [],
            'total_load': 0,
            'pv_total': 0
        }
        
        # Calculate consumption for each family type
        for family_type, apartments in family_types.items():
            family_cols = [col for col in apartments if col in data_df.columns]
            if family_cols:
                family_load = data_df[family_cols].sum(axis=1).values.tolist()
                result['families'][family_type] = {
                    'load': family_load,
                    'total_kwh': sum(family_load),
                    'avg_kw': sum(family_load) / len(family_load) if family_load else 0,
                    'apartments': family_cols
                }
        
        # Calculate totals
        apt_cols = [col for col in data_df.columns if col.startswith('apartment_')]
        if apt_cols:
            result['total_load'] = float(data_df[apt_cols].sum(axis=1).sum())
        if 'pv_1kw' in data_df.columns:
            result['pv_total'] = float(data_df['pv_1kw'].sum())
        
        return jsonify({
            'success': True,
            'data': result,
            'count': len(data_df)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/data/storage-energy', methods=['GET'])
def get_storage_energy():
    """Get energy storage data from PV and battery"""
    global df_master, battery_results
    
    if df_master is None or len(df_master) == 0:
        return jsonify({'success': False, 'error': 'Master dataset not loaded'}), 400
    
    try:
        limit = int(request.args.get('limit', 1000))
        
        result = {
            'pv_generation': {},
            'battery_storage': {},
            'grid_export': {}
        }
        
        # PV data
        apt_cols = [col for col in df_master.columns if col.startswith('apartment_')]
        total_load = df_master[apt_cols].sum(axis=1) if apt_cols else pd.Series([0] * len(df_master), index=df_master.index)
        pv_gen = df_master['pv_1kw'] if 'pv_1kw' in df_master.columns else pd.Series([0] * len(df_master), index=df_master.index)
        
        # Net energy (PV - Load)
        net_energy = pv_gen - total_load
        
        # Excess PV that could be stored
        excess_pv = net_energy.clip(lower=0)
        
        data_df = df_master.iloc[-limit:] if len(df_master) > limit else df_master
        
        result['pv_generation'] = {
            'timestamps': [ts.isoformat() for ts in data_df.index],
            'pv_kw': pv_gen.loc[data_df.index].values.tolist(),
            'load_kw': total_load.loc[data_df.index].values.tolist(),
            'net_kw': net_energy.loc[data_df.index].values.tolist(),
            'excess_kw': excess_pv.loc[data_df.index].values.tolist(),
            'total_pv_kwh': float(pv_gen.sum()),
            'total_excess_kwh': float(excess_pv.sum()),
            'self_consumption_kwh': float((pv_gen - excess_pv).sum())
        }
        
        # Battery data if available
        if battery_results is not None and len(battery_results) > 0:
            battery_df = battery_results.iloc[-limit:] if len(battery_results) > limit else battery_results
            
            result['battery_storage'] = {
                'timestamps': [ts.isoformat() for ts in battery_df.index],
                'soc': battery_df['battery_soc'].values.tolist(),
                'charge_kw': battery_df['battery_charge_total'].values.tolist(),
                'discharge_kw': battery_df['battery_discharge_total'].values.tolist(),
                'total_charge_kwh': float(battery_df['battery_charge_total'].sum()),
                'total_discharge_kwh': float(battery_df['battery_discharge_total'].sum()),
                'avg_soc': float(battery_df['battery_soc'].mean())
            }
            
            if 'grid_export' in battery_df.columns:
                result['grid_export'] = {
                    'timestamps': [ts.isoformat() for ts in battery_df.index],
                    'export_kw': battery_df['grid_export'].values.tolist(),
                    'total_export_kwh': float(battery_df['grid_export'].sum())
                }
        
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/data/all-data-summary', methods=['GET'])
def get_all_data_summary():
    """Get complete summary of all data: families, PV storage, battery, prices"""
    global df_master, battery_results, economic_results
    
    try:
        summary = {}
        
        # Family consumption
        if df_master is not None and len(df_master) > 0:
            family_types = {
                'couple_working': [f'apartment_{i:02d}' for i in range(1, 6)],
                'family_one_child': [f'apartment_{i:02d}' for i in range(6, 11)],
                'one_working': [f'apartment_{i:02d}' for i in range(11, 16)],
                'retired': [f'apartment_{i:02d}' for i in range(16, 21)]
            }
            
            apt_cols = [col for col in df_master.columns if col.startswith('apartment_')]
            total_load = df_master[apt_cols].sum(axis=1) if apt_cols else pd.Series([0] * len(df_master))
            pv_gen = df_master['pv_1kw'] if 'pv_1kw' in df_master.columns else pd.Series([0] * len(df_master))
            net_energy = pv_gen - total_load
            excess_pv = net_energy.clip(lower=0)
            
            summary['families'] = {}
            for family_type, apartments in family_types.items():
                family_cols = [col for col in apartments if col in df_master.columns]
                if family_cols:
                    family_load = df_master[family_cols].sum(axis=1)
                    summary['families'][family_type] = {
                        'total_consumption_kwh': float(family_load.sum()),
                        'avg_consumption_kw': float(family_load.mean()),
                        'max_consumption_kw': float(family_load.max()),
                        'n_apartments': len(family_cols)
                    }
            
            summary['pv_storage'] = {
                'total_pv_generation_kwh': float(pv_gen.sum()),
                'total_excess_pv_kwh': float(excess_pv.sum()),
                'self_consumption_kwh': float((pv_gen - excess_pv).sum()),
                'self_consumption_rate': float((pv_gen - excess_pv).sum() / pv_gen.sum() * 100) if pv_gen.sum() > 0 else 0
            }
        
        # Battery
        if battery_results is not None and len(battery_results) > 0:
            summary['battery'] = {
                'total_charge_kwh': float(battery_results['battery_charge_total'].sum()),
                'total_discharge_kwh': float(battery_results['battery_discharge_total'].sum()),
                'avg_soc': float(battery_results['battery_soc'].mean()),
                'energy_stored_kwh': float(battery_results['battery_charge_total'].sum()) * 0.9  # With efficiency
            }
        
        # Economic
        if economic_results is not None and len(economic_results) > 0:
            summary['prices'] = {
                'total_import_cost_eur': float(economic_results['grid_import_cost'].sum()) if 'grid_import_cost' in economic_results.columns else 0,
                'total_export_revenue_eur': float(economic_results['grid_export_revenue'].sum()) if 'grid_export_revenue' in economic_results.columns else 0,
                'net_cost_eur': float(summary.get('prices', {}).get('total_import_cost_eur', 0) - summary.get('prices', {}).get('total_export_revenue_eur', 0))
            }
        
        return jsonify({
            'success': True,
            'summary': summary
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/data/database-info', methods=['GET'])
def get_database_info():
    """Get complete information about all available data: what's loaded, what files exist"""
    global df_master, battery_results, economic_results, df
    
    try:
        info = {
            'data_loaded': {},
            'data_files': [],
            'loaded_data_stats': {}
        }
        
        # Ensure DATA_DIR exists
        if not os.path.exists(DATA_DIR):
            print(f"Warning: DATA_DIR does not exist: {DATA_DIR}")
            os.makedirs(DATA_DIR, exist_ok=True)
        
        # Check what's loaded in memory
        info['data_loaded'] = {
            'master_dataset': df_master is not None and len(df_master) > 0,
            'battery_results': battery_results is not None and len(battery_results) > 0,
            'economic_results': economic_results is not None and len(economic_results) > 0,
            'historical_data': df is not None and len(df) > 0
        }
        
        # Get stats for loaded data
        if df_master is not None and len(df_master) > 0:
            apt_cols = [col for col in df_master.columns if col.startswith('apartment_')]
            info['loaded_data_stats']['master_dataset'] = {
                'rows': len(df_master),
                'columns': list(df_master.columns),
                'date_range': {
                    'start': str(df_master.index.min()),
                    'end': str(df_master.index.max())
                },
                'n_apartments': len(apt_cols),
                'total_load_range': {
                    'min': float(df_master[apt_cols].sum(axis=1).min()) if apt_cols else 0,
                    'max': float(df_master[apt_cols].sum(axis=1).max()) if apt_cols else 0,
                    'mean': float(df_master[apt_cols].sum(axis=1).mean()) if apt_cols else 0
                },
                'pv_range': {
                    'min': float(df_master['pv_1kw'].min()) if 'pv_1kw' in df_master.columns else 0,
                    'max': float(df_master['pv_1kw'].max()) if 'pv_1kw' in df_master.columns else 0,
                    'mean': float(df_master['pv_1kw'].mean()) if 'pv_1kw' in df_master.columns else 0
                }
            }
        
        if battery_results is not None and len(battery_results) > 0:
            info['loaded_data_stats']['battery_results'] = {
                'rows': len(battery_results),
                'columns': list(battery_results.columns),
                'date_range': {
                    'start': str(battery_results.index.min()),
                    'end': str(battery_results.index.max())
                },
                'total_charge_kwh': float(battery_results['battery_charge_total'].sum()) if 'battery_charge_total' in battery_results.columns else 0,
                'total_discharge_kwh': float(battery_results['battery_discharge_total'].sum()) if 'battery_discharge_total' in battery_results.columns else 0,
                'avg_soc': float(battery_results['battery_soc'].mean()) if 'battery_soc' in battery_results.columns else 0
            }
        
        if economic_results is not None and len(economic_results) > 0:
            info['loaded_data_stats']['economic_results'] = {
                'rows': len(economic_results),
                'columns': list(economic_results.columns),
                'date_range': {
                    'start': str(economic_results.index.min()),
                    'end': str(economic_results.index.max())
                },
                'total_import_cost_eur': float(economic_results['grid_import_cost'].sum()) if 'grid_import_cost' in economic_results.columns else 0,
                'total_export_revenue_eur': float(economic_results['grid_export_revenue'].sum()) if 'grid_export_revenue' in economic_results.columns else 0
            }
        
        # Check what files exist in data directory
        if os.path.exists(DATA_DIR):
            for file in os.listdir(DATA_DIR):
                file_path = os.path.join(DATA_DIR, file)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    file_info = {
                        'name': file,
                        'size_bytes': file_size,
                        'size_mb': round(file_size / (1024 * 1024), 2),
                        'path': file_path
                    }
                    
                    # If it's a CSV, try to get basic info
                    if file.endswith('.csv'):
                        try:
                            sample_df = pd.read_csv(file_path, nrows=5)
                            file_info['type'] = 'CSV'
                            file_info['columns'] = list(sample_df.columns)
                            file_info['sample_rows'] = 5
                        except:
                            file_info['type'] = 'CSV (unreadable)'
                    else:
                        file_info['type'] = 'Other'
                    
                    info['data_files'].append(file_info)
        
        return jsonify({
            'success': True,
            'info': info
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_msg = traceback.format_exc()
        print(f"Error in get_database_info: {error_msg}")
        print(f"Traceback: {traceback_msg}")
        return jsonify({
            'success': False, 
            'error': error_msg,
            'traceback': traceback_msg if app.debug else None
        }), 400

# ... existing code continues ...