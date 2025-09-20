#!/usr/bin/env python3
"""
PVGIS Full-Stack Application
Backend API for PVGIS data visualization and analysis
"""

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import yaml
from pvgis_extractor import PVDataExtractor

app = Flask(__name__)
CORS(app)

class PVGISDataManager:
    """
    Manages PVGIS data and provides analysis functions
    """
    
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "project", "data")
        self.pv_data = None
        self.tou_data = None
        self.load_data = None
        self.battery_specs = None
        
        # Initialize PVGIS data extractor
        self.pv_extractor = PVDataExtractor(use_pvgis=True)
        
        self.load_data()
    
    def load_data(self):
        """Load all data files"""
        try:
            # Load PV data from existing real CSV files
            print("Loading real PV data from CSV files...")
            
            pv_24h_file = os.path.join(self.data_dir, "pv_24h.csv")
            pv_8760_file = os.path.join(self.data_dir, "pv_8760.csv")
            
            if os.path.exists(pv_24h_file):
                self.pv_24h = pd.read_csv(pv_24h_file)
                print("✅ Real PV 24h data loaded successfully")
            else:
                print("⚠️ PV 24h data not found")
                
            if os.path.exists(pv_8760_file):
                self.pv_8760 = pd.read_csv(pv_8760_file)
                print("✅ Real PV 8760h data loaded successfully")
            else:
                print("⚠️ PV 8760h data not found")
            
            # Load TOU data
            tou_24h_file = os.path.join(self.data_dir, "tou_24h.csv")
            if os.path.exists(tou_24h_file):
                self.tou_24h = pd.read_csv(tou_24h_file)
            
            # Load load data
            load_24h_file = os.path.join(self.data_dir, "load_24h.csv")
            if os.path.exists(load_24h_file):
                self.load_24h = pd.read_csv(load_24h_file)
            
            # Load battery specs
            battery_file = os.path.join(self.data_dir, "battery.yaml")
            if os.path.exists(battery_file):
                with open(battery_file, 'r') as f:
                    self.battery_specs = yaml.safe_load(f)
            
            print("✅ All PVGIS data loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def get_pv_performance_summary(self):
        """Get PV performance summary similar to PVGIS results"""
        if not hasattr(self, 'pv_8760'):
            return None
        
        # Calculate performance metrics
        total_annual = self.pv_8760['pv_generation_kw'].sum()
        daily_average = total_annual / 365
        peak_generation = self.pv_8760['pv_generation_kw'].max()
        capacity_factor = total_annual / (120 * 8760)  # 120 kWp system
        
        # Monthly analysis
        monthly_data = []
        for month in range(1, 13):
            month_start = (month - 1) * 30 * 24  # Approximate
            month_end = month_start + 30 * 24
            month_data = self.pv_8760.iloc[month_start:month_end]
            monthly_generation = month_data['pv_generation_kw'].sum()
            monthly_data.append({
                'month': month,
                'generation_kwh': monthly_generation,
                'daily_average_kwh': monthly_generation / 30
            })
        
        return {
            'yearly_energy_production_kwh': total_annual,
            'daily_average_kwh': daily_average,
            'peak_generation_kw': peak_generation,
            'capacity_factor_percent': capacity_factor * 100,
            'monthly_data': monthly_data,
            'system_specs': {
                'installed_power_kwp': 120,
                'system_loss_percent': 14,
                'tilt_angle_deg': 30,
                'azimuth_angle_deg': 180,
                'location': 'Turin, Italy',
                'coordinates': '45.0703°N, 7.6869°E'
            }
        }
    
    def get_hourly_pv_data(self, period='24h'):
        """Get hourly PV data for visualization"""
        if period == '24h' and hasattr(self, 'pv_24h') and self.pv_24h is not None:
            return {
                'hours': self.pv_24h['hour'].tolist(),
                'values': self.pv_24h['pv_generation_kw'].tolist(),
                'label': 'PV Generation (kW) - Real PVGIS Data'
            }
        elif period == '8760h' and hasattr(self, 'pv_8760') and self.pv_8760 is not None:
            return {
                'hours': self.pv_8760['hour'].tolist(),
                'values': self.pv_8760['pv_generation_kw'].tolist(),
                'label': 'PV Generation (kW) - Real PVGIS Data'
            }
        return None
    
    def get_monthly_pv_data(self):
        """Get monthly PV data for visualization"""
        if not hasattr(self, 'pv_8760'):
            return None
        
        monthly_data = []
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month in range(12):
            month_start = month * 30 * 24  # Approximate
            month_end = month_start + 30 * 24
            month_data = self.pv_8760.iloc[month_start:month_end]
            monthly_generation = month_data['pv_generation_kw'].sum()
            monthly_data.append({
                'month': month_names[month],
                'generation_kwh': monthly_generation
            })
        
        return monthly_data
    
    def get_daily_pv_profile(self, day_type='summer'):
        """Get daily PV profile for different seasons"""
        if not hasattr(self, 'pv_8760'):
            return None
        
        # Select representative days
        if day_type == 'summer':
            day_start = 179 * 24  # Day 180 (summer)
        elif day_type == 'winter':
            day_start = 0 * 24    # Day 1 (winter)
        elif day_type == 'spring':
            day_start = 90 * 24   # Day 91 (spring)
        else:  # autumn
            day_start = 270 * 24  # Day 271 (autumn)
        
        day_end = day_start + 24
        day_data = self.pv_8760.iloc[day_start:day_end]
        
        return {
            'hours': list(range(24)),
            'values': day_data['pv_generation_kw'].tolist(),
            'label': f'PV Generation - {day_type.title()} Day (kW)'
        }
    
    def get_energy_balance_analysis(self):
        """Get energy balance analysis"""
        if not all(hasattr(self, attr) for attr in ['pv_24h', 'load_24h']):
            return None
        
        # Calculate energy balance
        pv_generation = self.pv_24h['pv_generation_kw'].sum()
        load_consumption = self.load_24h['load_kw'].sum()
        energy_balance = pv_generation - load_consumption
        
        # Calculate self-sufficiency
        self_sufficiency = min(100, (pv_generation / load_consumption) * 100) if load_consumption > 0 else 0
        
        return {
            'pv_generation_kwh': pv_generation,
            'load_consumption_kwh': load_consumption,
            'energy_balance_kwh': energy_balance,
            'self_sufficiency_percent': self_sufficiency,
            'grid_dependency_percent': 100 - self_sufficiency
        }
    
    def get_optimization_scenario(self, battery_soc=0.5):
        """Get energy optimization scenario"""
        if not all(hasattr(self, attr) for attr in ['pv_24h', 'load_24h', 'tou_24h', 'battery_specs']):
            return None
        
        # Load data
        pv_data = self.pv_24h['pv_generation_kw'].values
        load_data = self.load_24h['load_kw'].values
        tou_data = self.tou_24h['price_buy'].values
        
        # Battery parameters
        battery_capacity = self.battery_specs['Ebat_kWh']
        battery_power = self.battery_specs['Pdis_max_kW']
        battery_efficiency = self.battery_specs['eta_dis']
        
        # Calculate net load
        net_load = load_data - pv_data
        
        # Simple optimization logic
        battery_charge = []
        battery_discharge = []
        grid_import = []
        grid_export = []
        current_soc = battery_soc
        
        for hour in range(24):
            net = net_load[hour]
            price = tou_data[hour]
            
            if net > 0:  # Need energy
                # Use battery during high price hours
                if price > 0.4:  # High price threshold
                    max_discharge = min(battery_power, current_soc * battery_capacity)
                    if max_discharge > 0:
                        discharge = min(net, max_discharge)
                        battery_discharge.append(discharge)
                        current_soc -= discharge / battery_capacity
                        remaining = net - discharge
                    else:
                        battery_discharge.append(0)
                        remaining = net
                else:
                    battery_discharge.append(0)
                    remaining = net
                
                grid_import.append(remaining)
                grid_export.append(0)
                
            else:  # Excess energy
                # Charge battery during low price hours
                if price < 0.3:  # Low price threshold
                    max_charge = min(battery_power, (1 - current_soc) * battery_capacity)
                    if max_charge > 0:
                        charge = min(-net, max_charge)
                        battery_charge.append(charge)
                        current_soc += charge / battery_capacity
                        remaining = -net - charge
                    else:
                        battery_charge.append(0)
                        remaining = -net
                else:
                    battery_charge.append(0)
                    remaining = -net
                
                grid_import.append(0)
                grid_export.append(remaining)
        
        return {
            'hours': list(range(24)),
            'net_load': net_load.tolist(),
            'battery_charge': battery_charge,
            'battery_discharge': battery_discharge,
            'grid_import': grid_import,
            'grid_export': grid_export,
            'final_soc': current_soc
        }

# Initialize data manager
pvgis_manager = PVGISDataManager()

@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('pvgis_dashboard.html')

@app.route('/api/pvgis/summary')
def get_pvgis_summary():
    """Get PVGIS performance summary"""
    summary = pvgis_manager.get_pv_performance_summary()
    if summary:
        return jsonify(summary)
    else:
        return jsonify({'error': 'PV data not available'}), 404

@app.route('/api/pvgis/hourly')
def get_hourly_pv_data():
    """Get hourly PV data"""
    period = request.args.get('period', '24h')
    data = pvgis_manager.get_hourly_pv_data(period)
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'PV data not available'}), 404

@app.route('/api/pvgis/monthly')
def get_monthly_pv_data():
    """Get monthly PV data"""
    data = pvgis_manager.get_monthly_pv_data()
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'PV data not available'}), 404

@app.route('/api/pvgis/daily')
def get_daily_pv_profile():
    """Get daily PV profile for different seasons"""
    day_type = request.args.get('type', 'summer')
    data = pvgis_manager.get_daily_pv_profile(day_type)
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'PV data not available'}), 404

@app.route('/api/pvgis/energy-balance')
def get_energy_balance():
    """Get energy balance analysis"""
    data = pvgis_manager.get_energy_balance_analysis()
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'Data not available'}), 404

@app.route('/api/pvgis/optimization')
def get_optimization_scenario():
    """Get energy optimization scenario"""
    battery_soc = float(request.args.get('soc', 0.5))
    data = pvgis_manager.get_optimization_scenario(battery_soc)
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'Data not available'}), 404

@app.route('/api/pvgis/system-specs')
def get_system_specs():
    """Get system specifications"""
    if pvgis_manager.battery_specs:
        return jsonify(pvgis_manager.battery_specs)
    else:
        return jsonify({'error': 'System specs not available'}), 404

@app.route('/api/pvgis/fetch-real-data')
def fetch_real_pvgis_data():
    """Fetch real-time PVGIS data"""
    try:
        print("Fetching real-time PVGIS data...")
        
        # Create new extractor instance
        extractor = PVDataExtractor(use_pvgis=True)
        data = extractor.load_data()
        
        if data is not None:
            # Get daily profile
            daily_profile = extractor.get_daily_profile()
            total_generation = extractor.get_total_daily_generation()
            peak_generation = extractor.get_peak_generation()
            
            return jsonify({
                'success': True,
                'data': {
                    'daily_profile': daily_profile,
                    'total_daily_generation_kwh': total_generation,
                    'peak_generation': peak_generation,
                    'coordinates': f"{extractor.lat}°N, {extractor.lon}°E",
                    'location': 'Turin, Italy',
                    'data_source': 'PVGIS API v5.3',
                    'years': '2005-2023'
                }
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to fetch PVGIS data'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pvgis/status')
def get_pvgis_status():
    """Get PVGIS connection status and data source info"""
    try:
        status = {
            'pvgis_connected': pvgis_manager.pv_extractor.data is not None,
            'data_source': 'PVGIS API v5.3' if pvgis_manager.pv_extractor.data is not None else 'Local CSV',
            'location': 'Turin, Italy',
            'coordinates': f"{pvgis_manager.pv_extractor.lat}°N, {pvgis_manager.pv_extractor.lon}°E",
            'api_url': pvgis_manager.pv_extractor.pvgis_url,
            'years_available': '2005-2023',
            'database': 'PVGIS-SARAH3'
        }
        
        if pvgis_manager.pv_extractor.data is not None:
            status['data_loaded'] = True
            status['total_records'] = len(pvgis_manager.pv_extractor.data)
            status['daily_generation_kwh'] = pvgis_manager.pv_extractor.get_total_daily_generation()
        else:
            status['data_loaded'] = False
            
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("PVGIS FULL-STACK APPLICATION")
    print("=" * 60)
    print("Starting PVGIS data visualization server...")
    print("Dashboard: http://localhost:5001")
    print()
    
    app.run(host='0.0.0.0', port=5001, debug=True)
