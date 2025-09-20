#!/usr/bin/env python3
"""
Energy Management System - Backend API
Flask-based REST API for building energy data management
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime, timedelta
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "project", "data")
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'yaml', 'json'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class EnergyDataManager:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.load_data()
    
    def load_data(self):
        """Load all energy data files"""
        try:
            # Load building load data
            self.load_24h = pd.read_csv(f"{self.data_dir}/load_24h.csv")
            self.load_8760 = pd.read_csv(f"{self.data_dir}/load_8760.csv")
            
            # Load PV generation data
            self.pv_24h = pd.read_csv(f"{self.data_dir}/pv_24h.csv")
            self.pv_8760 = pd.read_csv(f"{self.data_dir}/pv_8760.csv")
            
            # Load TOU pricing data
            self.tou_24h = pd.read_csv(f"{self.data_dir}/tou_24h.csv")
            
            # Load battery specifications
            with open(f"{self.data_dir}/battery.yaml", 'r') as f:
                self.battery_specs = yaml.safe_load(f)
            
            # Load family data
            self.family_data = {}
            for family in ['familyA', 'familyB', 'familyC', 'familyD']:
                self.family_data[family] = pd.read_csv(f"{self.data_dir}/{family}.csv")
            
            print("✓ All energy data loaded successfully")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.load_data = None
    
    def get_system_summary(self):
        """Get system overview summary"""
        if self.load_data is None:
            return {"error": "Data not loaded"}
        
        # Calculate key metrics
        daily_load = self.load_24h['load_kw'].sum()
        daily_pv = self.pv_24h['pv_generation_kw'].sum()
        peak_load = self.load_24h['load_kw'].max()
        peak_pv = self.pv_24h['pv_generation_kw'].max()
        
        # Calculate energy balance
        energy_balance = daily_pv - daily_load
        self_sufficiency = min(100, (daily_pv / daily_load) * 100) if daily_load > 0 else 0
        
        return {
            "building": {
                "units": 20,
                "daily_consumption_kwh": round(daily_load, 1),
                "peak_demand_kw": round(peak_load, 1),
                "annual_consumption_kwh": round(self.load_8760['load_kw'].sum(), 0)
            },
            "pv_system": {
                "capacity_kwp": 120,
                "daily_generation_kwh": round(daily_pv, 1),
                "peak_generation_kw": round(peak_pv, 1),
                "annual_generation_kwh": round(self.pv_8760['pv_generation_kw'].sum(), 0)
            },
            "battery": {
                "capacity_kwh": self.battery_specs['Ebat_kWh'],
                "max_power_kw": self.battery_specs['Pdis_max_kW'],
                "charge_power_kw": self.battery_specs['Pch_max_kW'],
                "efficiency": self.battery_specs['eta_dis'],
                "soc_min": self.battery_specs['SOCmin'],
                "soc_max": self.battery_specs['SOCmax']
            },
            "energy_balance": {
                "daily_balance_kwh": round(energy_balance, 1),
                "self_sufficiency_percent": round(self_sufficiency, 1),
                "grid_dependency_percent": round(100 - self_sufficiency, 1)
            }
        }
    
    def get_hourly_data(self, data_type="load", period="24h"):
        """Get hourly data for charts"""
        if data_type == "load":
            data = self.load_24h if period == "24h" else self.load_8760
            return {
                "hours": data['hour'].tolist(),
                "values": data['load_kw'].tolist(),
                "label": "Building Load (kW)"
            }
        elif data_type == "pv":
            data = self.pv_24h if period == "24h" else self.pv_8760
            return {
                "hours": data['hour'].tolist(),
                "values": data['pv_generation_kw'].tolist(),
                "label": "PV Generation (kW)"
            }
        elif data_type == "tou":
            return {
                "hours": self.tou_24h['hour'].tolist(),
                "values": self.tou_24h['price_buy'].tolist(),
                "label": "TOU Buy Price (€/kWh)",
                "sell_prices": self.tou_24h['price_sell'].tolist()
            }
    
    def get_family_breakdown(self):
        """Get family type consumption breakdown"""
        family_summary = {}
        for family, data in self.family_data.items():
            annual_consumption = data['consumption_kw'].sum()
            daily_average = annual_consumption / 365
            peak_power = data['consumption_kw'].max()
            
            family_summary[family] = {
                "annual_kwh": round(annual_consumption, 0),
                "daily_avg_kwh": round(daily_average, 1),
                "peak_kw": round(peak_power, 1)
            }
        
        return family_summary
    
    def calculate_optimization(self, battery_soc=0.5):
        """Calculate energy optimization scenario using Italian ARERA TOU structure"""
        # Load data
        load_data = self.load_24h['load_kw'].values
        pv_data = self.pv_24h['pv_generation_kw'].values
        tou_buy_data = self.tou_24h['price_buy'].values
        tou_sell_data = self.tou_24h['price_sell'].values
        
        battery_capacity = self.battery_specs['Ebat_kWh']
        battery_power = self.battery_specs['Pdis_max_kW']
        battery_efficiency = self.battery_specs['eta_dis']
        charge_efficiency = self.battery_specs['eta_ch']
        soc_min = self.battery_specs['SOCmin']
        soc_max = self.battery_specs['SOCmax']
        
        # Calculate net load (load - PV)
        net_load = load_data - pv_data
        
        # Italian ARERA TOU optimization logic
        battery_charge = []
        battery_discharge = []
        grid_import = []
        grid_export = []
        current_soc = battery_soc
        
        # Define Italian tariff bands
        F3_valley = 0.24  # €/kWh
        F2_flat = 0.34    # €/kWh  
        F1_peak = 0.48    # €/kWh
        
        for hour in range(24):
            net = net_load[hour]
            buy_price = tou_buy_data[hour]
            sell_price = tou_sell_data[hour]
            
            if net > 0:  # Need energy
                # Strategy: Use battery during peak hours (F1), grid during valley (F3)
                if buy_price >= F1_peak:  # Peak hours - prioritize battery
                    # Check SOC bounds
                    available_energy = max(0, (current_soc - soc_min) * battery_capacity)
                    max_discharge = min(battery_power, available_energy)
                    if max_discharge > 0:
                        discharge = min(net, max_discharge)
                        battery_discharge.append(discharge)
                        current_soc -= discharge / battery_capacity
                        remaining = net - discharge
                    else:
                        battery_discharge.append(0)
                        remaining = net
                else:  # Valley/flat hours - use grid
                    battery_discharge.append(0)
                    remaining = net
                
                grid_import.append(remaining)
                grid_export.append(0)
                
            else:  # Excess energy
                # Strategy: Charge battery during valley hours, export during peak
                if buy_price <= F3_valley:  # Valley hours - charge battery
                    # Check SOC bounds
                    available_capacity = max(0, (soc_max - current_soc) * battery_capacity)
                    max_charge = min(battery_power, available_capacity)
                    if max_charge > 0:
                        charge = min(-net, max_charge)
                        battery_charge.append(charge)
                        current_soc += charge / battery_capacity
                        remaining = -net - charge
                    else:
                        battery_charge.append(0)
                        remaining = -net
                else:  # Peak/flat hours - export to grid
                    battery_charge.append(0)
                    remaining = -net
                
                grid_import.append(0)
                grid_export.append(remaining)
        
        return {
            "hours": list(range(24)),
            "net_load": net_load.tolist(),
            "battery_charge": battery_charge,
            "battery_discharge": battery_discharge,
            "grid_import": grid_import,
            "grid_export": grid_export,
            "final_soc": current_soc
        }

# Initialize data manager
data_manager = EnergyDataManager()

# API Routes
@app.route('/')
def index():
    """Serve the main dashboard"""
    return send_from_directory('../frontend', 'index.html')

@app.route('/api/summary')
def get_summary():
    """Get system summary"""
    return jsonify(data_manager.get_system_summary())

@app.route('/api/data/<data_type>')
def get_data(data_type):
    """Get hourly data for charts"""
    period = request.args.get('period', '24h')
    return jsonify(data_manager.get_hourly_data(data_type, period))

@app.route('/api/families')
def get_families():
    """Get family breakdown data"""
    return jsonify(data_manager.get_family_breakdown())

@app.route('/api/battery')
def get_battery():
    """Get battery specifications"""
    return jsonify(data_manager.battery_specs)

@app.route('/api/optimization')
def get_optimization():
    """Get energy optimization scenario"""
    battery_soc = float(request.args.get('soc', 0.5))
    return jsonify(data_manager.calculate_optimization(battery_soc))

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload new data files"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({"message": f"File {filename} uploaded successfully"})
    
    return jsonify({"error": "Invalid file type"}), 400

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": data_manager.load_data is not None
    })

if __name__ == '__main__':
    print("=" * 60)
    print("ENERGY MANAGEMENT SYSTEM - BACKEND API")
    print("=" * 60)
    print("Starting Flask server...")
    print("API Endpoints:")
    print("  GET  /api/summary      - System overview")
    print("  GET  /api/data/<type>  - Hourly data (load, pv, tou)")
    print("  GET  /api/families     - Family breakdown")
    print("  GET  /api/battery      - Battery specifications")
    print("  GET  /api/optimization - Energy optimization")
    print("  GET  /api/health       - Health check")
    print()
    print("Frontend: http://localhost:5000")
    print("API Base: http://localhost:5000/api")
    print()
    
    port = int(os.environ.get('FLASK_PORT', 5002))
    app.run(debug=True, host='0.0.0.0', port=port)
