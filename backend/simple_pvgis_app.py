#!/usr/bin/env python3
"""
Simple PVGIS Application
A working version that displays real PVGIS data
"""

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import pandas as pd
import os
import json

app = Flask(__name__)
CORS(app)

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "project", "data")

class SimplePVGISManager:
    """Simple PVGIS data manager"""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.pv_24h = None
        self.pv_8760 = None
        self.load_data()
    
    def load_data(self):
        """Load real PV data"""
        try:
            # Load PV 24h data
            pv_24h_file = os.path.join(self.data_dir, "pv_24h.csv")
            if os.path.exists(pv_24h_file):
                self.pv_24h = pd.read_csv(pv_24h_file)
                print("✅ Real PV 24h data loaded")
            
            # Load PV 8760h data
            pv_8760_file = os.path.join(self.data_dir, "pv_8760.csv")
            if os.path.exists(pv_8760_file):
                self.pv_8760 = pd.read_csv(pv_8760_file)
                print("✅ Real PV 8760h data loaded")
                
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def get_pv_summary(self):
        """Get PV performance summary"""
        if self.pv_24h is not None:
            daily_generation = self.pv_24h['pv_generation_kw'].sum()
            peak_power = self.pv_24h['pv_generation_kw'].max()
            peak_hour = self.pv_24h.loc[self.pv_24h['pv_generation_kw'].idxmax(), 'hour']
            
            return {
                "daily_generation_kwh": round(daily_generation, 2),
                "peak_power_kw": round(peak_power, 3),
                "peak_hour": int(peak_hour),
                "data_source": "Real PVGIS Data",
                "location": "Turin, Italy",
                "coordinates": "45.0703°N, 7.6869°E",
                "years": "2005-2023"
            }
        return None
    
    def get_hourly_data(self, period='24h'):
        """Get hourly PV data"""
        if period == '24h' and self.pv_24h is not None:
            return {
                "hours": self.pv_24h['hour'].tolist(),
                "values": self.pv_24h['pv_generation_kw'].tolist(),
                "label": "Real PVGIS Generation (kW)"
            }
        elif period == '8760h' and self.pv_8760 is not None:
            return {
                "hours": self.pv_8760['hour'].tolist(),
                "values": self.pv_8760['pv_generation_kw'].tolist(),
                "label": "Real PVGIS Generation (kW)"
            }
        return None
    
    def get_monthly_data(self):
        """Get monthly aggregated data"""
        if self.pv_8760 is not None:
            # Create monthly aggregation
            monthly_data = []
            for month in range(1, 13):
                # Get data for this month (approximately 30 days each)
                start_hour = (month - 1) * 30 * 24
                end_hour = month * 30 * 24
                month_data = self.pv_8760[
                    (self.pv_8760['hour'] >= start_hour) & 
                    (self.pv_8760['hour'] < end_hour)
                ]
                
                if not month_data.empty:
                    monthly_generation = month_data['pv_generation_kw'].sum()
                    monthly_data.append({
                        "month": month,
                        "generation_kwh": round(monthly_generation, 2)
                    })
            
            return monthly_data
        return None

# Initialize data manager
pvgis_manager = SimplePVGISManager()

@app.route('/')
def index():
    """Serve the PVGIS dashboard"""
    return render_template('simple_pvgis_dashboard.html')

@app.route('/api/pvgis/summary')
def get_pvgis_summary():
    """Get PVGIS performance summary"""
    summary = pvgis_manager.get_pv_summary()
    if summary:
        return jsonify(summary)
    else:
        return jsonify({"error": "PV data not available"}), 404

@app.route('/api/pvgis/data')
def get_pvgis_data():
    """Get PVGIS hourly data"""
    period = request.args.get('period', '24h')
    data = pvgis_manager.get_hourly_data(period)
    if data:
        return jsonify(data)
    else:
        return jsonify({"error": "PV data not available"}), 404

@app.route('/api/pvgis/monthly')
def get_pvgis_monthly():
    """Get PVGIS monthly data"""
    data = pvgis_manager.get_monthly_data()
    if data:
        return jsonify(data)
    else:
        return jsonify({"error": "Monthly data not available"}), 404

@app.route('/api/pvgis/status')
def get_pvgis_status():
    """Get PVGIS connection status"""
    status = {
        'data_loaded': pvgis_manager.pv_24h is not None,
        'data_source': 'Real PVGIS Data',
        'location': 'Turin, Italy',
        'coordinates': '45.0703°N, 7.6869°E',
        'years': '2005-2023',
        'database': 'PVGIS-SARAH3'
    }
    
    if pvgis_manager.pv_24h is not None:
        status['total_records_24h'] = len(pvgis_manager.pv_24h)
        status['daily_generation_kwh'] = pvgis_manager.pv_24h['pv_generation_kw'].sum()
    
    if pvgis_manager.pv_8760 is not None:
        status['total_records_8760h'] = len(pvgis_manager.pv_8760)
    
    return jsonify(status)

if __name__ == '__main__':
    print("=" * 60)
    print("SIMPLE PVGIS APPLICATION")
    print("=" * 60)
    print("🌞 Real PVGIS Data Dashboard")
    print("📍 Location: Turin, Italy")
    print("📅 Data: 2005-2023 (19 years)")
    print("🔗 Dashboard: http://localhost:5001")
    print("=" * 60)
    
    app.run(debug=True, port=5001)

