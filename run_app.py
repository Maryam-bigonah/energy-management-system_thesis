#!/usr/bin/env python3
"""
Energy Management System - Application Launcher
Starts the full-stack application with backend and frontend
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import flask
        import pandas
        import numpy
        import yaml
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def check_data_files():
    """Check if all required data files exist"""
    data_dir = Path("project/data")
    required_files = [
        "load_24h.csv",
        "load_8760.csv", 
        "pv_24h.csv",
        "pv_8760.csv",
        "tou_24h.csv",
        "battery.yaml",
        "familyA.csv",
        "familyB.csv", 
        "familyC.csv",
        "familyD.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not (data_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing data files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run the data generation scripts first:")
        print("  python3 create_24h_inputs.py")
        print("  python3 generate_yearly_profiles.py")
        print("  python3 aggregate_building_load.py")
        print("  python3 generate_pv_realistic.py")
        return False
    else:
        print("✓ All required data files found")
        return True

def start_backend():
    """Start the Flask backend server"""
    print("Starting Flask backend server...")
    
    # Change to backend directory
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found")
        return None
    
    # Start Flask app
    try:
        process = subprocess.Popen([
            sys.executable, "app.py"
        ], cwd=backend_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        if process.poll() is None:
            print("✓ Backend server started successfully")
            print("  API: http://localhost:5000/api")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start backend server:")
            print(f"  Error: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def open_frontend():
    """Open the frontend in browser"""
    print("Opening frontend dashboard...")
    
    frontend_file = Path("frontend/index.html")
    if not frontend_file.exists():
        print("❌ Frontend file not found")
        return False
    
    try:
        # Open in default browser
        webbrowser.open("http://localhost:5000")
        print("✓ Frontend opened in browser")
        return True
    except Exception as e:
        print(f"❌ Error opening frontend: {e}")
        return False

def main():
    """Main application launcher"""
    print("=" * 60)
    print("ENERGY MANAGEMENT SYSTEM - FULL STACK APPLICATION")
    print("=" * 60)
    print("Starting 20-Unit Building Energy Management System...")
    print()
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Check data files
    if not check_data_files():
        return 1
    
    print()
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        return 1
    
    print()
    
    # Open frontend
    if not open_frontend():
        print("You can manually open: http://localhost:5000")
    
    print()
    print("=" * 60)
    print("✅ APPLICATION STARTED SUCCESSFULLY")
    print("=" * 60)
    print("🌐 Frontend Dashboard: http://localhost:5000")
    print("🔌 API Endpoints: http://localhost:5000/api")
    print("📊 Features:")
    print("  - Real-time energy monitoring")
    print("  - PV generation tracking")
    print("  - Battery optimization")
    print("  - Family consumption breakdown")
    print("  - TOU pricing analysis")
    print()
    print("Press Ctrl+C to stop the application")
    print("=" * 60)
    
    try:
        # Keep the application running
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down application...")
        backend_process.terminate()
        backend_process.wait()
        print("✓ Application stopped")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

