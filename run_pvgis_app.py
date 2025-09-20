#!/usr/bin/env python3
"""
Run PVGIS Full-Stack Application
Starts the PVGIS data visualization dashboard
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['flask', 'flask-cors', 'pandas', 'numpy', 'pyyaml', 'requests']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_data_files():
    """Check if data files exist"""
    data_dir = "project/data"
    required_files = [
        "tou_24h.csv",
        "load_24h.csv", 
        "battery.yaml"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print("⚠️ Some data files are missing:")
        for file in missing_files:
            print(f"   - {file}")
        print("The application will use PVGIS data for PV generation.")
    else:
        print("✅ All data files found")
    
    return True

def start_pvgis_app():
    """Start the PVGIS application"""
    print("=" * 60)
    print("PVGIS FULL-STACK APPLICATION")
    print("=" * 60)
    print("Starting PVGIS data visualization dashboard...")
    print()
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check data files
    check_data_files()
    print()
    
    # Change to backend directory
    backend_dir = os.path.join(os.path.dirname(__file__), "backend")
    if not os.path.exists(backend_dir):
        print("❌ Backend directory not found")
        return False
    
    os.chdir(backend_dir)
    
    # Start the Flask application
    try:
        print("🚀 Starting PVGIS Flask application...")
        print("📊 Dashboard will be available at: http://localhost:5001")
        print("🔗 PVGIS API: https://re.jrc.ec.europa.eu/pvg_tools/en/")
        print("📍 Location: Turin, Italy (45.0703°N, 7.6869°E)")
        print("📅 Data Range: 2005-2023 (19 years)")
        print()
        print("Press Ctrl+C to stop the application")
        print("=" * 60)
        
        # Run the Flask app
        subprocess.run([sys.executable, "pvgis_app.py"])
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return False

def main():
    """Main function"""
    try:
        success = start_pvgis_app()
        if success:
            print("✅ Application stopped successfully")
        else:
            print("❌ Application failed to start")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

