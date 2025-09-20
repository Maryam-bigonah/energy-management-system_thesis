#!/usr/bin/env python3
"""
Test script for the Energy Management System
Verifies that all components are working correctly
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import flask
        print("✓ Flask imported successfully")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    try:
        import pandas
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import yaml
        print("✓ PyYAML imported successfully")
    except ImportError as e:
        print(f"❌ PyYAML import failed: {e}")
        return False
    
    return True

def test_data_files():
    """Test if all required data files exist"""
    print("\nTesting data files...")
    
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
    
    all_exist = True
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            print(f"✓ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_exist = False
    
    return all_exist

def test_backend_import():
    """Test if backend can be imported"""
    print("\nTesting backend import...")
    
    try:
        # Add backend directory to path
        sys.path.insert(0, str(Path("backend")))
        
        # Import the app
        from app import app, data_manager
        
        print("✓ Backend app imported successfully")
        print(f"✓ Data manager loaded: {data_manager.load_data is not None}")
        
        return True
    except Exception as e:
        print(f"❌ Backend import failed: {e}")
        return False

def test_data_loading():
    """Test if data can be loaded correctly"""
    print("\nTesting data loading...")
    
    try:
        sys.path.insert(0, str(Path("backend")))
        from app import EnergyDataManager
        
        data_manager = EnergyDataManager()
        
        # Test summary generation
        summary = data_manager.get_system_summary()
        if "building" in summary:
            print("✓ System summary generated successfully")
            print(f"  - Daily consumption: {summary['building']['daily_consumption_kwh']} kWh")
            print(f"  - Daily PV generation: {summary['pv_system']['daily_generation_kwh']} kWh")
        else:
            print("❌ System summary generation failed")
            return False
        
        # Test hourly data
        load_data = data_manager.get_hourly_data("load", "24h")
        if "values" in load_data and len(load_data["values"]) == 24:
            print("✓ 24-hour load data loaded successfully")
        else:
            print("❌ 24-hour load data loading failed")
            return False
        
        # Test family breakdown
        family_data = data_manager.get_family_breakdown()
        if len(family_data) == 4:
            print("✓ Family breakdown data loaded successfully")
        else:
            print("❌ Family breakdown data loading failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_frontend_files():
    """Test if frontend files exist"""
    print("\nTesting frontend files...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory missing")
        return False
    
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        print("✓ Frontend index.html exists")
        
        # Check file size
        file_size = index_file.stat().st_size
        if file_size > 1000:  # Should be substantial
            print(f"✓ Frontend file size: {file_size} bytes")
            return True
        else:
            print("❌ Frontend file seems too small")
            return False
    else:
        print("❌ Frontend index.html missing")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("ENERGY MANAGEMENT SYSTEM - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Files", test_data_files),
        ("Backend Import", test_backend_import),
        ("Data Loading", test_data_loading),
        ("Frontend Files", test_frontend_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The application is ready to run.")
        print("Run: python3 run_app.py")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please fix the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

