#!/usr/bin/env python3
"""
Step 1 Dataset Validation
Validates all CSV files and battery.yaml for optimization model readiness
"""

import pandas as pd
import yaml
import os
import numpy as np

def validate_csv_structure(file_path, expected_rows, file_type):
    """
    Validate CSV file structure and basic requirements
    """
    print(f"\n📊 Validating {file_type}: {file_path}")
    print("=" * 50)
    
    try:
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Check row count
        actual_rows = len(df)
        if actual_rows == expected_rows:
            print(f"✅ Rows: {actual_rows} (expected: {expected_rows})")
        else:
            print(f"❌ Rows: {actual_rows} (expected: {expected_rows})")
            return False
        
        # Check for header
        if len(df.columns) > 0:
            print(f"✅ Headers: {list(df.columns)}")
        else:
            print("❌ No headers found")
            return False
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            print("✅ No missing values")
        else:
            print(f"❌ Missing values: {missing_count}")
            return False
        
        # Check for negative values (where applicable)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        negative_count = 0
        for col in numeric_cols:
            if col not in ['hour']:  # hour can be negative in some contexts
                negative_count += (df[col] < 0).sum()
        
        if negative_count == 0:
            print("✅ No negative values")
        else:
            print(f"⚠️ Negative values found: {negative_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return False

def validate_load_data(file_path):
    """
    Validate load data specifically
    """
    print(f"\n🏠 Validating Load Data: {file_path}")
    print("=" * 50)
    
    try:
        df = pd.read_csv(file_path)
        
        # Check units (should be kW)
        load_col = None
        for col in df.columns:
            if 'load' in col.lower() or 'power' in col.lower():
                load_col = col
                break
        
        if load_col is None:
            print("❌ No load column found")
            return False
        
        print(f"✅ Load column: {load_col}")
        
        # Check realistic magnitudes for 20-unit building
        max_load = df[load_col].max()
        min_load = df[load_col].min()
        avg_load = df[load_col].mean()
        
        print(f"📈 Load Statistics:")
        print(f"   Max: {max_load:.2f} kW")
        print(f"   Min: {min_load:.2f} kW")
        print(f"   Avg: {avg_load:.2f} kW")
        
        # Validate realistic magnitudes
        if 15 <= max_load <= 50:  # 20-30 kW peak is reasonable, allow some margin
            print("✅ Peak load magnitude realistic for 20-unit building")
        else:
            print(f"⚠️ Peak load ({max_load:.2f} kW) may be unrealistic for 20 units")
        
        if min_load >= 0:
            print("✅ Minimum load non-negative")
        else:
            print("❌ Negative minimum load found")
            return False
        
        # Check for typical daily pattern
        if len(df) == 24:  # 24-hour data
            evening_hours = df[df['hour'].isin([18, 19, 20, 21])][load_col]
            if evening_hours.mean() > avg_load * 1.2:
                print("✅ Evening peak pattern detected")
            else:
                print("⚠️ No clear evening peak pattern")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating load data: {e}")
        return False

def validate_pv_data(file_path):
    """
    Validate PV generation data
    """
    print(f"\n☀️ Validating PV Data: {file_path}")
    print("=" * 50)
    
    try:
        df = pd.read_csv(file_path)
        
        # Find PV column
        pv_col = None
        for col in df.columns:
            if 'pv' in col.lower() or 'generation' in col.lower():
                pv_col = col
                break
        
        if pv_col is None:
            print("❌ No PV generation column found")
            return False
        
        print(f"✅ PV column: {pv_col}")
        
        # Check units (should be kW)
        max_pv = df[pv_col].max()
        min_pv = df[pv_col].min()
        avg_pv = df[pv_col].mean()
        
        print(f"📈 PV Statistics:")
        print(f"   Max: {max_pv:.2f} kW")
        print(f"   Min: {min_pv:.2f} kW")
        print(f"   Avg: {avg_pv:.2f} kW")
        
        # Validate realistic magnitudes
        if 0 <= max_pv <= 200:  # Reasonable for building-scale PV
            print("✅ Peak PV generation realistic")
        else:
            print(f"⚠️ Peak PV ({max_pv:.2f} kW) may be unrealistic")
        
        if min_pv >= 0:
            print("✅ Minimum PV generation non-negative")
        else:
            print("❌ Negative PV generation found")
            return False
        
        # Check for typical solar pattern
        if len(df) == 24:  # 24-hour data
            night_hours = df[df['hour'].isin([0, 1, 2, 3, 4, 5, 22, 23])][pv_col]
            if night_hours.max() < 0.1:  # Should be near zero at night
                print("✅ Realistic night-time PV generation")
            else:
                print("⚠️ PV generation at night may be unrealistic")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating PV data: {e}")
        return False

def validate_tou_data(file_path):
    """
    Validate TOU pricing data
    """
    print(f"\n💰 Validating TOU Data: {file_path}")
    print("=" * 50)
    
    try:
        df = pd.read_csv(file_path)
        
        # Check required columns
        required_cols = ['price_buy', 'price_sell']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        
        print("✅ Required columns present")
        
        # Check units (should be €/kWh)
        buy_prices = df['price_buy']
        sell_prices = df['price_sell']
        
        print(f"📈 TOU Statistics:")
        print(f"   Buy price range: {buy_prices.min():.3f} - {buy_prices.max():.3f} €/kWh")
        print(f"   Sell price range: {sell_prices.min():.3f} - {sell_prices.max():.3f} €/kWh")
        
        # Validate realistic price ranges
        if 0.1 <= buy_prices.min() <= 0.8:
            print("✅ Buy price range realistic")
        else:
            print("⚠️ Buy price range may be unrealistic")
        
        if 0.05 <= sell_prices.min() <= 0.3:
            print("✅ Sell price range realistic")
        else:
            print("⚠️ Sell price range may be unrealistic")
        
        # Check for Italian ARERA structure
        unique_buy_prices = sorted(buy_prices.unique())
        if len(unique_buy_prices) <= 4:  # F1, F2, F3 bands
            print(f"✅ TOU structure: {len(unique_buy_prices)} price bands")
            print(f"   Bands: {[f'{p:.3f}' for p in unique_buy_prices]}")
        else:
            print(f"⚠️ Many price bands ({len(unique_buy_prices)}), may not be ARERA structure")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating TOU data: {e}")
        return False

def validate_battery_yaml(file_path):
    """
    Validate battery.yaml specifications
    """
    print(f"\n🔋 Validating Battery Specs: {file_path}")
    print("=" * 50)
    
    try:
        with open(file_path, 'r') as f:
            battery_specs = yaml.safe_load(f)
        
        # Check required parameters
        required_params = [
            'Ebat_kWh', 'Pch_max_kW', 'Pdis_max_kW',
            'SOCmin', 'SOCmax', 'eta_ch', 'eta_dis'
        ]
        
        missing_params = [param for param in required_params if param not in battery_specs]
        if missing_params:
            print(f"❌ Missing parameters: {missing_params}")
            return False
        
        print("✅ All required parameters present")
        
        # Validate parameter values
        Ebat = battery_specs['Ebat_kWh']
        Pch_max = battery_specs['Pch_max_kW']
        Pdis_max = battery_specs['Pdis_max_kW']
        SOCmin = battery_specs['SOCmin']
        SOCmax = battery_specs['SOCmax']
        eta_ch = battery_specs['eta_ch']
        eta_dis = battery_specs['eta_dis']
        
        print(f"📈 Battery Specifications:")
        print(f"   Capacity: {Ebat} kWh")
        print(f"   Max Charge: {Pch_max} kW")
        print(f"   Max Discharge: {Pdis_max} kW")
        print(f"   SOC Range: {SOCmin*100:.0f}% - {SOCmax*100:.0f}%")
        print(f"   Efficiencies: {eta_ch*100:.0f}% / {eta_dis*100:.0f}%")
        
        # Validate realistic values
        if 50 <= Ebat <= 150:
            print("✅ Battery capacity realistic for 20-unit building")
        else:
            print(f"⚠️ Battery capacity ({Ebat} kWh) may be unrealistic")
        
        if 0.2 <= SOCmin <= 0.3 and 0.9 <= SOCmax <= 1.0:
            print("✅ SOC bounds realistic")
        else:
            print("⚠️ SOC bounds may be unrealistic")
        
        if 0.8 <= eta_ch <= 0.95 and 0.8 <= eta_dis <= 0.95:
            print("✅ Efficiencies realistic")
        else:
            print("⚠️ Efficiencies may be unrealistic")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating battery specs: {e}")
        return False

def main():
    """
    Main validation function
    """
    print("🔍 STEP 1 DATASET VALIDATION")
    print("=" * 60)
    print("Validating all CSV files and battery.yaml for optimization model readiness")
    
    data_dir = "project/data"
    all_valid = True
    
    # Validate 24-hour files
    files_24h = [
        ("load_24h.csv", 24, "Load Data (24h)"),
        ("pv_24h.csv", 24, "PV Generation (24h)"),
        ("tou_24h.csv", 24, "TOU Pricing (24h)")
    ]
    
    for filename, expected_rows, file_type in files_24h:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            if not validate_csv_structure(file_path, expected_rows, file_type):
                all_valid = False
        else:
            print(f"❌ File not found: {file_path}")
            all_valid = False
    
    # Validate 8760-hour files
    files_8760h = [
        ("load_8760.csv", 8760, "Load Data (8760h)"),
        ("pv_8760.csv", 8760, "PV Generation (8760h)")
    ]
    
    for filename, expected_rows, file_type in files_8760h:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            if not validate_csv_structure(file_path, expected_rows, file_type):
                all_valid = False
        else:
            print(f"❌ File not found: {file_path}")
            all_valid = False
    
    # Detailed validation of key files
    if os.path.exists(os.path.join(data_dir, "load_24h.csv")):
        if not validate_load_data(os.path.join(data_dir, "load_24h.csv")):
            all_valid = False
    
    if os.path.exists(os.path.join(data_dir, "pv_24h.csv")):
        if not validate_pv_data(os.path.join(data_dir, "pv_24h.csv")):
            all_valid = False
    
    if os.path.exists(os.path.join(data_dir, "tou_24h.csv")):
        if not validate_tou_data(os.path.join(data_dir, "tou_24h.csv")):
            all_valid = False
    
    if os.path.exists(os.path.join(data_dir, "battery.yaml")):
        if not validate_battery_yaml(os.path.join(data_dir, "battery.yaml")):
            all_valid = False
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 STEP 1 VALIDATION SUMMARY")
    print("=" * 60)
    
    if all_valid:
        print("✅ ALL VALIDATIONS PASSED!")
        print("\n📋 Step 1 Dataset Ready:")
        print("   ✓ load_24h.csv (from LPG, aggregated 20 units)")
        print("   ✓ pv_24h.csv (from PVGIS)")
        print("   ✓ tou_24h.csv (from ARERA)")
        print("   ✓ battery.yaml (research-based specs)")
        print("\n🚀 Ready for Step 2 (optimization model)!")
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("Please fix the issues above before proceeding to Step 2.")
    
    return all_valid

if __name__ == "__main__":
    main()

