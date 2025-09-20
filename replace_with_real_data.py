#!/usr/bin/env python3
"""
Replace Generated Data with Real Data
This script replaces the generated/simulated data with real data from specified sources
"""

import os
import shutil
import pandas as pd
from datetime import datetime

def backup_generated_data():
    """
    Backup the current generated data files
    """
    print("📦 Backing up generated data files...")
    
    data_dir = "project/data"
    backup_dir = "project/data/generated_backup"
    
    os.makedirs(backup_dir, exist_ok=True)
    
    # Files to backup
    files_to_backup = [
        "load_24h.csv",
        "load_8760.csv", 
        "pv_24h.csv",
        "pv_8760.csv",
        "tou_24h.csv",
        "tou_8760.csv"
    ]
    
    backed_up = []
    for file in files_to_backup:
        src = os.path.join(data_dir, file)
        dst = os.path.join(backup_dir, file)
        
        if os.path.exists(src):
            shutil.copy2(src, dst)
            backed_up.append(file)
            print(f"  ✓ Backed up: {file}")
        else:
            print(f"  ⚠️ Not found: {file}")
    
    print(f"✅ Backed up {len(backed_up)} files to {backup_dir}")
    return backed_up

def check_real_data_availability():
    """
    Check which real data files are available
    """
    print("🔍 Checking real data availability...")
    
    data_dir = "project/data"
    
    # Check for real data files
    real_files = {
        "PV Data": {
            "24h": "pv_24h_real.csv",
            "8760h": "pv_8760_real.csv"
        },
        "TOU Data": {
            "24h": "tou_24h_real.csv", 
            "8760h": "tou_8760_real.csv"
        },
        "Load Data": {
            "24h": "load_24h_real.csv",
            "8760h": "load_8760_real.csv"
        }
    }
    
    available = {}
    for data_type, files in real_files.items():
        available[data_type] = {}
        for period, filename in files.items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                available[data_type][period] = True
                print(f"  ✅ {data_type} ({period}): {filename}")
            else:
                available[data_type][period] = False
                print(f"  ❌ {data_type} ({period}): {filename} - NOT FOUND")
    
    return available

def replace_data_files(available):
    """
    Replace generated data with real data where available
    """
    print("🔄 Replacing data files...")
    
    data_dir = "project/data"
    
    replacements = {
        "pv_24h_real.csv": "pv_24h.csv",
        "pv_8760_real.csv": "pv_8760.csv",
        "tou_24h_real.csv": "tou_24h.csv",
        "tou_8760_real.csv": "tou_8760.csv",
        "load_24h_real.csv": "load_24h.csv",
        "load_8760_real.csv": "load_8760.csv"
    }
    
    replaced = []
    for real_file, target_file in replacements.items():
        real_path = os.path.join(data_dir, real_file)
        target_path = os.path.join(data_dir, target_file)
        
        if os.path.exists(real_path):
            shutil.copy2(real_path, target_path)
            replaced.append(f"{real_file} → {target_file}")
            print(f"  ✅ Replaced: {target_file}")
        else:
            print(f"  ⚠️ Skipped: {target_file} (no real data available)")
    
    return replaced

def validate_real_data():
    """
    Validate the real data files
    """
    print("🔍 Validating real data...")
    
    data_dir = "project/data"
    
    # Check file sizes and basic structure
    files_to_check = [
        "load_24h.csv",
        "load_8760.csv",
        "pv_24h.csv", 
        "pv_8760.csv",
        "tou_24h.csv",
        "tou_8760.csv"
    ]
    
    validation_results = {}
    
    for file in files_to_check:
        filepath = os.path.join(data_dir, file)
        
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                
                # Basic validation
                row_count = len(df)
                col_count = len(df.columns)
                file_size = os.path.getsize(filepath)
                
                # Expected row counts
                expected_rows = 24 if "24h" in file else 8760
                
                validation_results[file] = {
                    "exists": True,
                    "rows": row_count,
                    "columns": col_count,
                    "size_kb": file_size / 1024,
                    "expected_rows": expected_rows,
                    "valid_rows": row_count == expected_rows
                }
                
                status = "✅" if row_count == expected_rows else "⚠️"
                print(f"  {status} {file}: {row_count} rows, {col_count} cols, {file_size/1024:.1f} KB")
                
            except Exception as e:
                validation_results[file] = {
                    "exists": True,
                    "error": str(e)
                }
                print(f"  ❌ {file}: Error reading file - {e}")
        else:
            validation_results[file] = {"exists": False}
            print(f"  ❌ {file}: File not found")
    
    return validation_results

def create_data_source_report(available, replaced, validation_results):
    """
    Create a report of data sources and replacements
    """
    print("📊 Creating data source report...")
    
    report = f"""
# Data Source Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Real Data Availability
"""
    
    for data_type, files in available.items():
        report += f"\n### {data_type}\n"
        for period, is_available in files.items():
            status = "✅ Available" if is_available else "❌ Not Available"
            report += f"- {period}: {status}\n"
    
    report += f"\n## Files Replaced\n"
    for replacement in replaced:
        report += f"- {replacement}\n"
    
    report += f"\n## Validation Results\n"
    for file, results in validation_results.items():
        if results.get("exists"):
            if "error" in results:
                report += f"- {file}: ❌ Error - {results['error']}\n"
            else:
                status = "✅ Valid" if results.get("valid_rows") else "⚠️ Invalid rows"
                report += f"- {file}: {status} ({results.get('rows', 'N/A')} rows)\n"
        else:
            report += f"- {file}: ❌ Not found\n"
    
    # Save report
    with open("project/data/DATA_SOURCE_REPORT.md", "w") as f:
        f.write(report)
    
    print("✅ Data source report saved: project/data/DATA_SOURCE_REPORT.md")

def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("REPLACING GENERATED DATA WITH REAL DATA")
    print("=" * 60)
    print("This script will replace generated/simulated data with real data")
    print("from the specified sources (PVGIS, LPG, ARERA)")
    print()
    
    try:
        # Step 1: Backup generated data
        backed_up = backup_generated_data()
        print()
        
        # Step 2: Check real data availability
        available = check_real_data_availability()
        print()
        
        # Step 3: Replace data files
        replaced = replace_data_files(available)
        print()
        
        # Step 4: Validate real data
        validation_results = validate_real_data()
        print()
        
        # Step 5: Create report
        create_data_source_report(available, replaced, validation_results)
        print()
        
        # Summary
        print("=" * 60)
        print("REPLACEMENT SUMMARY")
        print("=" * 60)
        
        total_available = sum(sum(files.values()) for files in available.values())
        total_replaced = len(replaced)
        
        print(f"📊 Real data files available: {total_available}")
        print(f"🔄 Files replaced: {total_replaced}")
        print(f"📦 Files backed up: {len(backed_up)}")
        
        if total_replaced > 0:
            print("\n✅ SUCCESS: Some data has been replaced with real data")
            print("📋 Check project/data/DATA_SOURCE_REPORT.md for details")
        else:
            print("\n⚠️ WARNING: No real data files were found")
            print("📋 You need to fetch real data from:")
            print("   - PVGIS: https://re.jrc.ec.europa.eu/pvg_tools/en/")
            print("   - LPG: https://www.loadprofilegenerator.de")
            print("   - ARERA: https://www.arera.it")
        
        return total_replaced > 0
        
    except Exception as e:
        print(f"❌ Error during data replacement: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Data replacement completed successfully!")
    else:
        print("\n❌ Data replacement failed or no real data available")

