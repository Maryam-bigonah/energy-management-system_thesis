"""
Create Feature Summary Table
Generates a summary of all features in the master dataset
"""

import pandas as pd

def create_feature_summary(df_master):
    """
    Create feature summary table from master dataset
    
    Parameters:
    -----------
    df_master : pd.DataFrame
        Master dataset with all features
    
    Returns:
    --------
    pd.DataFrame with feature summary
    """
    
    feature_summary = []
    
    # Apartment loads
    apt_cols = [col for col in df_master.columns if col.startswith('apartment_')]
    for i, col in enumerate(sorted(apt_cols), 1):
        # Determine family type based on assignment (0-4: couple_working, etc.)
        apt_num = int(col.split('_')[1])
        family_type_idx = (apt_num - 1) // 5
        family_types = ['couple_working', 'family_one_child', 'one_working', 'retired']
        family_type = family_types[family_type_idx]
        
        feature_summary.append({
            'Feature': col,
            'Type': 'numeric',
            'Units': 'kWh/hour (≈kW)',
            'Range': f"{df_master[col].min():.4f} - {df_master[col].max():.4f}",
            'Family Type': family_type,
            'Why it helps': f'Load for apartment {apt_num} ({family_type})'
        })
    
    # PV generation
    if 'pv_1kw' in df_master.columns:
        feature_summary.append({
            'Feature': 'pv_1kw',
            'Type': 'numeric',
            'Units': 'kW',
            'Range': f"{df_master['pv_1kw'].min():.4f} - {df_master['pv_1kw'].max():.4f}",
            'Family Type': 'N/A',
            'Why it helps': 'PV generation from PVGIS'
        })
    
    # Calendar features
    calendar_features = [
        ('hour', 'numeric', '0-23', 'N/A', 'Captures daily cycle'),
        ('dayofweek', 'numeric', '0-6 (Mon=0, Sun=6)', 'N/A', 'Captures weekly pattern'),
        ('month', 'numeric', '1-12', 'N/A', 'Month trend'),
        ('is_weekend', 'binary', '0 or 1', 'N/A', 'Differentiates weekend load'),
        ('season', 'categorical', '0-3 (winter, spring, summer, autumn)', 'N/A', 'Captures seasonal pattern')
    ]
    
    for feat_name, feat_type, feat_range, family_type, why in calendar_features:
        if feat_name in df_master.columns:
            feature_summary.append({
                'Feature': feat_name,
                'Type': feat_type,
                'Units': feat_range,
                'Range': f"Min: {df_master[feat_name].min()}, Max: {df_master[feat_name].max()}",
                'Family Type': family_type,
                'Why it helps': why
            })
    
    return pd.DataFrame(feature_summary)


def display_feature_table(df_summary):
    """
    Display feature summary in a formatted table
    """
    print("=" * 100)
    print("Updated Feature List for Each Timestamp")
    print("=" * 100)
    print()
    
    # Group by type
    print("## Input Features (Past 24 Hours)")
    print()
    
    # Apartment loads
    apt_features = df_summary[df_summary['Feature'].str.startswith('apartment_')]
    print(f"### Apartment Loads ({len(apt_features)} apartments)")
    print("| Feature | Type | Units | Family Type | Why it helps |")
    print("|---------|------|-------|--------------|--------------|")
    for _, row in apt_features.iterrows():
        print(f"| `{row['Feature']}` | {row['Type']} | {row['Units']} | {row['Family Type']} | {row['Why it helps']} |")
    print()
    
    # PV generation
    pv_features = df_summary[df_summary['Feature'] == 'pv_1kw']
    print(f"### PV Generation")
    print("| Feature | Type | Units | Why it helps |")
    print("|---------|------|-------|--------------|")
    for _, row in pv_features.iterrows():
        print(f"| `{row['Feature']}` | {row['Type']} | {row['Units']} | {row['Why it helps']} |")
    print()
    
    # Calendar features
    calendar_features = df_summary[~df_summary['Feature'].str.startswith('apartment_') & 
                                   (df_summary['Feature'] != 'pv_1kw')]
    print(f"### Calendar Features ({len(calendar_features)} features)")
    print("| Feature | Type | Why it helps |")
    print("|---------|------|--------------|")
    for _, row in calendar_features.iterrows():
        print(f"| `{row['Feature']}` | {row['Type']} | {row['Why it helps']} |")
    print()
    
    # Summary
    print("=" * 100)
    print("Summary")
    print("=" * 100)
    print(f"Total Features: {len(df_summary)}")
    print(f"  - Apartment loads: {len(apt_features)}")
    print(f"  - PV generation: {len(pv_features)}")
    print(f"  - Calendar features: {len(calendar_features)}")
    print()


def main():
    """
    Create feature summary from master dataset
    """
    # If master dataset exists, use it
    # Otherwise, create a sample structure
    
    print("=" * 100)
    print("Feature Summary Generator")
    print("=" * 100)
    print()
    
    # Check if master dataset exists
    master_path = 'data/master_dataset_2024.csv'
    
    try:
        df_master = pd.read_csv(master_path, index_col=0, parse_dates=True)
        print(f"✓ Loaded master dataset from: {master_path}")
        print(f"  Shape: {df_master.shape}")
        print(f"  Date range: {df_master.index.min()} to {df_master.index.max()}")
    except FileNotFoundError:
        print(f"⚠ Master dataset not found at: {master_path}")
        print("  Creating feature summary template...")
        
        # Create sample structure for demonstration
        dates = pd.date_range('2024-01-01', '2024-01-07', freq='h')
        df_master = pd.DataFrame(index=dates)
        
        # Add apartment columns
        for i in range(20):
            df_master[f'apartment_{i+1:02d}'] = 0.5
        
        # Add other features
        df_master['pv_1kw'] = 0.5
        df_master['hour'] = df_master.index.hour
        df_master['dayofweek'] = df_master.index.dayofweek
        df_master['month'] = df_master.index.month
        df_master['is_weekend'] = (df_master.index.dayofweek >= 5).astype(int)
        df_master['season'] = df_master['month'].apply(lambda m: 0 if m in [12,1,2] else (1 if m in [3,4,5] else (2 if m in [6,7,8] else 3)))
        
        print("  ✓ Created sample structure for demonstration")
    
    # Create feature summary
    df_summary = create_feature_summary(df_master)
    
    # Display formatted table
    display_feature_table(df_summary)
    
    # Save to CSV
    output_path = 'data/feature_summary.csv'
    df_summary.to_csv(output_path, index=False)
    print(f"✓ Feature summary saved to: {output_path}")
    
    # Save to markdown
    output_md = 'FEATURE_LIST_DETAILED.md'
    with open(output_md, 'w') as f:
        f.write("# Feature List - Detailed Summary\n\n")
        
        # Write markdown table manually
        f.write("| Feature | Type | Units | Family Type | Why it helps |\n")
        f.write("|---------|------|-------|--------------|--------------|\n")
        for _, row in df_summary.iterrows():
            f.write(f"| `{row['Feature']}` | {row['Type']} | {row['Units']} | {row['Family Type']} | {row['Why it helps']} |\n")
        
        f.write("\n\n")
        f.write("## Season Mapping\n\n")
        f.write("| Season Code | Season Name | Months |\n")
        f.write("|-------------|-------------|--------|\n")
        f.write("| 0 | Winter | December, January, February |\n")
        f.write("| 1 | Spring | March, April, May |\n")
        f.write("| 2 | Summer | June, July, August |\n")
        f.write("| 3 | Autumn | September, October, November |\n")
    
    print(f"✓ Feature summary saved to: {output_md}")
    
    return df_summary


if __name__ == "__main__":
    df_summary = main()

