"""
Check for multicollinearity and ill-conditioning in the master dataset.
Creates visualizations and reports for thesis documentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from numpy.linalg import cond

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    VIF_AVAILABLE = True
except ImportError:
    VIF_AVAILABLE = False


def analyze_multicollinearity(csv_path: str, output_dir: str):
    """
    Analyze dataset for multicollinearity, ill-conditioning, and feature redundancy.
    """
    df = pd.read_csv(csv_path, parse_dates=['time'])
    df = df.set_index('time').sort_index()
    
    # Select numeric features
    numeric_features = ['total_load', 'temp', 'humidity', 'wind_speed', 'clouds',
                        'PV_true', 'Gb', 'Gd', 'Gr', 'H_sun', 'T2m', 'WS10m']
    numeric_features = [f for f in numeric_features if f in df.columns]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Correlation matrix visualization
    corr_matrix = df[numeric_features].corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, 
                mask=mask, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Matrix\n(Redundant pairs highlighted)', 
                 fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path / 'multicollinearity_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[ok] Correlation matrix saved: {output_path / 'multicollinearity_correlation_matrix.png'}")
    
    # 2. VIF visualization
    if VIF_AVAILABLE:
        X_clean = df[numeric_features].dropna()
        vif_values = [variance_inflation_factor(X_clean.values, i) 
                     for i in range(len(numeric_features))]
        vif_df = pd.DataFrame({'Feature': numeric_features, 'VIF': vif_values})
        vif_df = vif_df.sort_values('VIF', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['red' if v > 10 else 'orange' if v > 5 else 'green' for v in vif_df['VIF']]
        bars = ax.barh(vif_df['Feature'], vif_df['VIF'], color=colors, alpha=0.7)
        ax.axvline(x=5, color='orange', linestyle='--', linewidth=2, label='VIF = 5 (moderate)')
        ax.axvline(x=10, color='red', linestyle='--', linewidth=2, label='VIF = 10 (high)')
        ax.set_xlabel('Variance Inflation Factor (VIF)', fontsize=12, fontweight='bold')
        ax.set_title('Multicollinearity Check: VIF Values\n(VIF > 10 indicates high multicollinearity)', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(output_path / 'multicollinearity_vif.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[ok] VIF plot saved: {output_path / 'multicollinearity_vif.png'}")
    
    # 3. Condition number check
    X = df[numeric_features].values
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    X_cov = X_std.T @ X_std
    cond_num = cond(X_cov)
    
    # 4. Generate text report
    report = []
    report.append("=" * 70)
    report.append("MULTICOLLINEARITY AND ILL-CONDITIONING ANALYSIS REPORT")
    report.append("=" * 70)
    report.append(f"\nDataset: {csv_path}")
    report.append(f"Features analyzed: {len(numeric_features)}")
    report.append(f"Total rows: {len(df)}")
    report.append(f"Date range: {df.index.min()} to {df.index.max()}")
    
    report.append("\n" + "=" * 70)
    report.append("1. CONDITION NUMBER (ILL-CONDITIONING CHECK)")
    report.append("=" * 70)
    report.append(f"Condition number of X^T X: {cond_num:.2e}")
    if cond_num > 1e12:
        report.append("  ⚠ CRITICAL: Matrix is ill-conditioned (condition number > 1e12)")
        report.append("    This will cause numerical instability in optimization!")
    elif cond_num > 1e6:
        report.append("  ⚠ WARNING: Matrix is poorly conditioned (condition number > 1e6)")
        report.append("    May cause issues in optimization/regression")
    else:
        report.append("  ✓ Matrix is well-conditioned")
        report.append("    Suitable for optimization and regression")
    
    report.append("\n" + "=" * 70)
    report.append("2. HIGH CORRELATIONS (|ρ| > 0.7)")
    report.append("=" * 70)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))
                report.append(f"  {corr_matrix.columns[i]:15s} <-> {corr_matrix.columns[j]:15s}: {corr_val:7.3f}")
    
    if not high_corr_pairs:
        report.append("  ✓ No high correlations found")
    else:
        report.append(f"\n  Found {len(high_corr_pairs)} high correlation pairs")
    
    if VIF_AVAILABLE:
        report.append("\n" + "=" * 70)
        report.append("3. VARIANCE INFLATION FACTOR (VIF)")
        report.append("=" * 70)
        report.append("VIF > 10 = high multicollinearity, VIF > 5 = moderate")
        report.append("\nVIF values:")
        for _, row in vif_df.iterrows():
            vif_val = row['VIF']
            if vif_val > 10:
                status = "⚠ CRITICAL"
            elif vif_val > 5:
                status = "⚠ WARNING"
            else:
                status = "✓ OK"
            report.append(f"  {row['Feature']:15s}: VIF = {vif_val:7.2f} {status}")
        
        high_vif = vif_df[vif_df['VIF'] > 10]
        if len(high_vif) > 0:
            report.append(f"\n⚠ {len(high_vif)} features with VIF > 10 (high multicollinearity)")
    
    report.append("\n" + "=" * 70)
    report.append("4. SPECIFIC ISSUES IDENTIFIED")
    report.append("=" * 70)
    
    issues = []
    if 'T2m' in df.columns and 'temp' in df.columns:
        t2m_temp_corr = df[['T2m', 'temp']].corr().iloc[0, 1]
        report.append(f"T2m (PVGIS) vs temp (OpenWeather): ρ = {t2m_temp_corr:.3f}")
        if abs(t2m_temp_corr) > 0.95:
            report.append("  ⚠ CRITICAL: T2m and temp are nearly identical (redundant features)")
            report.append("    Recommendation: Use only ONE in forecasting models")
            issues.append("Redundant temperature features (T2m and temp)")
    
    if 'WS10m' in df.columns and 'wind_speed' in df.columns:
        ws_corr = df[['WS10m', 'wind_speed']].corr().iloc[0, 1]
        report.append(f"\nWS10m (PVGIS) vs wind_speed (OpenWeather): ρ = {ws_corr:.3f}")
        if abs(ws_corr) > 0.9:
            report.append("  ⚠ WARNING: WS10m and wind_speed are highly correlated")
            issues.append("Redundant wind speed features")
    
    # Expected correlations (not issues)
    if 'Gb' in df.columns and 'PV_true' in df.columns:
        gb_pv_corr = df[['Gb', 'PV_true']].corr().iloc[0, 1]
        report.append(f"\nGb vs PV_true: ρ = {gb_pv_corr:.3f}")
        report.append("  ✓ Expected: Gb is the primary driver of PV (this is correct)")
        report.append("    High correlation is expected and desired for PV forecasting")
    
    report.append("\n" + "=" * 70)
    report.append("5. RECOMMENDATIONS FOR THESIS")
    report.append("=" * 70)
    
    if cond_num > 1e6 or (VIF_AVAILABLE and len(vif_df[vif_df['VIF'] > 10]) > 0):
        report.append("\n⚠ MULTICOLLINEARITY ISSUES DETECTED:")
        report.append("\n1. FEATURE SELECTION:")
        report.append("   - Remove redundant features:")
        if 'T2m' in df.columns and 'temp' in df.columns:
            report.append("     * Use either T2m (PVGIS) OR temp (OpenWeather), not both")
        report.append("   - For PV forecasting: Use Gb as primary feature (VIF issue due to")
        report.append("     correlation with PV_true is expected - Gb is the target driver)")
        report.append("\n2. MODELING STRATEGIES:")
        report.append("   - Use regularization (Ridge/Lasso) in regression models")
        report.append("   - Apply feature selection (e.g., recursive feature elimination)")
        report.append("   - Consider dimensionality reduction (PCA) if needed")
        report.append("\n3. THESIS DOCUMENTATION:")
        report.append("   - Document these checks in 'Data Quality and Preprocessing' section")
        report.append("   - Explain that high correlations between PV features and PV_true")
        report.append("     are EXPECTED (these are the drivers, not problems)")
        report.append("   - Note that redundant features (T2m vs temp) were identified and")
        report.append("     only one was used in final models")
        report.append("   - Include correlation matrix figure in appendix")
    else:
        report.append("\n✓ NO CRITICAL ISSUES:")
        report.append("   - Matrix is well-conditioned")
        report.append("   - No problematic multicollinearity")
        report.append("   - Data is suitable for forecasting and optimization")
        report.append("\nTHESIS DOCUMENTATION:")
        report.append("   - Briefly mention that multicollinearity checks were performed")
        report.append("   - Note that condition number is acceptable")
    
    report.append("\n" + "=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    # Save report
    report_text = "\n".join(report)
    with open(output_path / 'multicollinearity_report.txt', 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\n[ok] Report saved: {output_path / 'multicollinearity_report.txt'}")
    
    return {
        'condition_number': cond_num,
        'high_corr_pairs': high_corr_pairs,
        'vif_data': vif_df if VIF_AVAILABLE else None,
        'issues': issues
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Check multicollinearity in dataset')
    parser.add_argument('--input', type=str,
                       default='/Users/mariabigonah/Desktop/thesis/code/outputs/MASTER_20_APARTMENTS_2022_2023.csv',
                       help='Path to master dataset CSV')
    parser.add_argument('--output', type=str,
                       default='/Users/mariabigonah/Desktop/thesis/code/outputs/multicollinearity_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    analyze_multicollinearity(args.input, args.output)

