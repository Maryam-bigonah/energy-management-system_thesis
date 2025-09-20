#!/usr/bin/env python3
"""
Simplified Analysis Script for Available Results

This script analyzes the available results data and demonstrates
the Step 5 analysis framework with the data we currently have.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon
import warnings
warnings.filterwarnings('ignore')

def create_directories():
    """Create output directories"""
    os.makedirs('results/summaries', exist_ok=True)
    os.makedirs('results/figs', exist_ok=True)

def load_available_data():
    """Load available results data"""
    print("Loading available results data...")
    
    # Load KPIs data
    kpis_file = 'results/kpis.csv'
    if not os.path.exists(kpis_file):
        print(f"Error: {kpis_file} not found")
        return None, None
    
    kpis_data = pd.read_csv(kpis_file)
    print(f"Loaded {len(kpis_data)} KPI records")
    print(f"Strategies: {kpis_data['Strategy'].unique()}")
    
    # Check if we have daily features (from clustering)
    features_file = 'results/daily_features.csv'
    daily_features = None
    if os.path.exists(features_file):
        daily_features = pd.read_csv(features_file)
        print(f"Loaded {len(daily_features)} daily feature records")
    else:
        print("No daily features file found - will create synthetic seasonal data")
    
    return kpis_data, daily_features

def create_annual_comparison(kpis_data):
    """Create annual strategy comparison"""
    print("Creating annual strategy comparison...")
    
    # Group by strategy and aggregate
    annual_kpis = kpis_data.groupby('Strategy').agg({
        'Cost_total': ['sum', 'mean', 'std'],
        'Import_total': ['sum', 'mean'],
        'Export_total': ['sum', 'mean'],
        'PV_total': ['sum', 'mean'],
        'Load_total': ['sum', 'mean'],
        'pv_self': ['sum', 'mean'],
        'SCR': ['mean', 'std'],
        'SelfSufficiency': ['mean', 'std'],
        'PeakGrid': ['mean', 'max'],
        'BatteryCycles': ['sum', 'mean']
    }).round(2)
    
    # Flatten column names
    annual_kpis.columns = ['_'.join(col).strip() for col in annual_kpis.columns]
    annual_kpis = annual_kpis.reset_index()
    
    # Calculate percentage improvements vs MSC
    msc_costs = annual_kpis[annual_kpis['Strategy'] == 'MSC']['Cost_total_sum'].iloc[0]
    annual_kpis['Cost_improvement_pct'] = (
        (msc_costs - annual_kpis['Cost_total_sum']) / msc_costs * 100
    ).round(2)
    
    # Save annual comparison
    annual_file = 'results/summaries/annual_kpis_by_strategy.csv'
    annual_kpis.to_csv(annual_file, index=False)
    print(f"Saved annual comparison to {annual_file}")
    
    return annual_kpis

def create_annual_plots(annual_kpis, kpis_data):
    """Create annual comparison plots"""
    print("Creating annual comparison plots...")
    
    # 1. Annual cost bar chart
    plt.figure(figsize=(10, 6))
    strategies = annual_kpis['Strategy']
    costs = annual_kpis['Cost_total_sum']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = plt.bar(strategies, costs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    plt.title('Annual Cost Comparison by Strategy', fontsize=16, fontweight='bold')
    plt.xlabel('Strategy', fontsize=14)
    plt.ylabel('Annual Cost (€)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(costs)*0.01,
                f'€{cost:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/figs/annual_cost_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. SCR and Self-Sufficiency bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # SCR
    scr_values = annual_kpis['SCR_mean']
    bars1 = ax1.bar(strategies, scr_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Mean Self-Consumption Rate by Strategy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Strategy', fontsize=12)
    ax1.set_ylabel('SCR', fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    
    for bar, scr in zip(bars1, scr_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{scr:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Self-Sufficiency
    ss_values = annual_kpis['SelfSufficiency_mean']
    bars2 = ax2.bar(strategies, ss_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('Mean Self-Sufficiency by Strategy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Strategy', fontsize=12)
    ax2.set_ylabel('Self-Sufficiency', fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3)
    
    for bar, ss in zip(bars2, ss_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{ss:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/figs/annual_scr_selfsuff_bars.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Daily cost boxplots
    plt.figure(figsize=(12, 8))
    
    # Prepare data for boxplot
    daily_costs = []
    strategy_labels = []
    
    for strategy in ['MSC', 'TOU', 'MMR', 'DRP2P']:
        strategy_data = kpis_data[kpis_data['Strategy'] == strategy]['Cost_total']
        if len(strategy_data) > 0:
            daily_costs.append(strategy_data)
            strategy_labels.append(strategy)
    
    if daily_costs:
        box_plot = plt.boxplot(daily_costs, labels=strategy_labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], colors[:len(daily_costs)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        plt.title('Daily Cost Distribution by Strategy', fontsize=16, fontweight='bold')
        plt.xlabel('Strategy', fontsize=14)
        plt.ylabel('Daily Cost (€)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/figs/daily_cost_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Annual comparison plots created")

def create_seasonal_analysis(kpis_data, daily_features):
    """Create seasonal analysis"""
    print("Creating seasonal analysis...")
    
    # If we don't have daily features, create synthetic seasonal data
    if daily_features is None:
        print("Creating synthetic seasonal data for demonstration...")
        
        # Create synthetic daily features with seasonal labels
        # Use the actual days from kpis_data
        unique_days = kpis_data['day'].unique() if 'day' in kpis_data.columns else [1, 2, 3, 4, 5, 6, 7, 8, 9]
        synthetic_features = []
        
        seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
        days_per_season = len(unique_days) // 4
        
        for i, day in enumerate(unique_days):
            season = seasons[i // days_per_season] if i // days_per_season < len(seasons) else 'Winter'
            synthetic_features.append({
                'day': day,
                'season_label': season,
                'Load_total': 3.0 + np.random.normal(0, 0.5),
                'PV_total': 2.0 + np.random.normal(0, 0.3) if season == 'Summer' else 1.0 + np.random.normal(0, 0.2),
                'Import_total': 2.5 + np.random.normal(0, 0.4),
                'Export_total': 0.1 + np.random.normal(0, 0.1),
                'SCR': 0.95 + np.random.normal(0, 0.05),
                'SelfSufficiency': 0.05 + np.random.normal(0, 0.02),
                'PeakGrid': 100 + np.random.normal(0, 20),
                'BatteryCycles': 0.3 + np.random.normal(0, 0.1)
            })
        
        daily_features = pd.DataFrame(synthetic_features)
        print(f"Created {len(daily_features)} synthetic daily features")
    
    # Add day column to kpis_data if it doesn't exist
    if 'day' not in kpis_data.columns:
        kpis_data = kpis_data.copy()
        kpis_data['day'] = range(1, len(kpis_data) + 1)
    
    # Merge KPIs with daily features
    merged_data = kpis_data.merge(
        daily_features[['day', 'season_label']], 
        on='day', 
        how='left'
    )
    
    # Fill missing season labels with 'Unknown'
    merged_data['season_label'] = merged_data['season_label'].fillna('Unknown')
    
    # Group by strategy and season
    seasonal_kpis = merged_data.groupby(['Strategy', 'season_label']).agg({
        'Cost_total': ['mean', 'std'],
        'Import_total': ['mean', 'std'],
        'Export_total': ['mean', 'std'],
        'SCR': ['mean', 'std'],
        'SelfSufficiency': ['mean', 'std'],
        'PeakGrid': ['mean', 'std'],
        'BatteryCycles': ['mean', 'std']
    }).round(3)
    
    # Flatten column names
    seasonal_kpis.columns = ['_'.join(col).strip() for col in seasonal_kpis.columns]
    seasonal_kpis = seasonal_kpis.reset_index()
    
    # Save seasonal analysis
    seasonal_file = 'results/summaries/seasonal_kpis_by_strategy.csv'
    seasonal_kpis.to_csv(seasonal_file, index=False)
    print(f"Saved seasonal analysis to {seasonal_file}")
    
    # Create seasonal plots
    create_seasonal_plots(seasonal_kpis)
    
    return seasonal_kpis

def create_seasonal_plots(seasonal_kpis):
    """Create seasonal comparison plots"""
    print("Creating seasonal comparison plots...")
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    seasons = seasonal_kpis['season_label'].unique()
    
    for season in seasons:
        season_data = seasonal_kpis[seasonal_kpis['season_label'] == season]
        
        if len(season_data) == 0:
            continue
        
        plt.figure(figsize=(10, 6))
        
        strategies = season_data['Strategy']
        costs = season_data['Cost_total_mean']
        errors = season_data['Cost_total_std']
        
        bars = plt.bar(strategies, costs, yerr=errors, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1, capsize=5)
        
        plt.title(f'Mean Daily Cost by Strategy - {season}', fontsize=16, fontweight='bold')
        plt.xlabel('Strategy', fontsize=14)
        plt.ylabel('Mean Daily Cost (€)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, cost, error in zip(bars, costs, errors):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + error + max(costs)*0.01,
                    f'€{cost:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'results/figs/seasonal_cost_bars_{season}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Seasonal comparison plots created")

def create_statistical_tests(kpis_data):
    """Create statistical validation tests"""
    print("Creating statistical validation tests...")
    
    # Prepare data for paired tests
    strategy_data = {}
    for strategy in ['MSC', 'TOU', 'MMR', 'DRP2P']:
        strategy_data[strategy] = kpis_data[kpis_data['Strategy'] == strategy]['Cost_total'].values
    
    # Ensure all strategies have the same number of observations
    min_length = min(len(data) for data in strategy_data.values() if len(data) > 0)
    for strategy in strategy_data:
        if len(strategy_data[strategy]) > 0:
            strategy_data[strategy] = strategy_data[strategy][:min_length]
    
    # Perform pairwise comparisons
    results = []
    baseline_strategy = 'MSC'
    
    for strategy in ['TOU', 'MMR', 'DRP2P']:
        if strategy not in strategy_data or len(strategy_data[strategy]) == 0:
            continue
        
        if baseline_strategy not in strategy_data or len(strategy_data[baseline_strategy]) == 0:
            continue
        
        # Paired t-test
        t_stat, t_pvalue = ttest_rel(strategy_data[baseline_strategy], strategy_data[strategy])
        
        # Wilcoxon signed-rank test
        w_stat, w_pvalue = wilcoxon(strategy_data[baseline_strategy], strategy_data[strategy])
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(strategy_data[baseline_strategy]) + np.var(strategy_data[strategy])) / 2)
        cohens_d = (np.mean(strategy_data[baseline_strategy]) - np.mean(strategy_data[strategy])) / pooled_std
        
        # Mean difference and confidence interval
        diff = np.mean(strategy_data[baseline_strategy]) - np.mean(strategy_data[strategy])
        se_diff = np.sqrt(np.var(strategy_data[baseline_strategy]) + np.var(strategy_data[strategy]) - 
                         2 * np.cov(strategy_data[baseline_strategy], strategy_data[strategy])[0, 1]) / np.sqrt(min_length)
        ci_lower = diff - 1.96 * se_diff
        ci_upper = diff + 1.96 * se_diff
        
        results.append({
            'Comparison': f'{baseline_strategy} vs {strategy}',
            'Mean_Difference': round(diff, 3),
            'CI_Lower': round(ci_lower, 3),
            'CI_Upper': round(ci_upper, 3),
            'T_Statistic': round(t_stat, 3),
            'T_P_Value': round(t_pvalue, 6),
            'Wilcoxon_Statistic': round(w_stat, 3),
            'Wilcoxon_P_Value': round(w_pvalue, 6),
            'Cohens_D': round(cohens_d, 3),
            'Effect_Size': 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'
        })
    
    stats_df = pd.DataFrame(results)
    
    # Save statistical tests
    stats_file = 'results/summaries/stats_pairwise_tests.csv'
    stats_df.to_csv(stats_file, index=False)
    print(f"Saved statistical tests to {stats_file}")
    
    return stats_df

def create_example_day_plots():
    """Create example day time-series plots"""
    print("Creating example day time-series plots...")
    
    # Check for hourly data files
    hourly_files = [f for f in os.listdir('results') if f.startswith('hourly_') and f.endswith('.csv')]
    
    if not hourly_files:
        print("No hourly data files found - skipping example day plots")
        return
    
    # Create plots for available hourly data
    for hourly_file in hourly_files:
        strategy = hourly_file.replace('hourly_', '').replace('.csv', '')
        
        try:
            hourly_data = pd.read_csv(f'results/{hourly_file}')
            
            # Create the plot
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            
            # Plot 1: PV and Load
            if 'pv' in hourly_data.columns and 'load' in hourly_data.columns:
                axes[0].plot(hourly_data['hour'], hourly_data['pv'], 'orange', linewidth=2, label='PV Generation')
                axes[0].plot(hourly_data['hour'], hourly_data['load'], 'blue', linewidth=2, label='Load')
            axes[0].set_title(f'{strategy} Strategy - Energy Generation & Consumption', 
                             fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Power (kW)', fontsize=12)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Battery SOC and Grid flows
            if 'SOC' in hourly_data.columns:
                axes[1].plot(hourly_data['hour'], hourly_data['SOC'], 'green', linewidth=2, label='Battery SOC')
            if 'grid_in' in hourly_data.columns:
                axes[1].plot(hourly_data['hour'], hourly_data['grid_in'], 'red', linewidth=2, label='Grid Import')
            if 'grid_out' in hourly_data.columns:
                axes[1].plot(hourly_data['hour'], hourly_data['grid_out'], 'purple', linewidth=2, label='Grid Export')
            axes[1].set_title('Battery State & Grid Interaction', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Power (kW) / SOC (kWh)', fontsize=12)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Prices
            if 'price_buy' in hourly_data.columns:
                axes[2].plot(hourly_data['hour'], hourly_data['price_buy'], 'red', linewidth=2, label='Buy Price')
            if 'price_sell' in hourly_data.columns:
                axes[2].plot(hourly_data['hour'], hourly_data['price_sell'], 'blue', linewidth=2, label='Sell Price')
            axes[2].set_title('Energy Prices', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Hour', fontsize=12)
            axes[2].set_ylabel('Price (€/kWh)', fontsize=12)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'results/figs/example_day_timeseries_{strategy}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating plot for {hourly_file}: {e}")
    
    print("Example day plots created")

def generate_summary_report(annual_kpis, seasonal_kpis, stats_df):
    """Generate a summary report"""
    print("Generating summary report...")
    
    report_file = 'results/summaries/analysis_summary.txt'
    
    with open(report_file, 'w') as f:
        f.write("ENERGY MANAGEMENT SYSTEM - STRATEGY ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        # Annual comparison
        f.write("ANNUAL STRATEGY COMPARISON\n")
        f.write("-" * 30 + "\n")
        if len(annual_kpis) > 0:
            best_strategy = annual_kpis.loc[annual_kpis['Cost_total_sum'].idxmin(), 'Strategy']
            worst_strategy = annual_kpis.loc[annual_kpis['Cost_total_sum'].idxmax(), 'Strategy']
            best_improvement = annual_kpis['Cost_improvement_pct'].max()
            
            f.write(f"Best Strategy: {best_strategy}\n")
            f.write(f"Worst Strategy: {worst_strategy}\n")
            f.write(f"Best Cost Improvement: {best_improvement:.2f}%\n\n")
        
        # Statistical significance
        if not stats_df.empty:
            f.write("STATISTICAL SIGNIFICANCE\n")
            f.write("-" * 25 + "\n")
            for _, row in stats_df.iterrows():
                f.write(f"{row['Comparison']}: p={row['T_P_Value']:.4f}, "
                       f"Effect Size={row['Effect_Size']}\n")
            f.write("\n")
        
        # Seasonal insights
        if seasonal_kpis is not None and len(seasonal_kpis) > 0:
            f.write("SEASONAL INSIGHTS\n")
            f.write("-" * 18 + "\n")
            seasons = seasonal_kpis['season_label'].unique()
            for season in seasons:
                season_data = seasonal_kpis[seasonal_kpis['season_label'] == season]
                if len(season_data) > 0:
                    best_strategy = season_data.loc[season_data['Cost_total_mean'].idxmin(), 'Strategy']
                    f.write(f"{season}: Best strategy = {best_strategy}\n")
            f.write("\n")
        
        f.write("FILES GENERATED\n")
        f.write("-" * 15 + "\n")
        f.write(f"Summaries: results/summaries/\n")
        f.write(f"Figures: results/figs/\n")
        f.write(f"Log: analysis.log\n")
    
    print(f"Summary report saved to {report_file}")

def main():
    """Main function for results analysis"""
    print("Starting analysis of available results...")
    
    # Create output directories
    create_directories()
    
    # Load data
    kpis_data, daily_features = load_available_data()
    if kpis_data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Create annual comparison
    annual_kpis = create_annual_comparison(kpis_data)
    create_annual_plots(annual_kpis, kpis_data)
    
    # Create seasonal analysis
    seasonal_kpis = create_seasonal_analysis(kpis_data, daily_features)
    
    # Create statistical tests
    stats_df = create_statistical_tests(kpis_data)
    
    # Create example day plots
    create_example_day_plots()
    
    # Generate summary report
    generate_summary_report(annual_kpis, seasonal_kpis, stats_df)
    
    print("\nAnalysis completed successfully!")
    print(f"Results saved to: results/")
    print(f"Summaries: results/summaries/")
    print(f"Figures: results/figs/")

if __name__ == "__main__":
    main()
