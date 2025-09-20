#!/usr/bin/env python3
"""
Step 5 - Strategy Comparison, Visualization & Sensitivity Analysis

This script analyzes the results from Steps 3 and 4 to create thesis-ready
figures, tables, and statistical conclusions comparing the four energy
management strategies (MSC, TOU, MMR-P2P, DR-P2P).

Author: Energy Management System
Date: 2024
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Configuration for results analysis"""
    results_dir: str = "results"
    summaries_dir: str = "results/summaries"
    figs_dir: str = "results/figs"
    include_example_days: bool = True
    skip_stats: bool = False
    skip_seasonal: bool = False
    fig_dpi: int = 300
    fig_size: Tuple[int, int] = (10, 6)

class ResultsAnalyzer:
    """Main class for analyzing optimization results"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.kpis_data = None
        self.daily_features = None
        self.strategies = ['MSC', 'TOU', 'MMR', 'DRP2P']
        self.seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
        
        # Create output directories
        os.makedirs(self.config.summaries_dir, exist_ok=True)
        os.makedirs(self.config.figs_dir, exist_ok=True)
        
        # Set matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = self.config.fig_size
        plt.rcParams['figure.dpi'] = self.config.fig_dpi
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def load_data(self) -> bool:
        """Load KPIs and daily features data"""
        logger.info("Loading analysis data...")
        
        # Load KPIs data
        kpis_file = os.path.join(self.config.results_dir, 'kpis.csv')
        if not os.path.exists(kpis_file):
            logger.error(f"KPIs file not found: {kpis_file}")
            return False
        
        self.kpis_data = pd.read_csv(kpis_file)
        logger.info(f"Loaded {len(self.kpis_data)} KPI records")
        
        # Load daily features with season labels
        features_file = os.path.join(self.config.results_dir, 'daily_features.csv')
        if not os.path.exists(features_file):
            logger.warning(f"Daily features file not found: {features_file}")
            logger.info("Proceeding without seasonal analysis...")
            self.daily_features = None
        else:
            self.daily_features = pd.read_csv(features_file)
            logger.info(f"Loaded {len(self.daily_features)} daily feature records")
        
        return True
    
    def create_annual_comparison(self) -> pd.DataFrame:
        """Create annual strategy comparison"""
        logger.info("Creating annual strategy comparison...")
        
        # Aggregate by strategy
        annual_kpis = self.kpis_data.groupby('Strategy').agg({
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
        annual_file = os.path.join(self.config.summaries_dir, 'annual_kpis_by_strategy.csv')
        annual_kpis.to_csv(annual_file, index=False)
        logger.info(f"Saved annual comparison to {annual_file}")
        
        return annual_kpis
    
    def create_annual_plots(self, annual_kpis: pd.DataFrame):
        """Create annual comparison plots"""
        logger.info("Creating annual comparison plots...")
        
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
        plt.savefig(os.path.join(self.config.figs_dir, 'annual_cost_bars.png'), 
                   dpi=self.config.fig_dpi, bbox_inches='tight')
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
        plt.savefig(os.path.join(self.config.figs_dir, 'annual_scr_selfsuff_bars.png'), 
                   dpi=self.config.fig_dpi, bbox_inches='tight')
        plt.close()
        
        # 3. Daily cost boxplots
        plt.figure(figsize=(12, 8))
        
        # Prepare data for boxplot
        daily_costs = []
        strategy_labels = []
        
        for strategy in self.strategies:
            strategy_data = self.kpis_data[self.kpis_data['Strategy'] == strategy]['Cost_total']
            daily_costs.append(strategy_data)
            strategy_labels.append(strategy)
        
        box_plot = plt.boxplot(daily_costs, labels=strategy_labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        plt.title('Daily Cost Distribution by Strategy', fontsize=16, fontweight='bold')
        plt.xlabel('Strategy', fontsize=14)
        plt.ylabel('Daily Cost (€)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.figs_dir, 'daily_cost_boxplots.png'), 
                   dpi=self.config.fig_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info("Annual comparison plots created")
    
    def create_seasonal_analysis(self) -> Optional[pd.DataFrame]:
        """Create seasonal analysis by cluster/season"""
        if self.daily_features is None:
            logger.warning("Skipping seasonal analysis - no daily features data")
            return None
        
        if self.config.skip_seasonal:
            logger.info("Skipping seasonal analysis as requested")
            return None
        
        logger.info("Creating seasonal analysis...")
        
        # Merge KPIs with daily features to get season labels
        merged_data = self.kpis_data.merge(
            self.daily_features[['day', 'season_label']], 
            on='day', 
            how='left'
        )
        
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
        seasonal_file = os.path.join(self.config.summaries_dir, 'seasonal_kpis_by_strategy.csv')
        seasonal_kpis.to_csv(seasonal_file, index=False)
        logger.info(f"Saved seasonal analysis to {seasonal_file}")
        
        # Create seasonal plots
        self.create_seasonal_plots(seasonal_kpis)
        
        return seasonal_kpis
    
    def create_seasonal_plots(self, seasonal_kpis: pd.DataFrame):
        """Create seasonal comparison plots"""
        logger.info("Creating seasonal comparison plots...")
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for season in self.seasons:
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
            plt.savefig(os.path.join(self.config.figs_dir, f'seasonal_cost_bars_{season}.png'), 
                       dpi=self.config.fig_dpi, bbox_inches='tight')
            plt.close()
        
        logger.info("Seasonal comparison plots created")
    
    def create_statistical_tests(self) -> pd.DataFrame:
        """Create statistical validation tests"""
        if self.config.skip_stats:
            logger.info("Skipping statistical tests as requested")
            return pd.DataFrame()
        
        logger.info("Creating statistical validation tests...")
        
        # Prepare data for paired tests
        strategy_data = {}
        for strategy in self.strategies:
            strategy_data[strategy] = self.kpis_data[
                self.kpis_data['Strategy'] == strategy
            ]['Cost_total'].values
        
        # Ensure all strategies have the same number of observations
        min_length = min(len(data) for data in strategy_data.values())
        for strategy in strategy_data:
            strategy_data[strategy] = strategy_data[strategy][:min_length]
        
        # Perform pairwise comparisons
        results = []
        baseline_strategy = 'MSC'
        
        for strategy in self.strategies:
            if strategy == baseline_strategy:
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
        stats_file = os.path.join(self.config.summaries_dir, 'stats_pairwise_tests.csv')
        stats_df.to_csv(stats_file, index=False)
        logger.info(f"Saved statistical tests to {stats_file}")
        
        return stats_df
    
    def create_example_days_plots(self):
        """Create example day time-series plots per season"""
        if not self.config.include_example_days:
            logger.info("Skipping example days plots as requested")
            return
        
        if self.daily_features is None:
            logger.warning("Skipping example days - no daily features data")
            return
        
        logger.info("Creating example day time-series plots...")
        
        # Find representative days (closest to cluster centroids)
        representative_days = self.find_representative_days()
        
        for season, day in representative_days.items():
            logger.info(f"Creating example day plots for {season} (Day {day})")
            
            for strategy in self.strategies:
                self.create_single_day_plot(day, strategy, season)
        
        logger.info("Example day plots created")
    
    def find_representative_days(self) -> Dict[str, int]:
        """Find representative days closest to cluster centroids"""
        representative_days = {}
        
        for season in self.seasons:
            season_data = self.daily_features[self.daily_features['season_label'] == season]
            
            if len(season_data) == 0:
                continue
            
            # Calculate centroid
            numeric_cols = season_data.select_dtypes(include=[np.number]).columns
            centroid = season_data[numeric_cols].mean()
            
            # Find closest day
            distances = []
            for idx, row in season_data.iterrows():
                day_data = row[numeric_cols]
                distance = np.sqrt(np.sum((day_data - centroid) ** 2))
                distances.append((row['day'], distance))
            
            # Get day with minimum distance
            closest_day = min(distances, key=lambda x: x[1])[0]
            representative_days[season] = closest_day
        
        return representative_days
    
    def create_single_day_plot(self, day: int, strategy: str, season: str):
        """Create time-series plot for a single day and strategy"""
        # Load hourly data for this day and strategy
        hourly_file = os.path.join(self.config.results_dir, 'hourly', 
                                  f'hourly_{strategy}_day{day:03d}.csv')
        
        if not os.path.exists(hourly_file):
            logger.warning(f"Hourly file not found: {hourly_file}")
            return
        
        hourly_data = pd.read_csv(hourly_file)
        
        # Create the plot
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: PV and Load
        axes[0].plot(hourly_data['hour'], hourly_data['pv'], 'orange', linewidth=2, label='PV Generation')
        axes[0].plot(hourly_data['hour'], hourly_data['load'], 'blue', linewidth=2, label='Load')
        if 'L_DR' in hourly_data.columns:
            axes[0].plot(hourly_data['hour'], hourly_data['L_DR'], 'red', linewidth=2, 
                        linestyle='--', label='Adjusted Load (DR)')
        axes[0].set_title(f'{strategy} Strategy - {season} Day {day} - Energy Generation & Consumption', 
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Power (kW)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Battery SOC and Grid flows
        axes[1].plot(hourly_data['hour'], hourly_data['SOC'], 'green', linewidth=2, label='Battery SOC')
        axes[1].plot(hourly_data['hour'], hourly_data['grid_in'], 'red', linewidth=2, label='Grid Import')
        axes[1].plot(hourly_data['hour'], hourly_data['grid_out'], 'purple', linewidth=2, label='Grid Export')
        axes[1].set_title('Battery State & Grid Interaction', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Power (kW) / SOC (kWh)', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Prices
        axes[2].plot(hourly_data['hour'], hourly_data['price_buy'], 'red', linewidth=2, label='Buy Price')
        axes[2].plot(hourly_data['hour'], hourly_data['price_sell'], 'blue', linewidth=2, label='Sell Price')
        if 'p2p_price_buy' in hourly_data.columns:
            axes[2].plot(hourly_data['hour'], hourly_data['p2p_price_buy'], 'green', linewidth=2, 
                        linestyle='--', label='P2P Buy Price')
        if 'p2p_price_sell' in hourly_data.columns:
            axes[2].plot(hourly_data['hour'], hourly_data['p2p_price_sell'], 'orange', linewidth=2, 
                        linestyle='--', label='P2P Sell Price')
        axes[2].set_title('Energy Prices', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Hour', fontsize=12)
        axes[2].set_ylabel('Price (€/kWh)', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.figs_dir, 
                                f'example_day_timeseries_{season}_{strategy}.png'), 
                   dpi=self.config.fig_dpi, bbox_inches='tight')
        plt.close()
    
    def create_sensitivity_analysis(self):
        """Create sensitivity analysis plots if data exists"""
        logger.info("Checking for sensitivity analysis data...")
        
        # Check for battery sensitivity data
        battery_files = [f for f in os.listdir(self.config.results_dir) 
                        if f.startswith('sensitivity_battery_') and f.endswith('.csv')]
        
        if battery_files:
            self.create_battery_sensitivity_plot(battery_files)
        
        # Check for PV sensitivity data
        pv_files = [f for f in os.listdir(self.config.results_dir) 
                   if f.startswith('sensitivity_pv_') and f.endswith('.csv')]
        
        if pv_files:
            self.create_pv_sensitivity_plot(pv_files)
        
        # Check for tariff sensitivity data
        tariff_files = [f for f in os.listdir(self.config.results_dir) 
                       if f.startswith('sensitivity_tariff_') and f.endswith('.csv')]
        
        if tariff_files:
            self.create_tariff_sensitivity_plot(tariff_files)
    
    def create_battery_sensitivity_plot(self, battery_files: List[str]):
        """Create battery size sensitivity plot"""
        logger.info("Creating battery sensitivity analysis...")
        
        # Load and combine battery sensitivity data
        battery_data = []
        for file in battery_files:
            df = pd.read_csv(os.path.join(self.config.results_dir, file))
            battery_data.append(df)
        
        combined_data = pd.concat(battery_data, ignore_index=True)
        
        # Create cost vs battery size plot
        plt.figure(figsize=(12, 8))
        
        for strategy in self.strategies:
            strategy_data = combined_data[combined_data['Strategy'] == strategy]
            if len(strategy_data) > 0:
                plt.plot(strategy_data['Ebat_kWh'], strategy_data['Annual_Cost'], 
                        marker='o', linewidth=2, label=strategy)
        
        plt.title('Annual Cost vs Battery Capacity', fontsize=16, fontweight='bold')
        plt.xlabel('Battery Capacity (kWh)', fontsize=14)
        plt.ylabel('Annual Cost (€)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.figs_dir, 'sensitivity_cost_vs_battery.png'), 
                   dpi=self.config.fig_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info("Battery sensitivity plot created")
    
    def create_pv_sensitivity_plot(self, pv_files: List[str]):
        """Create PV size sensitivity plot"""
        logger.info("Creating PV sensitivity analysis...")
        
        # Load and combine PV sensitivity data
        pv_data = []
        for file in pv_files:
            df = pd.read_csv(os.path.join(self.config.results_dir, file))
            pv_data.append(df)
        
        combined_data = pd.concat(pv_data, ignore_index=True)
        
        # Create cost vs PV scale plot
        plt.figure(figsize=(12, 8))
        
        for strategy in self.strategies:
            strategy_data = combined_data[combined_data['Strategy'] == strategy]
            if len(strategy_data) > 0:
                plt.plot(strategy_data['PV_Scale'], strategy_data['Annual_Cost'], 
                        marker='o', linewidth=2, label=strategy)
        
        plt.title('Annual Cost vs PV Scale Factor', fontsize=16, fontweight='bold')
        plt.xlabel('PV Scale Factor', fontsize=14)
        plt.ylabel('Annual Cost (€)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.figs_dir, 'sensitivity_cost_vs_pv.png'), 
                   dpi=self.config.fig_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info("PV sensitivity plot created")
    
    def create_tariff_sensitivity_plot(self, tariff_files: List[str]):
        """Create tariff sensitivity plot"""
        logger.info("Creating tariff sensitivity analysis...")
        
        # Load and combine tariff sensitivity data
        tariff_data = []
        for file in tariff_files:
            df = pd.read_csv(os.path.join(self.config.results_dir, file))
            tariff_data.append(df)
        
        combined_data = pd.concat(tariff_data, ignore_index=True)
        
        # Create heatmap of cost savings
        plt.figure(figsize=(12, 8))
        
        # Pivot data for heatmap
        pivot_data = combined_data.pivot_table(
            values='Cost_Savings_Pct', 
            index='Buy_Price_Change', 
            columns='Strategy', 
            aggfunc='mean'
        )
        
        im = plt.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto')
        plt.colorbar(im, label='Cost Savings (%)')
        
        plt.title('Cost Savings vs Tariff Changes', fontsize=16, fontweight='bold')
        plt.xlabel('Strategy', fontsize=14)
        plt.ylabel('Buy Price Change (%)', fontsize=14)
        plt.xticks(range(len(pivot_data.columns)), pivot_data.columns)
        plt.yticks(range(len(pivot_data.index)), pivot_data.index)
        
        # Add text annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                plt.text(j, i, f'{pivot_data.iloc[i, j]:.1f}%', 
                        ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.figs_dir, 'sensitivity_tariff_heatmap.png'), 
                   dpi=self.config.fig_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info("Tariff sensitivity plot created")
    
    def generate_summary_report(self, annual_kpis: pd.DataFrame, 
                              seasonal_kpis: Optional[pd.DataFrame],
                              stats_df: pd.DataFrame):
        """Generate a summary report"""
        logger.info("Generating summary report...")
        
        report_file = os.path.join(self.config.summaries_dir, 'analysis_summary.txt')
        
        with open(report_file, 'w') as f:
            f.write("ENERGY MANAGEMENT SYSTEM - STRATEGY ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # Annual comparison
            f.write("ANNUAL STRATEGY COMPARISON\n")
            f.write("-" * 30 + "\n")
            f.write(f"Best Strategy: {annual_kpis.loc[annual_kpis['Cost_total_sum'].idxmin(), 'Strategy']}\n")
            f.write(f"Worst Strategy: {annual_kpis.loc[annual_kpis['Cost_total_sum'].idxmax(), 'Strategy']}\n")
            f.write(f"Best Cost Improvement: {annual_kpis['Cost_improvement_pct'].max():.2f}%\n\n")
            
            # Statistical significance
            if not stats_df.empty:
                f.write("STATISTICAL SIGNIFICANCE\n")
                f.write("-" * 25 + "\n")
                for _, row in stats_df.iterrows():
                    f.write(f"{row['Comparison']}: p={row['T_P_Value']:.4f}, "
                           f"Effect Size={row['Effect_Size']}\n")
                f.write("\n")
            
            # Seasonal insights
            if seasonal_kpis is not None:
                f.write("SEASONAL INSIGHTS\n")
                f.write("-" * 18 + "\n")
                for season in self.seasons:
                    season_data = seasonal_kpis[seasonal_kpis['season_label'] == season]
                    if len(season_data) > 0:
                        best_strategy = season_data.loc[season_data['Cost_total_mean'].idxmin(), 'Strategy']
                        f.write(f"{season}: Best strategy = {best_strategy}\n")
                f.write("\n")
            
            f.write("FILES GENERATED\n")
            f.write("-" * 15 + "\n")
            f.write(f"Summaries: {self.config.summaries_dir}/\n")
            f.write(f"Figures: {self.config.figs_dir}/\n")
            f.write(f"Log: analysis.log\n")
        
        logger.info(f"Summary report saved to {report_file}")
    
    def run_analysis(self):
        """Run the complete analysis"""
        logger.info("Starting comprehensive results analysis...")
        
        # Load data
        if not self.load_data():
            logger.error("Failed to load data. Exiting.")
            return
        
        # Create annual comparison
        annual_kpis = self.create_annual_comparison()
        self.create_annual_plots(annual_kpis)
        
        # Create seasonal analysis
        seasonal_kpis = self.create_seasonal_analysis()
        
        # Create statistical tests
        stats_df = self.create_statistical_tests()
        
        # Create example days plots
        self.create_example_days_plots()
        
        # Create sensitivity analysis
        self.create_sensitivity_analysis()
        
        # Generate summary report
        self.generate_summary_report(annual_kpis, seasonal_kpis, stats_df)
        
        logger.info("Analysis completed successfully!")
        logger.info(f"Results saved to: {self.config.results_dir}")
        logger.info(f"Summaries: {self.config.summaries_dir}")
        logger.info(f"Figures: {self.config.figs_dir}")

def main():
    """Main function for results analysis"""
    parser = argparse.ArgumentParser(description='Strategy Comparison & Analysis')
    parser.add_argument('--results-dir', type=str, default='results', 
                       help='Results directory (default: results)')
    parser.add_argument('--skip-example-days', action='store_true', 
                       help='Skip example day plots')
    parser.add_argument('--skip-stats', action='store_true', 
                       help='Skip statistical tests')
    parser.add_argument('--skip-seasonal', action='store_true', 
                       help='Skip seasonal analysis')
    parser.add_argument('--fig-dpi', type=int, default=300, 
                       help='Figure DPI (default: 300)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = AnalysisConfig(
        results_dir=args.results_dir,
        summaries_dir=os.path.join(args.results_dir, 'summaries'),
        figs_dir=os.path.join(args.results_dir, 'figs'),
        include_example_days=not args.skip_example_days,
        skip_stats=args.skip_stats,
        skip_seasonal=args.skip_seasonal,
        fig_dpi=args.fig_dpi
    )
    
    # Run analysis
    analyzer = ResultsAnalyzer(config)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
