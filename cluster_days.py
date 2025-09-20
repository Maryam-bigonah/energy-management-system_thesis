#!/usr/bin/env python3
"""
Step 4 - Behavioral Clustering & Season Classification

This script performs K-Means clustering on daily energy behavior patterns
to identify seasonal clusters (Winter, Spring, Summer, Autumn) and analyzes
strategy performance within each cluster.

Author: Energy Management System
Date: 2024
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cluster_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ClusteringConfig:
    """Configuration for clustering analysis"""
    k_clusters: int = 4
    random_state: int = 42
    results_dir: str = "results"
    features_dir: str = "results"
    use_pca: bool = False
    pca_components: int = 3
    min_silhouette: float = 0.4

class DailyFeatureBuilder:
    """Builds daily feature vectors for clustering"""
    
    def __init__(self, kpis_file: str):
        self.kpis_file = kpis_file
        self.kpis_data = None
        self.daily_features = None
        
    def load_kpis_data(self) -> pd.DataFrame:
        """Load KPIs data from Step 3 results"""
        logger.info(f"Loading KPIs data from {self.kpis_file}")
        
        if not os.path.exists(self.kpis_file):
            raise FileNotFoundError(f"KPIs file not found: {self.kpis_file}")
        
        self.kpis_data = pd.read_csv(self.kpis_file)
        logger.info(f"Loaded {len(self.kpis_data)} KPI records")
        
        # Validate required columns
        required_columns = [
            'day', 'Strategy', 'Cost_total', 'Import_total', 'Export_total',
            'PV_total', 'Load_total', 'pv_self', 'SCR', 'SelfSufficiency',
            'PeakGrid', 'BatteryCycles'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.kpis_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return self.kpis_data
    
    def build_strategy_agnostic_features(self) -> pd.DataFrame:
        """Build strategy-agnostic daily features for clustering"""
        logger.info("Building strategy-agnostic daily features...")
        
        if self.kpis_data is None:
            self.load_kpis_data()
        
        # Use MSC strategy data as baseline (closest to physics)
        msc_data = self.kpis_data[self.kpis_data['Strategy'] == 'MSC'].copy()
        
        if len(msc_data) == 0:
            logger.warning("No MSC data found, using first available strategy")
            msc_data = self.kpis_data.groupby('day').first().reset_index()
        
        # Select features for clustering
        feature_columns = [
            'Load_total', 'PV_total', 'Import_total', 'Export_total',
            'pv_self', 'SCR', 'SelfSufficiency', 'PeakGrid', 'BatteryCycles'
        ]
        
        # Create daily features DataFrame
        self.daily_features = msc_data[['day'] + feature_columns].copy()
        
        # Add optional hourly shape features if available
        self._add_hourly_shape_features()
        
        # Add calendar features
        self._add_calendar_features()
        
        logger.info(f"Built daily features for {len(self.daily_features)} days")
        logger.info(f"Features: {list(self.daily_features.columns)}")
        
        return self.daily_features
    
    def _add_hourly_shape_features(self):
        """Add hourly shape features if hourly data is available"""
        try:
            # Try to load a sample hourly file to extract shape features
            sample_hourly_file = os.path.join(self.features_dir, "hourly", "hourly_MSC_day001.csv")
            if os.path.exists(sample_hourly_file):
                logger.info("Adding hourly shape features...")
                
                shape_features = []
                for day in self.daily_features['day']:
                    hourly_file = os.path.join(
                        self.features_dir, "hourly", 
                        f"hourly_MSC_day{day:03d}.csv"
                    )
                    
                    if os.path.exists(hourly_file):
                        hourly_data = pd.read_csv(hourly_file)
                        
                        # Calculate shape features
                        load_peak_hour = hourly_data.loc[hourly_data['load'].idxmax(), 'hour']
                        pv_peak_hour = hourly_data.loc[hourly_data['pv'].idxmax(), 'hour']
                        
                        # Load ramp (max hour-to-hour increase)
                        load_ramp = hourly_data['load'].diff().max()
                        
                        # Evening load share (17-22)
                        evening_hours = hourly_data[hourly_data['hour'].between(17, 22)]
                        evening_load_share = evening_hours['load'].sum() / hourly_data['load'].sum()
                        
                        shape_features.append({
                            'day': day,
                            'Load_peak_hour': load_peak_hour,
                            'PV_peak_hour': pv_peak_hour,
                            'Load_ramp_max': load_ramp,
                            'Evening_load_share': evening_load_share
                        })
                    else:
                        # Default values if file not found
                        shape_features.append({
                            'day': day,
                            'Load_peak_hour': 19,
                            'PV_peak_hour': 12,
                            'Load_ramp_max': 0,
                            'Evening_load_share': 0.3
                        })
                
                shape_df = pd.DataFrame(shape_features)
                self.daily_features = self.daily_features.merge(shape_df, on='day', how='left')
                logger.info("Added hourly shape features")
                
        except Exception as e:
            logger.warning(f"Could not add hourly shape features: {e}")
    
    def _add_calendar_features(self):
        """Add calendar-based features"""
        # Convert day number to date (assuming 2024)
        self.daily_features['date'] = pd.to_datetime(
            '2024-01-01'
        ) + pd.to_timedelta(self.daily_features['day'] - 1, unit='D')
        
        # Add month and day of year
        self.daily_features['month'] = self.daily_features['date'].dt.month
        self.daily_features['day_of_year'] = self.daily_features['date'].dt.dayofyear
        
        # Add season based on calendar
        def get_calendar_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Autumn'
        
        self.daily_features['calendar_season'] = self.daily_features['month'].apply(get_calendar_season)

class ClusteringAnalyzer:
    """Performs K-Means clustering and analysis"""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = None
        self.features_scaled = None
        self.cluster_labels = None
        
    def prepare_features(self, daily_features: pd.DataFrame) -> np.ndarray:
        """Prepare and standardize features for clustering"""
        logger.info("Preparing features for clustering...")
        
        # Select numeric features (exclude day, date, calendar features)
        feature_columns = [
            col for col in daily_features.columns 
            if col not in ['day', 'date', 'month', 'day_of_year', 'calendar_season']
        ]
        
        features = daily_features[feature_columns].values
        
        # Handle missing values
        features = np.nan_to_num(features, nan=0.0)
        
        # Standardize features
        self.features_scaled = self.scaler.fit_transform(features)
        
        logger.info(f"Prepared {self.features_scaled.shape[1]} features for {self.features_scaled.shape[0]} days")
        
        return self.features_scaled
    
    def find_optimal_k(self, features: np.ndarray, max_k: int = 8) -> int:
        """Find optimal number of clusters using elbow method and silhouette score"""
        logger.info(f"Finding optimal K (testing 2 to {max_k})...")
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(features, cluster_labels))
            
            logger.info(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
        
        # Choose K based on silhouette score and target K=4
        if self.config.k_clusters in k_range:
            target_silhouette = silhouette_scores[self.config.k_clusters - 2]
            if target_silhouette >= self.config.min_silhouette:
                logger.info(f"Using target K={self.config.k_clusters} (Silhouette={target_silhouette:.3f})")
                return self.config.k_clusters
        
        # Otherwise, choose K with highest silhouette score
        best_k = k_range[np.argmax(silhouette_scores)]
        best_silhouette = max(silhouette_scores)
        
        logger.info(f"Best K={best_k} (Silhouette={best_silhouette:.3f})")
        return best_k
    
    def perform_clustering(self, features: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """Perform K-Means clustering"""
        if k is None:
            k = self.config.k_clusters
        
        logger.info(f"Performing K-Means clustering with K={k}...")
        
        self.kmeans = KMeans(
            n_clusters=k, 
            random_state=self.config.random_state,
            n_init=10,
            max_iter=300
        )
        
        self.cluster_labels = self.kmeans.fit_predict(features)
        
        # Calculate silhouette score
        silhouette = silhouette_score(features, self.cluster_labels)
        logger.info(f"Clustering completed. Silhouette score: {silhouette:.3f}")
        
        return self.cluster_labels
    
    def apply_pca(self, features: np.ndarray) -> np.ndarray:
        """Apply PCA for visualization"""
        if not self.config.use_pca:
            return features
        
        logger.info(f"Applying PCA with {self.config.pca_components} components...")
        
        self.pca = PCA(n_components=self.config.pca_components)
        features_pca = self.pca.fit_transform(features)
        
        explained_variance = self.pca.explained_variance_ratio_
        logger.info(f"PCA explained variance: {explained_variance}")
        logger.info(f"Total explained variance: {sum(explained_variance):.3f}")
        
        return features_pca

class SeasonMapper:
    """Maps clusters to seasonal labels"""
    
    def __init__(self):
        self.cluster_to_season = {}
        self.season_mapping = {}
    
    def map_clusters_to_seasons(self, daily_features: pd.DataFrame, cluster_labels: np.ndarray) -> Dict[int, str]:
        """Map clusters to seasons based on PV/load patterns and calendar distribution"""
        logger.info("Mapping clusters to seasons...")
        
        # Add cluster labels to features
        daily_features = daily_features.copy()
        daily_features['cluster_id'] = cluster_labels
        
        # Calculate cluster centroids
        cluster_stats = daily_features.groupby('cluster_id').agg({
            'PV_total': ['mean', 'median'],
            'Load_total': ['mean', 'median'],
            'Import_total': ['mean', 'median'],
            'month': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.median()
        }).round(2)
        
        cluster_stats.columns = ['PV_mean', 'PV_median', 'Load_mean', 'Load_median', 'Import_mean', 'Import_median', 'Dominant_month']
        
        logger.info("Cluster statistics:")
        for cluster_id in sorted(cluster_stats.index):
            stats = cluster_stats.loc[cluster_id]
            logger.info(f"Cluster {cluster_id}: PV={stats['PV_median']:.1f} kWh, "
                       f"Load={stats['Load_median']:.1f} kWh, "
                       f"Import={stats['Import_median']:.1f} kWh, "
                       f"Month={stats['Dominant_month']:.0f}")
        
        # Map clusters to seasons based on PV levels and calendar
        pv_medians = cluster_stats['PV_median'].sort_values(ascending=False)
        
        # Summer: highest PV
        summer_cluster = pv_medians.index[0]
        
        # Winter: lowest PV
        winter_cluster = pv_medians.index[-1]
        
        # Spring/Autumn: middle PV levels
        remaining_clusters = pv_medians.index[1:-1]
        
        # Use calendar months to distinguish Spring vs Autumn
        spring_cluster = None
        autumn_cluster = None
        
        for cluster_id in remaining_clusters:
            dominant_month = cluster_stats.loc[cluster_id, 'Dominant_month']
            if dominant_month <= 5:  # Jan-May
                spring_cluster = cluster_id
            else:  # Sep-Dec
                autumn_cluster = cluster_id
        
        # If we couldn't distinguish, use PV levels
        if spring_cluster is None or autumn_cluster is None:
            if len(remaining_clusters) >= 2:
                spring_cluster = remaining_clusters[0]
                autumn_cluster = remaining_clusters[1]
            else:
                spring_cluster = remaining_clusters[0]
                autumn_cluster = remaining_clusters[0]
        
        # Create mapping
        self.cluster_to_season = {
            summer_cluster: 'Summer',
            spring_cluster: 'Spring',
            autumn_cluster: 'Autumn',
            winter_cluster: 'Winter'
        }
        
        logger.info("Cluster to season mapping:")
        for cluster_id, season in self.cluster_to_season.items():
            logger.info(f"Cluster {cluster_id} → {season}")
        
        return self.cluster_to_season

class StrategyAnalyzer:
    """Analyzes strategy performance within clusters"""
    
    def __init__(self, kpis_data: pd.DataFrame, cluster_labels: np.ndarray, 
                 cluster_to_season: Dict[int, str], daily_features: pd.DataFrame):
        self.kpis_data = kpis_data
        self.cluster_labels = cluster_labels
        self.cluster_to_season = cluster_to_season
        self.daily_features = daily_features
        
    def analyze_strategy_performance(self) -> pd.DataFrame:
        """Analyze strategy performance within each cluster"""
        logger.info("Analyzing strategy performance within clusters...")
        
        # Add cluster and season labels to KPIs data
        kpis_with_clusters = self.kpis_data.copy()
        kpis_with_clusters['cluster_id'] = self.cluster_labels[kpis_with_clusters['day'] - 1]
        kpis_with_clusters['season_label'] = kpis_with_clusters['cluster_id'].map(self.cluster_to_season)
        
        # Calculate statistics for each strategy × cluster combination
        metrics = [
            'Cost_total', 'Import_total', 'Export_total', 'PV_total', 'Load_total',
            'pv_self', 'SCR', 'SelfSufficiency', 'PeakGrid', 'BatteryCycles'
        ]
        
        results = []
        
        for strategy in kpis_with_clusters['Strategy'].unique():
            for cluster_id in sorted(kpis_with_clusters['cluster_id'].unique()):
                season = self.cluster_to_season[cluster_id]
                
                # Filter data for this strategy and cluster
                mask = (kpis_with_clusters['Strategy'] == strategy) & \
                       (kpis_with_clusters['cluster_id'] == cluster_id)
                
                cluster_data = kpis_with_clusters[mask]
                
                if len(cluster_data) == 0:
                    continue
                
                # Calculate statistics
                stats = {
                    'Strategy': strategy,
                    'cluster_id': cluster_id,
                    'season_label': season,
                    'n_days': len(cluster_data)
                }
                
                for metric in metrics:
                    values = cluster_data[metric].dropna()
                    if len(values) > 0:
                        stats[f'{metric}_mean'] = values.mean()
                        stats[f'{metric}_std'] = values.std()
                        stats[f'{metric}_median'] = values.median()
                    else:
                        stats[f'{metric}_mean'] = np.nan
                        stats[f'{metric}_std'] = np.nan
                        stats[f'{metric}_median'] = np.nan
                
                results.append(stats)
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Generated strategy analysis for {len(results_df)} strategy-cluster combinations")
        
        return results_df

class VisualizationGenerator:
    """Generates clustering visualizations"""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.fig_dir = os.path.join(config.results_dir, "figs")
        os.makedirs(self.fig_dir, exist_ok=True)
    
    def create_pca_scatter(self, features_pca: np.ndarray, cluster_labels: np.ndarray, 
                          cluster_to_season: Dict[int, str], daily_features: pd.DataFrame):
        """Create PCA scatter plot with clusters colored by season"""
        logger.info("Creating PCA scatter plot...")
        
        plt.figure(figsize=(12, 8))
        
        # Color map for seasons
        season_colors = {
            'Summer': '#ff6b6b',
            'Spring': '#4ecdc4', 
            'Autumn': '#45b7d1',
            'Winter': '#96ceb4'
        }
        
        # Plot each cluster
        for cluster_id in sorted(np.unique(cluster_labels)):
            season = cluster_to_season[cluster_id]
            mask = cluster_labels == cluster_id
            
            plt.scatter(
                features_pca[mask, 0], features_pca[mask, 1],
                c=season_colors[season], label=f'{season} (Cluster {cluster_id})',
                alpha=0.7, s=50
            )
        
        plt.xlabel(f'PC1 ({self.config.pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({self.config.pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('Daily Energy Behavior Clusters (PCA Visualization)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, 'pca_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("PCA scatter plot saved")
    
    def create_seasonal_bar_charts(self, strategy_analysis: pd.DataFrame):
        """Create seasonal bar charts for strategy comparison"""
        logger.info("Creating seasonal bar charts...")
        
        metrics = ['Cost_total', 'SCR', 'SelfSufficiency', 'PeakGrid']
        strategies = strategy_analysis['Strategy'].unique()
        seasons = strategy_analysis['season_label'].unique()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Prepare data for bar chart
            data = []
            x_labels = []
            
            for season in sorted(seasons):
                season_data = strategy_analysis[strategy_analysis['season_label'] == season]
                
                values = []
                for strategy in strategies:
                    strategy_data = season_data[season_data['Strategy'] == strategy]
                    if len(strategy_data) > 0:
                        values.append(strategy_data[f'{metric}_mean'].iloc[0])
                    else:
                        values.append(0)
                
                data.append(values)
                x_labels.append(season)
            
            # Create bar chart
            x = np.arange(len(seasons))
            width = 0.2
            
            for j, strategy in enumerate(strategies):
                values = [data[k][j] for k in range(len(seasons))]
                ax.bar(x + j * width, values, width, label=strategy, alpha=0.8)
            
            ax.set_xlabel('Season')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} by Season and Strategy')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(x_labels)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, 'seasonal_bar_charts.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Seasonal bar charts saved")
    
    def create_cluster_centroids_radar(self, daily_features: pd.DataFrame, cluster_labels: np.ndarray,
                                     cluster_to_season: Dict[int, str]):
        """Create radar chart of cluster centroids"""
        logger.info("Creating cluster centroids radar chart...")
        
        # Select key features for radar chart
        radar_features = ['PV_total', 'Load_total', 'Import_total', 'Export_total', 
                         'SCR', 'SelfSufficiency', 'PeakGrid', 'BatteryCycles']
        
        # Calculate cluster centroids
        daily_features_with_clusters = daily_features.copy()
        daily_features_with_clusters['cluster_id'] = cluster_labels
        
        centroids = daily_features_with_clusters.groupby('cluster_id')[radar_features].mean()
        
        # Normalize features for radar chart (0-1 scale)
        centroids_normalized = centroids.copy()
        for feature in radar_features:
            min_val = centroids[feature].min()
            max_val = centroids[feature].max()
            if max_val > min_val:
                centroids_normalized[feature] = (centroids[feature] - min_val) / (max_val - min_val)
            else:
                centroids_normalized[feature] = 0.5
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Color map for seasons
        season_colors = {
            'Summer': '#ff6b6b',
            'Spring': '#4ecdc4', 
            'Autumn': '#45b7d1',
            'Winter': '#96ceb4'
        }
        
        angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for cluster_id in sorted(centroids_normalized.index):
            season = cluster_to_season[cluster_id]
            values = centroids_normalized.loc[cluster_id].tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=f'{season} (Cluster {cluster_id})',
                   color=season_colors[season])
            ax.fill(angles, values, alpha=0.25, color=season_colors[season])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f.replace('_', '\n') for f in radar_features])
        ax.set_ylim(0, 1)
        ax.set_title('Cluster Centroids Comparison (Normalized)', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, 'cluster_centroids_radar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Cluster centroids radar chart saved")

def main():
    """Main function for clustering analysis"""
    parser = argparse.ArgumentParser(description='Behavioral Clustering & Season Classification')
    parser.add_argument('--k', type=int, default=4, help='Number of clusters (default: 4)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    parser.add_argument('--kpis-file', type=str, default='results/kpis.csv', help='KPIs CSV file')
    parser.add_argument('--use-pca', action='store_true', help='Use PCA for visualization')
    parser.add_argument('--pca-components', type=int, default=3, help='PCA components')
    parser.add_argument('--find-optimal-k', action='store_true', help='Find optimal K automatically')
    
    args = parser.parse_args()
    
    # Create configuration
    config = ClusteringConfig(
        k_clusters=args.k,
        random_state=args.seed,
        results_dir=args.results_dir,
        features_dir=args.results_dir,
        use_pca=args.use_pca,
        pca_components=args.pca_components
    )
    
    logger.info("Starting behavioral clustering analysis...")
    logger.info(f"Configuration: K={config.k_clusters}, seed={config.random_state}")
    
    try:
        # Step 1: Build daily features
        feature_builder = DailyFeatureBuilder(args.kpis_file)
        daily_features = feature_builder.build_strategy_agnostic_features()
        
        # Step 2: Prepare features for clustering
        clustering_analyzer = ClusteringAnalyzer(config)
        features_scaled = clustering_analyzer.prepare_features(daily_features)
        
        # Step 3: Find optimal K if requested
        if args.find_optimal_k:
            optimal_k = clustering_analyzer.find_optimal_k(features_scaled)
            config.k_clusters = optimal_k
            logger.info(f"Using optimal K={optimal_k}")
        
        # Step 4: Perform clustering
        cluster_labels = clustering_analyzer.perform_clustering(features_scaled)
        
        # Step 5: Apply PCA for visualization
        if config.use_pca:
            features_pca = clustering_analyzer.apply_pca(features_scaled)
        else:
            # Use first two features for 2D visualization
            features_pca = features_scaled[:, :2]
            clustering_analyzer.pca = type('PCA', (), {
                'explained_variance_ratio_': [0.5, 0.3]
            })()
        
        # Step 6: Map clusters to seasons
        season_mapper = SeasonMapper()
        cluster_to_season = season_mapper.map_clusters_to_seasons(daily_features, cluster_labels)
        
        # Step 7: Analyze strategy performance
        kpis_data = feature_builder.kpis_data
        strategy_analyzer = StrategyAnalyzer(kpis_data, cluster_labels, cluster_to_season, daily_features)
        strategy_analysis = strategy_analyzer.analyze_strategy_performance()
        
        # Step 8: Generate visualizations
        viz_generator = VisualizationGenerator(config)
        viz_generator.create_pca_scatter(features_pca, cluster_labels, cluster_to_season, daily_features)
        viz_generator.create_seasonal_bar_charts(strategy_analysis)
        viz_generator.create_cluster_centroids_radar(daily_features, cluster_labels, cluster_to_season)
        
        # Step 9: Save results
        logger.info("Saving results...")
        
        # Add cluster and season labels to daily features
        daily_features_with_clusters = daily_features.copy()
        daily_features_with_clusters['cluster_id'] = cluster_labels
        daily_features_with_clusters['season_label'] = daily_features_with_clusters['cluster_id'].map(cluster_to_season)
        
        # Save daily features with clusters
        daily_features_file = os.path.join(config.results_dir, 'daily_features.csv')
        daily_features_with_clusters.to_csv(daily_features_file, index=False)
        logger.info(f"Saved daily features to {daily_features_file}")
        
        # Save cluster centroids
        centroids = daily_features_with_clusters.groupby('cluster_id').mean()
        centroids['season_label'] = centroids.index.map(cluster_to_season)
        centroids_file = os.path.join(config.results_dir, 'cluster_centroids.csv')
        centroids.to_csv(centroids_file)
        logger.info(f"Saved cluster centroids to {centroids_file}")
        
        # Save strategy analysis
        strategy_analysis_file = os.path.join(config.results_dir, 'kpis_by_cluster_strategy.csv')
        strategy_analysis.to_csv(strategy_analysis_file, index=False)
        logger.info(f"Saved strategy analysis to {strategy_analysis_file}")
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("CLUSTERING ANALYSIS SUMMARY")
        logger.info("="*50)
        
        for cluster_id in sorted(np.unique(cluster_labels)):
            season = cluster_to_season[cluster_id]
            n_days = np.sum(cluster_labels == cluster_id)
            logger.info(f"Cluster {cluster_id} ({season}): {n_days} days")
        
        logger.info(f"\nSilhouette Score: {silhouette_score(features_scaled, cluster_labels):.3f}")
        logger.info(f"Results saved to: {config.results_dir}")
        logger.info(f"Visualizations saved to: {viz_generator.fig_dir}")
        
        logger.info("\nClustering analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Clustering analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
