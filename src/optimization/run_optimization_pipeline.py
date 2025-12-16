"""
Run day-ahead optimization pipeline.

This script:
1. Loads forecasts (load and PV)
2. Runs optimization for each day
3. Saves results
4. Calculates KPIs
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.optimization.day_ahead_optimization import DayAheadOptimizer


def load_forecasts(forecast_dir: str = "outputs/forecasts") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load load and PV forecasts.
    
    Parameters:
    -----------
    forecast_dir : str
        Directory containing forecast files
        
    Returns:
    --------
    load_forecasts : pd.DataFrame
        Load forecasts with columns: time, total_load, load_hat_h1, ..., load_hat_h24
    pv_forecasts : pd.DataFrame
        PV forecasts with columns: time, PV_true, pv_hat_h1, ..., pv_hat_h24
    """
    forecast_path = Path(forecast_dir)
    
    # Load forecasts
    load_forecasts = pd.read_csv(
        forecast_path / "load_forecasts_2023.csv",
        parse_dates=['time'],
        index_col='time'
    )
    
    pv_forecasts = pd.read_csv(
        forecast_path / "pv_forecasts_2023.csv",
        parse_dates=['time'],
        index_col='time'
    )
    
    return load_forecasts, pv_forecasts


def run_optimization_for_day(
    optimizer: DayAheadOptimizer,
    load_forecast_24h: np.ndarray,
    pv_forecast_24h: np.ndarray,
    day_start: pd.Timestamp
) -> dict:
    """
    Run optimization for a single day (24 hours).
    
    Parameters:
    -----------
    optimizer : DayAheadOptimizer
        Optimizer instance
    load_forecast_24h : np.ndarray
        Load forecast for 24 hours (kW)
    pv_forecast_24h : np.ndarray
        PV forecast for 24 hours (kW)
    day_start : pd.Timestamp
        Start timestamp of the day
        
    Returns:
    --------
    dict
        Optimization results with timestamps
    """
    # Run optimization
    results = optimizer.optimize(load_forecast_24h, pv_forecast_24h)
    
    # Add timestamps
    hours = pd.date_range(day_start, periods=24, freq='H')
    results['time'] = hours
    results['day'] = day_start.date()
    
    return results


def calculate_kpis(
    results: dict,
    baseline_load: np.ndarray,
    pv_forecast: np.ndarray,
    grid_price_import: float,
    grid_price_export: float
) -> dict:
    """
    Calculate key performance indicators.
    
    Parameters:
    -----------
    results : dict
        Optimization results
    baseline_load : np.ndarray
        Baseline load (kW) for 24 hours
    pv_forecast : np.ndarray
        PV forecast (kW) for 24 hours
    grid_price_import : float
        Grid import price (€/kWh)
    grid_price_export : float
        Grid export price (€/kWh)
        
    Returns:
    --------
    dict
        KPIs dictionary
    """
    L_opt = results['L_opt']
    P_ch = results['P_ch']
    P_dis = results['P_dis']
    P_grid_in = results['P_grid_in']
    P_grid_out = results['P_grid_out']
    P_p2p_in = results['P_p2p_in']
    P_p2p_out = results['P_p2p_out']
    
    # Baseline cost (no optimization, no battery, no P2P)
    baseline_cost = np.sum(
        np.maximum(0, baseline_load - pv_forecast) * grid_price_import
        - np.maximum(0, pv_forecast - baseline_load) * grid_price_export
    )
    
    # Optimized cost
    optimized_cost = results['total_cost']
    
    # Economic KPIs
    cost_reduction = baseline_cost - optimized_cost
    cost_reduction_pct = (cost_reduction / baseline_cost * 100) if baseline_cost > 0 else 0
    
    # Energy KPIs
    total_pv = np.sum(pv_forecast)
    total_load = np.sum(L_opt)
    self_consumption = np.sum(np.minimum(L_opt, pv_forecast))
    self_consumption_rate = (self_consumption / total_pv * 100) if total_pv > 0 else 0
    self_sufficiency = (self_consumption / total_load * 100) if total_load > 0 else 0
    
    # Grid interaction
    grid_import_total = np.sum(P_grid_in)
    grid_export_total = np.sum(P_grid_out)
    grid_import_reduction = np.sum(baseline_load - pv_forecast) - grid_import_total
    grid_import_reduction_pct = (
        (grid_import_reduction / np.sum(baseline_load - pv_forecast) * 100)
        if np.sum(baseline_load - pv_forecast) > 0 else 0
    )
    
    # Flexibility
    peak_load_baseline = np.max(baseline_load)
    peak_load_optimized = np.max(L_opt)
    peak_load_reduction = peak_load_baseline - peak_load_optimized
    peak_load_reduction_pct = (peak_load_reduction / peak_load_baseline * 100) if peak_load_baseline > 0 else 0
    
    # Load shifting index (how much load was shifted)
    load_shift = np.sum(np.abs(L_opt - baseline_load)) / 2
    load_shift_pct = (load_shift / np.sum(baseline_load) * 100) if np.sum(baseline_load) > 0 else 0
    
    return {
        'baseline_cost': baseline_cost,
        'optimized_cost': optimized_cost,
        'cost_reduction': cost_reduction,
        'cost_reduction_pct': cost_reduction_pct,
        'self_consumption_rate': self_consumption_rate,
        'self_sufficiency_rate': self_sufficiency,
        'grid_import_total': grid_import_total,
        'grid_export_total': grid_export_total,
        'grid_import_reduction_pct': grid_import_reduction_pct,
        'peak_load_reduction_pct': peak_load_reduction_pct,
        'load_shift_pct': load_shift_pct,
        'total_pv': total_pv,
        'total_load': total_load,
    }


def main():
    """Run optimization pipeline."""
    print("=" * 80)
    print("DAY-AHEAD OPTIMIZATION PIPELINE")
    print("=" * 80)
    
    # Load forecasts
    print("\n1. Loading forecasts...")
    load_forecasts, pv_forecasts = load_forecasts()
    print(f"   Load forecasts: {len(load_forecasts)} rows")
    print(f"   PV forecasts: {len(pv_forecasts)} rows")
    
    # Initialize optimizer
    print("\n2. Initializing optimizer...")
    optimizer = DayAheadOptimizer(
        battery_capacity_kwh=50.0,
        battery_max_power_kw=25.0,
        battery_efficiency_charge=0.95,
        battery_efficiency_discharge=0.95,
        battery_soc_min=0.1,
        battery_soc_max=0.9,
        battery_soc_initial=0.5,
        dr_flexibility=0.10,
        grid_price_import=0.20,
        grid_price_export=0.10,
        p2p_price_margin=0.05
    )
    print("   Optimizer initialized with:")
    print(f"   - Battery: {optimizer.battery_capacity} kWh, {optimizer.battery_max_power} kW")
    print(f"   - DR flexibility: ±{optimizer.dr_flexibility*100}%")
    print(f"   - Grid prices: {optimizer.grid_price_import} €/kWh (import), {optimizer.grid_price_export} €/kWh (export)")
    
    # Get valid forecast rows (after lag requirement)
    load_valid = load_forecasts[load_forecasts['load_hat_h1'].notna()].copy()
    pv_valid = pv_forecasts[pv_forecasts['pv_hat_h1'].notna()].copy()
    
    # Align timestamps
    common_times = load_valid.index.intersection(pv_valid.index)
    load_valid = load_valid.loc[common_times]
    pv_valid = pv_valid.loc[common_times]
    
    print(f"\n3. Running optimization for {len(common_times)} hours...")
    print("   (Processing in daily batches)")
    
    # Group by day and optimize
    all_results = []
    all_kpis = []
    
    for day_start in pd.date_range(common_times.min(), common_times.max(), freq='D'):
        day_data = load_valid.loc[day_start:day_start + pd.Timedelta(hours=23)]
        
        if len(day_data) < 24:
            continue  # Skip incomplete days
        
        # Get forecasts for this day (use h=1 for now, can be changed to h=24)
        load_forecast_24h = day_data['load_hat_h1'].values
        pv_forecast_24h = pv_valid.loc[day_data.index, 'pv_hat_h1'].values
        
        # Run optimization
        try:
            results = run_optimization_for_day(
                optimizer,
                load_forecast_24h,
                pv_forecast_24h,
                day_start
            )
            
            # Calculate KPIs
            baseline_load = day_data['total_load'].values
            kpis = calculate_kpis(
                results,
                baseline_load,
                pv_forecast_24h,
                optimizer.grid_price_import,
                optimizer.grid_price_export
            )
            kpis['day'] = day_start.date()
            
            all_results.append(results)
            all_kpis.append(kpis)
            
            if len(all_results) % 10 == 0:
                print(f"   Processed {len(all_results)} days...")
                
        except Exception as e:
            print(f"   Warning: Optimization failed for {day_start.date()}: {e}")
            continue
    
    print(f"\n4. Optimization complete! Processed {len(all_results)} days.")
    
    # Save results
    print("\n5. Saving results...")
    output_dir = Path("outputs/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine all results
    results_df = pd.DataFrame({
        'time': [t for r in all_results for t in r['time']],
        'L_opt': [v for r in all_results for v in r['L_opt']],
        'P_ch': [v for r in all_results for v in r['P_ch']],
        'P_dis': [v for r in all_results for v in r['P_dis']],
        'SoC': [v for r in all_results for v in r['SoC']],
        'P_grid_in': [v for r in all_results for v in r['P_grid_in']],
        'P_grid_out': [v for r in all_results for v in r['P_grid_out']],
        'P_p2p_in': [v for r in all_results for v in r['P_p2p_in']],
        'P_p2p_out': [v for r in all_results for v in r['P_p2p_out']],
        'total_cost': [r['total_cost'] for r in all_results for _ in range(24)],
    })
    results_df.to_csv(output_dir / "optimization_results.csv", index=False)
    print(f"   Saved: {output_dir / 'optimization_results.csv'}")
    
    # Save KPIs
    kpis_df = pd.DataFrame(all_kpis)
    kpis_df.to_csv(output_dir / "optimization_kpis.csv", index=False)
    print(f"   Saved: {output_dir / 'optimization_kpis.csv'}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)
    print(f"Days optimized: {len(all_kpis)}")
    if len(all_kpis) > 0:
        print(f"Average cost reduction: {kpis_df['cost_reduction_pct'].mean():.2f}%")
        print(f"Average self-consumption rate: {kpis_df['self_consumption_rate'].mean():.2f}%")
        print(f"Average self-sufficiency rate: {kpis_df['self_sufficiency_rate'].mean():.2f}%")
    
    print("\n✅ Optimization pipeline complete!")


if __name__ == "__main__":
    from typing import Tuple
    main()

