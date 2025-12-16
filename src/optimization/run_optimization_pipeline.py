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
from typing import Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.optimization.day_ahead_optimization import DayAheadOptimizer


def load_forecast_files(forecast_dir: str = "outputs/forecasts") -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    load_fc = pd.read_csv(
        forecast_path / "load_forecasts_2023.csv",
        parse_dates=['time'],
        index_col='time'
    )
    
    pv_fc = pd.read_csv(
        forecast_path / "pv_forecasts_2023.csv",
        parse_dates=['time'],
        index_col='time'
    )
    
    return load_fc, pv_fc


def get_day_ahead_vectors(
    load_fc: pd.DataFrame,
    pv_fc: pd.DataFrame,
    issue_time: pd.Timestamp,
    horizon: int = 24
) -> Optional[Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]:
    """
    Build a 24-hour-ahead forecast vector for a given issue time (typically 00:00).

    We use the direct multi-step forecasts from the single issue timestamp:
      - load_hat_h1 .. load_hat_h24 correspond to issue_time+1h .. issue_time+24h
      - pv_hat_h1   .. pv_hat_h24   correspond to issue_time+1h .. issue_time+24h

    Returns:
      (load_forecast_24h, pv_forecast_24h, target_times)
    or None if not available.
    """
    if issue_time not in load_fc.index or issue_time not in pv_fc.index:
        return None

    load_row = load_fc.loc[issue_time]
    pv_row = pv_fc.loc[issue_time]

    load_vals = []
    pv_vals = []
    for h in range(1, horizon + 1):
        lc = f"load_hat_h{h}"
        pc = f"pv_hat_h{h}"
        if lc not in load_row.index or pc not in pv_row.index:
            return None
        load_vals.append(load_row[lc])
        pv_vals.append(pv_row[pc])

    load_vec = np.array(load_vals, dtype=float)
    pv_vec = np.array(pv_vals, dtype=float)

    if np.isnan(load_vec).any() or np.isnan(pv_vec).any():
        return None

    target_times = pd.date_range(issue_time + pd.Timedelta(hours=1), periods=horizon, freq='h')
    return load_vec, pv_vec, target_times


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
    grid_price_export: float,
    dr_flexibility: float = 0.10,
    battery_capacity_kwh: float = 50.0,
    battery_soc_min_frac: float = 0.10,
    battery_soc_max_frac: float = 0.90,
    dt_hours: float = 1.0
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
    # Prices are €/kWh and variables are kW at hourly resolution -> kWh = kW * dt_hours
    baseline_grid_in = np.maximum(0.0, baseline_load - pv_forecast)
    baseline_grid_out = np.maximum(0.0, pv_forecast - baseline_load)
    baseline_cost = np.sum(
        baseline_grid_in * grid_price_import * dt_hours
        - baseline_grid_out * grid_price_export * dt_hours
    )
    
    # Optimized cost
    optimized_cost = results['total_cost']
    
    # Economic KPIs
    cost_reduction = baseline_cost - optimized_cost
    cost_reduction_pct = (cost_reduction / baseline_cost * 100) if baseline_cost > 0 else 0
    
    # Energy totals (kWh)
    total_pv = float(np.sum(pv_forecast) * dt_hours)
    total_load_opt = float(np.sum(L_opt) * dt_hours)
    total_load_base = float(np.sum(baseline_load) * dt_hours)

    # Baseline PV usage (no battery, no P2P): PV used locally is min(PV, Load)
    pv_used_local_base = float(np.sum(np.minimum(baseline_load, pv_forecast) * dt_hours))
    scr_base = (pv_used_local_base / total_pv * 100.0) if total_pv > 0 else 0.0
    ssr_base = (pv_used_local_base / total_load_base * 100.0) if total_load_base > 0 else 0.0

    # Optimized PV self-consumption: PV not exported (exports leave the building)
    pv_export_total = float(np.sum((P_grid_out + P_p2p_out) * dt_hours))
    pv_used_local_opt = max(0.0, total_pv - pv_export_total)
    scr_opt = (pv_used_local_opt / total_pv * 100.0) if total_pv > 0 else 0.0
    ssr_opt = (pv_used_local_opt / total_load_opt * 100.0) if total_load_opt > 0 else 0.0
    
    # Grid interaction totals (kWh)
    grid_import_total = float(np.sum(P_grid_in) * dt_hours)
    grid_export_total = float(np.sum(P_grid_out) * dt_hours)
    p2p_import_total = float(np.sum(P_p2p_in) * dt_hours)
    p2p_export_total = float(np.sum(P_p2p_out) * dt_hours)

    baseline_grid_import_total = float(np.sum(np.maximum(0.0, baseline_load - pv_forecast)) * dt_hours)
    grid_import_reduction = baseline_grid_import_total - grid_import_total
    grid_import_reduction_pct = (
        (grid_import_reduction / baseline_grid_import_total * 100)
        if baseline_grid_import_total > 0 else 0
    )
    
    # Flexibility
    peak_load_baseline = float(np.max(baseline_load))
    peak_load_optimized = float(np.max(L_opt))
    peak_load_reduction = peak_load_baseline - peak_load_optimized
    peak_load_reduction_pct = (peak_load_reduction / peak_load_baseline * 100) if peak_load_baseline > 0 else 0
    
    # Load shifting index (how much load was shifted)
    # Load shifting index (LSI) and comfort violation check
    load_shift_energy = float(np.sum(np.abs(L_opt - baseline_load)) * dt_hours / 2.0)
    lsi = (load_shift_energy / total_load_base) if total_load_base > 0 else 0.0
    load_shift_pct = lsi * 100.0

    comfort_violations = int(np.sum(np.abs(L_opt - baseline_load) > (dr_flexibility * baseline_load + 1e-9)))

    # Battery KPIs
    bur = float(np.sum((P_ch + P_dis) * dt_hours) / (2.0 * battery_capacity_kwh)) if battery_capacity_kwh > 0 else 0.0
    cycles_equiv = float(np.sum(P_dis * dt_hours) / battery_capacity_kwh) if battery_capacity_kwh > 0 else 0.0

    # SoC violations
    soc = np.asarray(results.get("SoC", []), dtype=float)
    soc_min = battery_soc_min_frac * battery_capacity_kwh
    soc_max = battery_soc_max_frac * battery_capacity_kwh
    soc_min_violation = bool(np.nanmin(soc) < soc_min - 1e-6) if soc.size else False
    soc_max_violation = bool(np.nanmax(soc) > soc_max + 1e-6) if soc.size else False

    # Community trading KPIs (priority proof)
    # Demand share decomposition: local share + P2P share + grid share = 1 (approximately)
    gdr = (grid_import_total / total_load_opt) if total_load_opt > 0 else 0.0
    p2p_share = (p2p_import_total / total_load_opt) if total_load_opt > 0 else 0.0
    ler = max(0.0, 1.0 - gdr - p2p_share)
    
    return {
        # Group 1: Economic
        'baseline_cost': baseline_cost,
        'optimized_cost': optimized_cost,
        'cost_reduction': cost_reduction,
        'cost_reduction_pct': cost_reduction_pct,

        # Group 2: Energy
        'pv_self_consumption_rate_base_pct': scr_base,
        'pv_self_sufficiency_rate_base_pct': ssr_base,
        'pv_self_consumption_rate_opt_pct': scr_opt,
        'pv_self_sufficiency_rate_opt_pct': ssr_opt,
        'grid_import_total_kwh': grid_import_total,
        'grid_export_total_kwh': grid_export_total,
        'grid_import_reduction_pct': grid_import_reduction_pct,

        # Group 3: Demand response / flexibility
        'grid_import_reduction_pct': grid_import_reduction_pct,
        'peak_load_reduction_pct': peak_load_reduction_pct,
        'load_shift_pct': load_shift_pct,
        'comfort_violations_count': comfort_violations,
        'peak_load_base_kw': peak_load_baseline,
        'peak_load_opt_kw': peak_load_optimized,

        # Group 4: Battery
        'battery_utilization_rate': bur,
        'battery_equivalent_cycles': cycles_equiv,
        'soc_min_violation': soc_min_violation,
        'soc_max_violation': soc_max_violation,

        # Group 5: Community trading
        'local_energy_utilization_ratio': ler,
        'p2p_share_of_demand': p2p_share,
        'grid_dependency_ratio': gdr,

        # Totals (useful for tables/plots)
        'total_pv_kwh': total_pv,
        'total_load_base_kwh': total_load_base,
        'total_load_opt_kwh': total_load_opt,
        'p2p_import_total_kwh': p2p_import_total,
        'p2p_export_total_kwh': p2p_export_total,
    }


def main():
    """Run optimization pipeline."""
    print("=" * 80)
    print("DAY-AHEAD OPTIMIZATION PIPELINE")
    print("=" * 80)
    
    # Load forecasts
    print("\n1. Loading forecasts...")
    load_fc, pv_fc = load_forecast_files()
    print(f"   Load forecasts: {len(load_fc)} rows")
    print(f"   PV forecasts: {len(pv_fc)} rows")
    
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
    
    # We'll generate a *day-ahead* schedule for a calendar day D (00:00..23:00)
    # by taking forecasts issued at (D - 1 hour) = 23:00 of the previous day:
    # target_times = issue_time + 1h .. +24h  => D 00:00 .. D 23:00
    #
    # To keep runtime reasonable and thesis-representative, default to July 2023.
    start_day = pd.Timestamp("2023-07-01")
    end_day = pd.Timestamp("2023-07-31")
    days = pd.date_range(start_day, end_day, freq="D")

    print(f"\n3. Running day-ahead optimization for {len(days)} days (July 2023)...")
    
    # Group by day and optimize
    all_results = []
    all_kpis = []
    
    for day_start in days:
        issue_time = day_start - pd.Timedelta(hours=1)  # 23:00 previous day
        out = get_day_ahead_vectors(load_fc, pv_fc, issue_time, horizon=24)
        if out is None:
            continue
        load_forecast_24h, pv_forecast_24h, target_times = out

        # We optimize the calendar day starting at target_times[0] (should be day_start 00:00)
        optimized_day = target_times[0].date()

        try:
            results = optimizer.optimize(load_forecast_24h, pv_forecast_24h)
            results['time'] = target_times
            results['issue_time'] = issue_time
            results['day'] = optimized_day
            results['L_base'] = load_forecast_24h
            results['PV_base'] = pv_forecast_24h

            # Baseline uses the same day-ahead forecasts (no DR, no battery, no P2P)
            baseline_load = load_forecast_24h
            kpis = calculate_kpis(
                results,
                baseline_load,
                pv_forecast_24h,
                optimizer.grid_price_import,
                optimizer.grid_price_export,
                dr_flexibility=optimizer.dr_flexibility,
                battery_capacity_kwh=optimizer.battery_capacity,
                battery_soc_min_frac=optimizer.soc_min,
                battery_soc_max_frac=optimizer.soc_max,
                dt_hours=optimizer.dt_hours
            )
            kpis['issue_time'] = issue_time
            kpis['day'] = optimized_day

            all_results.append(results)
            all_kpis.append(kpis)

            if len(all_results) % 10 == 0:
                print(f"   Processed {len(all_results)} days...")

        except Exception as e:
            print(f"   Warning: Optimization failed for issue_time {issue_time}: {e}")
            continue
    
    print(f"\n4. Optimization complete! Processed {len(all_results)} days.")
    
    # Save results
    print("\n5. Saving results...")
    output_dir = Path("outputs/optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine all results
    def _soc_start(r):
        # SoC trajectory has length 25; align SoC at the *start* of each hour interval
        return r['SoC'][:-1]

    results_df = pd.DataFrame({
        'time': [t for r in all_results for t in r['time']],
        'issue_time': [r['issue_time'] for r in all_results for _ in range(24)],
        'day': [r['day'] for r in all_results for _ in range(24)],
        'hour': [pd.Timestamp(t).hour for r in all_results for t in r['time']],
        'L_base': [v for r in all_results for v in r['L_base']],
        'PV_base': [v for r in all_results for v in r['PV_base']],
        'L_opt': [v for r in all_results for v in r['L_opt']],
        'P_ch': [v for r in all_results for v in r['P_ch']],
        'P_dis': [v for r in all_results for v in r['P_dis']],
        'SoC': [v for r in all_results for v in _soc_start(r)],
        'P_grid_in': [v for r in all_results for v in r['P_grid_in']],
        'P_grid_out': [v for r in all_results for v in r['P_grid_out']],
        'P_p2p_in': [v for r in all_results for v in r['P_p2p_in']],
        'P_p2p_out': [v for r in all_results for v in r['P_p2p_out']],
        'total_cost': [r['total_cost'] for r in all_results for _ in range(24)],
        'success': [r.get('success', False) for r in all_results for _ in range(24)],
        'message': [r.get('message', '') for r in all_results for _ in range(24)],
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

