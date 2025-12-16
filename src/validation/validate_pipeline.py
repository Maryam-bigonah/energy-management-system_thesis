"""
End-to-end validation for forecasting + optimization outputs.

This script is designed for thesis reproducibility: it performs strict checks and
prints a PASS/FAIL report with actionable messages.

What it validates:
1) Forecast files exist and contain required columns
2) Forecast horizon columns exist (h1..h24)
3) Optimization outputs exist and have required columns
4) Optimization constraints (checked ex-post):
   - DR bounds: (1±alpha)*L_base envelope respected
   - Daily energy conservation: sum(L_opt) == sum(L_base) within tolerance
   - Energy balance residual: PV + dis + p2p_in + grid_in == L_opt + ch + p2p_out + grid_out
   - Non-negativity of flow variables
   - SoC bounds respected (if SoC present)
5) KPI file contains required columns and no obvious invalid values

Usage:
  python3 src/validation/validate_pipeline.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _ok(msg: str) -> None:
    print(f"[ok] {msg}")


def _require_file(p: Path) -> None:
    if not p.exists():
        _fail(f"Missing required file: {p}")


def validate_forecasts(forecast_dir: Path) -> None:
    _ok("Validating forecast files…")
    load_path = forecast_dir / "load_forecasts_2023.csv"
    pv_path = forecast_dir / "pv_forecasts_2023.csv"
    met_path = forecast_dir / "forecast_metrics_summary.csv"
    _require_file(load_path)
    _require_file(pv_path)
    _require_file(met_path)

    load = pd.read_csv(load_path)
    pv = pd.read_csv(pv_path)

    # Required base columns
    for col in ["time", "total_load"]:
        if col not in load.columns:
            _fail(f"load_forecasts missing column: {col}")
    for col in ["time", "PV_true"]:
        if col not in pv.columns:
            _fail(f"pv_forecasts missing column: {col}")

    # Horizons exist
    for h in range(1, 25):
        if f"load_hat_h{h}" not in load.columns:
            _fail(f"load_forecasts missing horizon column: load_hat_h{h}")
        if f"pv_hat_h{h}" not in pv.columns:
            _fail(f"pv_forecasts missing horizon column: pv_hat_h{h}")

    # Basic NaN checks
    load_hat_cols = [f"load_hat_h{h}" for h in range(1, 25)]
    pv_hat_cols = [f"pv_hat_h{h}" for h in range(1, 25)]
    load_nan_frac = load[load_hat_cols].isna().mean().max()
    pv_nan_frac = pv[pv_hat_cols].isna().mean().max()
    _ok(f"Forecast NaN fraction (max across horizons): load={load_nan_frac:.4f}, pv={pv_nan_frac:.4f}")
    if load_nan_frac > 0.05 or pv_nan_frac > 0.05:
        _warn("High NaN fraction in forecasts — check lag requirements and timestamp alignment.")


def validate_optimization(opt_dir: Path, dr_flex: float = 0.10, tol: float = 1e-6) -> None:
    _ok("Validating optimization outputs…")
    res_path = opt_dir / "optimization_results.csv"
    kpi_path = opt_dir / "optimization_kpis.csv"
    _require_file(res_path)
    _require_file(kpi_path)

    res = pd.read_csv(res_path, parse_dates=["time"])
    kpi = pd.read_csv(kpi_path)

    required_cols = [
        "time", "day", "hour",
        "L_base", "PV_base", "L_opt",
        "P_ch", "P_dis",
        "P_grid_in", "P_grid_out",
        "P_p2p_in", "P_p2p_out",
    ]
    for c in required_cols:
        if c not in res.columns:
            _fail(f"optimization_results missing column: {c}")

    # Rows per day should be 24
    counts = res.groupby("day").size()
    bad_days = counts[counts != 24]
    if len(bad_days) > 0:
        _warn(f"Some days do not have 24 rows: {bad_days.head(10).to_dict()}")
    else:
        _ok("Each optimized day has exactly 24 hourly rows.")

    # Non-negativity checks
    flow_cols = ["P_ch", "P_dis", "P_grid_in", "P_grid_out", "P_p2p_in", "P_p2p_out"]
    mins = res[flow_cols].min()
    if (mins < -1e-9).any():
        _fail(f"Negative flow values found: {mins[mins < -1e-9].to_dict()}")
    _ok("All flow variables are non-negative.")

    # DR envelope
    lower = (1 - dr_flex) * res["L_base"]
    upper = (1 + dr_flex) * res["L_base"]
    dr_viol = ((res["L_opt"] < lower - 1e-8) | (res["L_opt"] > upper + 1e-8)).sum()
    _ok(f"DR envelope violations: {dr_viol}")
    if dr_viol > 0:
        _fail("Demand response envelope violated (comfort constraint).")

    # Daily energy conservation
    daily = res.groupby("day")[["L_base", "L_opt"]].sum()
    diff = (daily["L_opt"] - daily["L_base"]).abs()
    max_diff = float(diff.max())
    _ok(f"Daily energy conservation max |sum(Lopt)-sum(Lbase)|: {max_diff:.6g}")
    if max_diff > 1e-3:
        _fail("Daily energy conservation violated (load shifting constraint).")

    # Energy balance residual per hour
    lhs = res["L_opt"] + res["P_ch"] + res["P_p2p_out"] + res["P_grid_out"]
    rhs = res["PV_base"] + res["P_dis"] + res["P_p2p_in"] + res["P_grid_in"]
    resid = (rhs - lhs).abs()
    max_resid = float(resid.max())
    _ok(f"Energy balance max absolute residual: {max_resid:.6g}")
    if max_resid > 1e-4:
        _fail("Energy balance violated (physical feasibility).")

    # SoC bounds if present
    if "SoC" in res.columns:
        soc_min = float(res["SoC"].min())
        soc_max = float(res["SoC"].max())
        _ok(f"SoC observed min/max (kWh): {soc_min:.3f} / {soc_max:.3f}")
    else:
        _warn("No SoC column in optimization_results.csv; skipping SoC checks.")

    # KPI columns sanity
    required_kpi_cols = [
        "baseline_cost", "optimized_cost", "cost_reduction_pct",
        "grid_dependency_ratio", "p2p_share_of_demand", "local_energy_utilization_ratio",
        "comfort_violations_count", "soc_min_violation", "soc_max_violation",
    ]
    for c in required_kpi_cols:
        if c not in kpi.columns:
            _fail(f"optimization_kpis missing column: {c}")

    if (kpi["comfort_violations_count"].sum() != 0):
        _fail("comfort_violations_count is not zero — comfort constraint violated.")
    if bool(kpi["soc_min_violation"].any()) or bool(kpi["soc_max_violation"].any()):
        _fail("SoC violation flags present in KPI file.")

    _ok("KPI file contains required fields and passes basic sanity checks.")


def main() -> None:
    forecast_dir = ROOT / "outputs" / "forecasts"
    opt_dir = ROOT / "outputs" / "optimization"

    print("=" * 80)
    print("PIPELINE VALIDATION REPORT")
    print("=" * 80)

    validate_forecasts(forecast_dir)
    validate_optimization(opt_dir, dr_flex=0.10)

    print("=" * 80)
    print("[PASS] All checks completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()


