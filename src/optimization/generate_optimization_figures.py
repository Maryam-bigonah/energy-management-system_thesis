"""
Generate thesis-ready optimization figures (O1–O5).

Figures:
  O1: Baseline vs Optimized Load (Demand Response) for a representative day
  O2: Battery operation (P_ch, P_dis) + SoC for the same day
  O3: Grid interaction (import/export) for the same day
  O4: Cost comparison (baseline vs optimized) aggregated over a period (default: July 2023)
  O5: Hierarchical energy supply breakdown (Local -> P2P -> Grid) stacked bars (July 2023)

Outputs:
  outputs/optimization/figures/
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _pick_representative_day(results_df: pd.DataFrame) -> str:
    """
    Prefer a mid-summer weekday (e.g., 2023-07-15). If not available,
    pick the first available day in July; else pick the first available day overall.
    """
    if 'day' not in results_df.columns:
        raise ValueError("optimization_results.csv must include a 'day' column.")

    days = pd.to_datetime(results_df['day']).dt.date.astype(str).unique().tolist()
    if "2023-07-15" in days:
        return "2023-07-15"

    july = [d for d in days if d.startswith("2023-07-")]
    if july:
        return sorted(july)[0]

    return sorted(days)[0]


def _load_inputs(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_path = base_dir / "optimization_results.csv"
    kpis_path = base_dir / "optimization_kpis.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}. Run optimization pipeline first.")
    if not kpis_path.exists():
        raise FileNotFoundError(f"Missing {kpis_path}. Run optimization pipeline first.")

    results_df = pd.read_csv(results_path, parse_dates=["time"])
    kpis_df = pd.read_csv(kpis_path, parse_dates=["issue_time"])
    return results_df, kpis_df


def figure_o1(results_df: pd.DataFrame, day: str, outdir: Path, dr_flex: float = 0.10) -> Path:
    ddf = results_df[results_df["day"].astype(str) == day].sort_values("hour")
    if len(ddf) != 24:
        raise ValueError(f"Expected 24 rows for day={day}, got {len(ddf)}.")

    hours = ddf["hour"].to_numpy()
    l_base = ddf["L_base"].to_numpy()
    l_opt = ddf["L_opt"].to_numpy()

    lower = (1 - dr_flex) * l_base
    upper = (1 + dr_flex) * l_base

    plt.figure(figsize=(12, 5))
    plt.fill_between(hours, lower, upper, alpha=0.18, label=f"±{int(dr_flex*100)}% flexibility band")
    plt.plot(hours, l_base, "--", linewidth=2.0, label="Baseline load $L_t^{base}$")
    plt.plot(hours, l_opt, "-", linewidth=2.5, label="Optimized load $L_t^{opt}$")
    plt.xticks(range(0, 24, 2))
    plt.xlabel("Hour of day")
    plt.ylabel("Electricity demand [kW] (hourly average)")
    plt.title(f"Figure O1 — Baseline vs Optimized Load (Demand Response)\nRepresentative day: {day}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    out = outdir / "figure_O1_baseline_vs_optimized_load.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def figure_o2(results_df: pd.DataFrame, day: str, outdir: Path, soc_min: float | None = None, soc_max: float | None = None) -> Path:
    ddf = results_df[results_df["day"].astype(str) == day].sort_values("hour")
    if len(ddf) != 24:
        raise ValueError(f"Expected 24 rows for day={day}, got {len(ddf)}.")

    hours = ddf["hour"].to_numpy()
    p_ch = ddf["P_ch"].to_numpy()
    p_dis = ddf["P_dis"].to_numpy()
    soc = ddf["SoC"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(12, 5))
    width = 0.38
    ax1.bar(hours - width / 2, p_ch, width=width, label="$P_t^{ch}$ (charge)", color="#4C78A8", alpha=0.85)
    ax1.bar(hours + width / 2, p_dis, width=width, label="$P_t^{dis}$ (discharge)", color="#F58518", alpha=0.85)
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Battery power [kW]")
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(hours, soc, color="#54A24B", linewidth=2.5, label="SoC")
    ax2.set_ylabel("State of charge [kWh]")

    if soc_min is not None:
        ax2.axhline(soc_min, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="SoC min")
    if soc_max is not None:
        ax2.axhline(soc_max, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="SoC max")

    fig.suptitle(f"Figure O2 — Battery Operation and SoC\nRepresentative day: {day}", y=0.98)

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    out = outdir / "figure_O2_battery_operation_soc.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def figure_o3(results_df: pd.DataFrame, day: str, outdir: Path) -> Path:
    ddf = results_df[results_df["day"].astype(str) == day].sort_values("hour")
    if len(ddf) != 24:
        raise ValueError(f"Expected 24 rows for day={day}, got {len(ddf)}.")

    hours = ddf["hour"].to_numpy()
    g_in = ddf["P_grid_in"].to_numpy()
    g_out = ddf["P_grid_out"].to_numpy()

    plt.figure(figsize=(12, 5))
    plt.plot(hours, g_in, linewidth=2.5, label="$P_t^{grid,in}$ (import)")
    plt.plot(hours, g_out, linewidth=2.5, label="$P_t^{grid,out}$ (export)")
    plt.xticks(range(0, 24, 2))
    plt.xlabel("Hour of day")
    plt.ylabel("Grid power [kW]")
    plt.title(f"Figure O3 — Grid Interaction (Import/Export)\nRepresentative day: {day}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    out = outdir / "figure_O3_grid_import_export.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def figure_o4(kpis_df: pd.DataFrame, outdir: Path, period_label: str = "July 2023") -> Path:
    # Restrict to July 2023 if possible
    if "day" in kpis_df.columns:
        day_str = pd.to_datetime(kpis_df["day"]).dt.strftime("%Y-%m-%d")
        mask = day_str.str.startswith("2023-07-")
        sub = kpis_df[mask].copy() if mask.any() else kpis_df.copy()
    else:
        sub = kpis_df.copy()

    baseline_cost = float(sub["baseline_cost"].sum())
    optimized_cost = float(sub["optimized_cost"].sum())
    reduction_pct = 100.0 * (baseline_cost - optimized_cost) / baseline_cost if baseline_cost > 0 else 0.0

    plt.figure(figsize=(7, 5))
    labels = ["Baseline", "Optimized"]
    values = [baseline_cost, optimized_cost]
    bars = plt.bar(labels, values, color=["#9ecae1", "#31a354"])
    plt.ylabel("Electricity cost [€]")
    plt.title(f"Figure O4 — Cost Comparison ({period_label})")
    plt.grid(True, axis="y", alpha=0.25)

    # Annotate values
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:,.0f} €", ha="center", va="bottom")
    plt.text(0.5, max(values) * 0.92, f"Cost reduction: {reduction_pct:.1f}%", ha="center", fontsize=11, fontweight="bold")

    out = outdir / "figure_O4_cost_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def figure_o5_supply_breakdown(kpis_df: pd.DataFrame, outdir: Path, period_label: str = "July 2023") -> Path:
    """
    Stacked bars showing supply to demand (kWh):
      Local supply (demand - P2P import - Grid import)
      + P2P import
      + Grid import

    Two bars: Baseline vs Optimized.
    """
    if "day" in kpis_df.columns:
        day_str = pd.to_datetime(kpis_df["day"]).dt.strftime("%Y-%m-%d")
        mask = day_str.str.startswith("2023-07-")
        sub = kpis_df[mask].copy() if mask.any() else kpis_df.copy()
    else:
        sub = kpis_df.copy()

    # Optimized aggregates (kWh)
    demand_opt = float(sub["total_load_opt_kwh"].sum())
    grid_in_opt = float(sub["grid_import_total_kwh"].sum())
    p2p_in_opt = float(sub["p2p_import_total_kwh"].sum())
    local_opt = max(0.0, demand_opt - grid_in_opt - p2p_in_opt)

    # Baseline aggregates (kWh)
    demand_base = float(sub["total_load_base_kwh"].sum())
    # Baseline has no P2P imports by definition
    # Approximate baseline grid import as demand - PV_used_local (equivalently, from kpis: total_load - pv_used_local)
    # We stored baseline grid import implicitly via reduction metrics, but not directly.
    # Here: baseline grid import = demand - (PV used locally in baseline)
    # PV used locally in baseline = SSR_base * total_load_base
    pv_used_local_base = float(sub["pv_self_sufficiency_rate_base_pct"].mean() / 100.0 * demand_base) if demand_base > 0 else 0.0
    grid_in_base = max(0.0, demand_base - pv_used_local_base)
    p2p_in_base = 0.0
    local_base = max(0.0, demand_base - grid_in_base - p2p_in_base)

    labels = ["Baseline", "Optimized"]
    local = [local_base, local_opt]
    p2p = [p2p_in_base, p2p_in_opt]
    grid = [grid_in_base, grid_in_opt]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, local, label="Local (PV + battery)", color="#2ca02c", alpha=0.85)
    plt.bar(labels, p2p, bottom=local, label="P2P import", color="#ff7f0e", alpha=0.85)
    plt.bar(labels, grid, bottom=(np.array(local) + np.array(p2p)), label="Grid import", color="#1f77b4", alpha=0.85)
    plt.ylabel("Energy supplied to demand [kWh]")
    plt.title(f"Figure O5 — Supply Breakdown (Local → P2P → Grid)\n{period_label}")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    out = outdir / "figure_O5_supply_breakdown_priority.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def main() -> None:
    base_dir = Path("outputs/optimization")
    outdir = base_dir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    results_df, kpis_df = _load_inputs(base_dir)
    rep_day = _pick_representative_day(results_df)

    # Try to infer SoC bounds from results (optional, for plotting horizontal lines)
    soc_min = None
    soc_max = None
    if "SoC" in results_df.columns and results_df["SoC"].notna().any():
        # Use observed min/max as fallback
        soc_min = float(np.nanmin(results_df["SoC"]))
        soc_max = float(np.nanmax(results_df["SoC"]))

    p1 = figure_o1(results_df, rep_day, outdir, dr_flex=0.10)
    p2 = figure_o2(results_df, rep_day, outdir, soc_min=None, soc_max=None)
    p3 = figure_o3(results_df, rep_day, outdir)
    p4 = figure_o4(kpis_df, outdir, period_label="July 2023")
    p5 = figure_o5_supply_breakdown(kpis_df, outdir, period_label="July 2023")

    print("[ok] Saved optimization figures:")
    for p in [p1, p2, p3, p4, p5]:
        print(" -", p)
    print("[ok] Representative day:", rep_day)


if __name__ == "__main__":
    main()


