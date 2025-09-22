from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from torino_energy.data_sources.lpg import CATEGORY_COLUMNS, LPGConfig, load_and_aggregate


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / "torino_energy" / "config" / "settings.yaml"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_settings() -> Dict:
    with open(SETTINGS, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_config(cfg: Dict) -> LPGConfig:
    lpg = cfg["sources"]["lpg"]
    return LPGConfig(
        timezone=lpg.get("timezone", "Europe/Rome"),
        files=lpg["files"],
        units=lpg["units"],
        appliance_category_map=lpg["appliance_category_map"],
    )


def validate_daily_totals(df: pd.DataFrame, tolerance: float = 0.005) -> None:
    # Daily aggregation per unit
    daily = df.groupby(["unit_id", pd.Grouper(key="timestamp", freq="1D")]).sum(numeric_only=True)
    cat_sum = daily[CATEGORY_COLUMNS].sum(axis=1)
    total = daily["e_load_total_kWh"]
    diff_ratio = ((total - cat_sum).abs() / total.clip(lower=1e-9)).fillna(0.0)
    if (diff_ratio > tolerance).any():
        worst = diff_ratio.max()
        raise AssertionError(f"Daily totals differ by more than {tolerance*100:.2f}% (worst {worst*100:.2f}%)")


def plot_per_unit(df: pd.DataFrame) -> None:
    units = sorted(df["unit_id"].unique())
    for u in units:
        sub = df[df["unit_id"] == u].set_index("timestamp")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(sub.index, sub["e_load_total_kWh"], label="total")
        ax.set_title(f"Unit {u} - Total hourly consumption (kWh)")
        ax.set_ylabel("kWh")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"unit_{u}_total.png", dpi=140)
        plt.close(fig)


def plot_compare_units(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(index="timestamp", columns="unit_id", values="e_load_total_kWh", aggfunc="sum")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pivot.index, pivot)
    ax.set_title("All units - total hourly consumption comparison")
    ax.set_ylabel("kWh")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "units_comparison.png", dpi=140)
    plt.close(fig)


def main() -> None:
    cfg = load_settings()
    lpg_cfg = build_config(cfg)
    df = load_and_aggregate(lpg_cfg)
    validate_daily_totals(df)
    plot_per_unit(df)
    plot_compare_units(df)
    print(f"Done. Plots in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
