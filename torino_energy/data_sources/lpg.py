from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import pandas as pd


@dataclass(frozen=True)
class LPGConfig:
    timezone: str
    files: Mapping[str, str]
    units: List[str]
    appliance_category_map: Mapping[str, List[str]]


CATEGORY_COLUMNS = [
    "e_base_kWh",
    "e_hvac_kWh",
    "e_kitchen_kWh",
    "e_laundry_kWh",
    "e_entertain_kWh",
    "e_misc_kWh",
]


def _read_lpg_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect columns: timestamp, total, plus appliance columns. Accept case-insensitive headers.
    cols = {c.lower(): c for c in df.columns}
    if "timestamp" in cols:
        ts_col = cols["timestamp"]
    elif "time" in cols:
        ts_col = cols["time"]
    else:
        raise ValueError(f"Missing timestamp column in {path}")

    # Normalize timestamp to tz-aware
    ts = pd.to_datetime(df[ts_col], utc=False, errors="coerce")
    if ts.isna().any():
        raise ValueError(f"Invalid timestamps in {path}")
    df = df.set_index(ts).drop(columns=[ts_col])
    return df


def _map_appliances_to_categories(df: pd.DataFrame, mapping: Mapping[str, List[str]]) -> Tuple[pd.DataFrame, List[str]]:
    # Build per-category sums; appliance names are matched case-insensitively by substring
    lower_cols = {c.lower(): c for c in df.columns}

    def find_cols(names: Iterable[str]) -> List[str]:
        matched: List[str] = []
        for name in names:
            key = name.lower()
            # exact or substring match
            for lc, orig in lower_cols.items():
                if key == lc or key in lc:
                    matched.append(orig)
        return sorted(set(matched))

    category_to_cols: Dict[str, List[str]] = {
        cat: find_cols(names) for cat, names in mapping.items()
    }

    out: Dict[str, pd.Series] = {}
    used_cols: List[str] = []
    for cat, cols in category_to_cols.items():
        used_cols.extend(cols)
        out[f"e_{cat}_kWh"] = df[cols].sum(axis=1) / 1000.0 if len(cols) else pd.Series(0.0, index=df.index)

    # total from data if present
    total_col = None
    for candidate in ["total", "total electricity", "total_electricity", "sum"]:
        if candidate in lower_cols:
            total_col = lower_cols[candidate]
            break

    if total_col is not None:
        e_total = df[total_col] / 1000.0
    else:
        # fallback: sum all numeric columns
        e_total = df.select_dtypes("number").sum(axis=1) / 1000.0

    out_df = pd.DataFrame(out, index=df.index).sort_index()
    out_df.insert(0, "e_load_total_kWh", e_total)

    return out_df, used_cols


def load_and_aggregate(cfg: LPGConfig) -> pd.DataFrame:
    """Read LPG CSVs per archetype, resample to hourly kWh, map to categories, and assemble 20 units.

    Notes:
    - Expects 1-minute Wh data; we sum to hourly and convert Wh->kWh.
    - No data is generated; CSVs must exist at cfg.files paths.
    """
    hourly_by_arch: Dict[str, pd.DataFrame] = {}

    for archetype, path in cfg.files.items():
        df_raw = _read_lpg_csv(path)
        # Sum to hourly in Wh, then convert to kWh inside mapping
        df_hour = df_raw.resample("1H").sum()
        mapped_df, used_cols = _map_appliances_to_categories(df_hour, cfg.appliance_category_map)
        # Ensure tz
        mapped_df.index = mapped_df.index.tz_localize(None).tz_localize(cfg.timezone, nonexistent="shift_forward", ambiguous="NaT").tz_convert(cfg.timezone)
        hourly_by_arch[archetype] = mapped_df

    # Assemble units according to cfg.units order
    frames: List[pd.DataFrame] = []
    for unit_id, archetype in enumerate(cfg.units, start=1):
        if archetype not in hourly_by_arch:
            raise ValueError(f"Archetype {archetype} missing in files config")
        df = hourly_by_arch[archetype].copy()
        df.insert(0, "unit_id", unit_id)
        df.insert(1, "archetype", archetype)
        df.insert(2, "timestamp", df.index)
        frames.append(df.reset_index(drop=True))

    result = pd.concat(frames, axis=0, ignore_index=True)

    # Validation checks
    category_sum = result[CATEGORY_COLUMNS].sum(axis=1)
    if (result["e_load_total_kWh"] + 1e-9 < category_sum).any():
        raise AssertionError("Total load is less than sum of categories for some rows")
    if (result[["e_load_total_kWh"] + CATEGORY_COLUMNS] < -1e-9).any().any():
        raise AssertionError("Negative energy values found")

    return result[[
        "timestamp",
        "unit_id",
        "archetype",
        "e_load_total_kWh",
        "e_base_kWh",
        "e_hvac_kWh",
        "e_kitchen_kWh",
        "e_laundry_kWh",
        "e_entertain_kWh",
        "e_misc_kWh",
    ]]
