"""
Cleaning + preparation utilities for the already-generated master dataset:

    outputs/MASTER_20_APARTMENTS_2022_2023.csv

This script does NOT create synthetic measurements. It only:
  - parses and sorts timestamps
  - validates basic integrity (duplicates, missing values, hourly gaps)
  - writes a cleaned copy with stable dtypes
  - creates time-based train/test splits (2022 train, 2023 test)

Outputs (by default) into:
  outputs/MASTER_20_APARTMENTS_2022_2023_clean.csv
  outputs/splits/train_2022.csv
  outputs/splits/test_2023.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    # src/forecasting/prepare_master_dataset.py -> project root
    return Path(__file__).resolve().parents[2]


def load_master_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def validate_master(df: pd.DataFrame, *, strict_hourly: bool = False) -> None:
    if "time" not in df.columns:
        raise ValueError("Expected a 'time' column.")

    if df["time"].isna().any():
        raise ValueError("Found NaT/NaN in 'time' column.")

    dup = int(df["time"].duplicated().sum())
    if dup:
        raise ValueError(f"Found {dup} duplicate timestamps in 'time'.")

    # check that total_load matches sum(load_ap*)
    load_cols = [c for c in df.columns if c.startswith("load_ap")]
    if not load_cols:
        raise ValueError("No apartment load columns found (expected load_ap1..load_ap20).")
    if "total_load" not in df.columns:
        raise ValueError("Missing 'total_load' column.")

    max_abs_err = float((df[load_cols].sum(axis=1) - df["total_load"]).abs().max())
    if max_abs_err > 1e-8:
        raise ValueError(
            f"total_load mismatch: max |sum(load_ap*) - total_load| = {max_abs_err}"
        )

    # check hourly continuity (optional strict mode)
    # NOTE: Some LPG profiles may not cover a full calendar year. In that case,
    # the combined multi-year dataset will naturally contain gaps (e.g., missing
    # the last few days of a year). We never fill those gaps.
    s = df.set_index("time").index
    expected = pd.date_range(s.min(), s.max(), freq="1h")
    missing = expected.difference(s)
    if len(missing) > 0:
        sample = ", ".join(str(x) for x in missing[:5])
        msg = (
            f"Found {len(missing)} missing hourly timestamps inside the dataset range. "
            f"First few: {sample}"
        )
        if strict_hourly:
            raise ValueError(msg)
        else:
            print(f"[warn] {msg}")

    # check for missing values in non-time columns
    non_time_cols = [c for c in df.columns if c != "time"]
    na_cols = df[non_time_cols].isna().mean()
    bad = na_cols[na_cols > 0]
    if len(bad) > 0:
        top = bad.sort_values(ascending=False).head(10)
        raise ValueError(
            "Found missing values in feature columns. Top missing fractions:\n"
            + top.to_string()
        )


def cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make dtypes consistent and smaller on disk.
    We keep all numeric values as-is (no rounding), only cast types.
    """
    out = df.copy()

    # integer-like calendar fields
    int_cols = ["hour", "dow", "is_weekend", "month", "season"]
    for c in int_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    # Everything else numeric -> float64 (safe) then optionally float32
    # We'll keep float64 for scientific reproducibility.
    for c in out.columns:
        if c == "time":
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train: year 2022
    Test : year 2023
    """
    train = df[df["time"].dt.year == 2022].copy()
    test = df[df["time"].dt.year == 2023].copy()
    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            "Train/test split produced an empty set. "
            "Check that your master dataset covers 2022 and 2023."
        )
    return train, test


def main() -> None:
    root = _project_root()
    default_in = root / "outputs" / "MASTER_20_APARTMENTS_2022_2023.csv"
    default_out = root / "outputs" / "MASTER_20_APARTMENTS_2022_2023_clean.csv"
    default_splits_dir = root / "outputs" / "splits"

    parser = argparse.ArgumentParser(description="Clean and split the master dataset.")
    parser.add_argument("--input", type=str, default=str(default_in), help="Input master CSV")
    parser.add_argument("--output", type=str, default=str(default_out), help="Output cleaned CSV")
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=str(default_splits_dir),
        help="Directory to write train/test CSVs",
    )
    parser.add_argument(
        "--strict-hourly",
        action="store_true",
        help="Fail if there are missing hourly timestamps between min(time) and max(time).",
    )
    args = parser.parse_args()

    in_path = Path(args.input).expanduser()
    out_path = Path(args.output).expanduser()
    splits_dir = Path(args.splits_dir).expanduser()

    df = load_master_csv(in_path)
    validate_master(df, strict_hourly=args.strict_hourly)
    df = cast_dtypes(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    train, test = split_train_test(df)
    splits_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(splits_dir / "train_2022.csv", index=False)
    test.to_csv(splits_dir / "test_2023.csv", index=False)

    print(f"[ok] cleaned: {out_path}")
    print(f"[ok] train  : {splits_dir / 'train_2022.csv'}  (rows={len(train)})")
    print(f"[ok] test   : {splits_dir / 'test_2023.csv'}   (rows={len(test)})")


if __name__ == "__main__":
    main()


