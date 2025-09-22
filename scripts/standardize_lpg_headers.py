from __future__ import annotations

import argparse
import pandas as pd


COMMON_RENAMES = {
    "time": "timestamp",
    "total electricity": "total",
    "total_electricity": "total",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("output_csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    renames = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in COMMON_RENAMES:
            renames[c] = COMMON_RENAMES[lc]
    if renames:
        df = df.rename(columns=renames)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
