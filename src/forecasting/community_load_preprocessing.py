"""
End-to-end preprocessing to build:

    * 4 household-type load profiles from LPG SumProfiles files
    * A 20-apartment load matrix (load_ap1 ... load_ap20, total_load)
    * A merged master dataset with PVGIS + OpenWeather + calendar features
      aligned to the 20-apartment loads.

The script uses the **user's real absolute file paths** as provided in the
conversation. Adjust paths or the `ASSIGNMENTS` dictionary below if you
change scenarios.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import pandas as pd

from .data_loading import _read_pvgis_timeseries


# -----------------------------
# ABSOLUTE INPUT FILE PATHS
# -----------------------------

PATH_PVGIS = Path(
    "/Users/mariabigonah/Desktop/thesis/building database/PVGIS_data_2022_2023.csv"
)

PATH_OPENWEATHER = Path(
    "/Users/mariabigonah/Desktop/thesis/building database/openweather_turin_2022_2023.csv"
)

# LPG load profiles (as provided by the user)
PATH_LPG_RETIRED = Path(
    "/Users/mariabigonah/Desktop/thesis/CHR54 Retired Couple, no work/"
    "SumProfiles_3600s.Electricity.Retired Couple, no work.csv"
)

PATH_LPG_WORKING_COUPLE = Path(
    "/Users/mariabigonah/Desktop/thesis/CHR54 Retired Couple, no work/"
    "CHR01 Couple both at Work/"
    "SumProfiles_couple work.Electricity.Retired Couple, no work.csv"
)

PATH_LPG_FAMILY_ONE_CHILD = Path(
    "/Users/mariabigonah/Desktop/thesis/CHR54 Retired Couple, no work/"
    "CHR03 Family,1 child both at work/"
    "SumProfiles.1 child both at work.Electricity.csv"
)

PATH_LPG_FAMILY_THREE_CHILDREN_ONE_HOME = Path(
    "/Users/mariabigonah/Desktop/thesis/building database/"
    "SumProfiles_3 children 1 at home,1 at work.HH1.Electricity.csv"
)

# NOTE:
# This profile represents a family with 3 children where one adult works
# and one adult stays at home, resulting in higher daytime electricity demand
# compared to dual-income households. This profile replaces the previous
# CHR44 "2 children" Apparent power profile to improve data quality and
# represent a more diverse household composition.
PATH_LPG_FAMILY_TWO_CHILDREN_DEFAULT = PATH_LPG_FAMILY_THREE_CHILDREN_ONE_HOME

# Default output location for the final master dataset
PATH_OUTPUT_MASTER = Path(
    "/Users/mariabigonah/Desktop/thesis/building database/"
    "MASTER_20_APARTMENTS_2022_2023.csv"
)

# Also write a copy inside the repo so it's easy to find in Finder/VSCode.
PATH_OUTPUT_MASTER_LOCAL_COPY = (
    Path(__file__).resolve().parents[2] / "outputs" / "MASTER_20_APARTMENTS_2022_2023.csv"
)


# -----------------------------
# LOAD HELPERS FOR LPG FILES
# -----------------------------

def _parse_lpg_hourly(filepath: Path) -> pd.DataFrame:
    """
    Parse an LPG SumProfiles file with HOURLY energy data.

    Expected format (semicolon separated):
        Electricity.Timestep;Time;Sum [kWh]
    or
        Apparent.Timestep;Time;Sum [kWh]

    Returns
    -------
    DataFrame with columns:
        time (datetime64[ns])
        load_kWh (float)
    covering one synthetic year (2016) at 1-hour resolution.
    """
    df = pd.read_csv(filepath, sep=";")

    if "Time" not in df.columns or "Sum [kWh]" not in df.columns:
        raise ValueError(
            f"Unexpected columns in LPG file {filepath}. "
            f"Expected 'Time' and 'Sum [kWh]'. Got {list(df.columns)}"
        )

    # LPG exports can mix strings like "1/1/2016 00:00" and "11/13/2016 00:00".
    # Using dayfirst=True breaks on "11/13" (13 cannot be a month), so we
    # instead let pandas infer with the default month-first interpretation,
    # which correctly parses all of the user's files.
    df["time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["load_kWh"] = pd.to_numeric(df["Sum [kWh]"], errors="coerce")
    df = df[["time", "load_kWh"]].dropna(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def _parse_lpg_minute_to_hourly(filepath: Path) -> pd.DataFrame:
    """
    Parse an LPG SumProfiles file with MINUTE resolution and aggregate to hours.

    Expected format (semicolon separated):
        Electricity.Timestep;Time;Sum [kWh]
    with 'Electricity.Timestep' increasing by 1 for each minute.

    Returns a DataFrame:
        time (hourly start, datetime64[ns])
        load_kWh (float, sum over the 60 minutes in that hour)
    """
    df = pd.read_csv(filepath, sep=";")

    if "Time" not in df.columns or "Sum [kWh]" not in df.columns:
        raise ValueError(
            f"Unexpected columns in LPG file {filepath}. "
            f"Expected 'Time' and 'Sum [kWh]'. Got {list(df.columns)}"
        )

    # Same reasoning as in _parse_lpg_hourly: allow pandas to infer the
    # format and do not force day-first to avoid ValueError on strings like
    # "11/13/2016 00:00".
    df["time"] = pd.to_datetime(df["Time"], errors="coerce")
    df["load_kWh_minute"] = pd.to_numeric(df["Sum [kWh]"], errors="coerce")

    df = df.dropna(subset=["time"])
    df = df.set_index("time").sort_index()

    # Aggregate minute values to hourly energy
    hourly = df["load_kWh_minute"].resample("1h").sum().to_frame("load_kWh")
    hourly = hourly.reset_index()
    return hourly


def extend_profile_years(
    df: pd.DataFrame,
    original_year: int = 2016,
    target_years: Iterable[int] = (2022, 2023),
) -> pd.DataFrame:
    """
    Replicate a one-year LPG profile (e.g. 2016) to multiple target years.

    Parameters
    ----------
    df : DataFrame
        Columns: 'time', 'load_kWh'. One synthetic year.
    original_year : int
        Year present in the LPG file (2016 in your case).
    target_years : iterable of int
        Years to which the profile should be copied (e.g. 2022–2024).
    """
    def _shift_year(ts: pd.Timestamp, target_year: int) -> pd.Timestamp:
        """
        Move a timestamp from original_year to target_year.

        Handles leap-day (29 Feb) specially: if the target year is not a
        leap year and the date would be invalid, the value is moved to
        28 Feb of the target year. This keeps all original measurements,
        only reassigning that single calendar day, without inventing any
        synthetic values.
        """
        if ts.year != original_year:
            return ts
        try:
            return ts.replace(year=target_year)
        except ValueError:
            # Typical case: 2016-02-29 -> 2022-02-29 (invalid)
            if ts.month == 2 and ts.day == 29:
                return ts.replace(year=target_year, day=28)
            raise

    out: List[pd.DataFrame] = []
    for year in target_years:
        tmp = df.copy()
        tmp["time"] = tmp["time"].apply(lambda ts, y=year: _shift_year(ts, y))
        # If original_year is a leap year (2016) and the target year is not,
        # the 29-Feb values get moved (see _shift_year). This can lead to
        # duplicate timestamps (e.g. 28-Feb receives both 28-Feb and 29-Feb).
        # We handle this by summing kWh for identical timestamps, which
        # preserves energy without inventing new values.
        tmp = tmp.groupby("time", as_index=False)["load_kWh"].sum()
        out.append(tmp)

    out_df = pd.concat(out, ignore_index=True)
    out_df = out_df.sort_values("time").reset_index(drop=True)
    return out_df


def build_household_profiles(
    path_family_two_children: Path = PATH_LPG_FAMILY_TWO_CHILDREN_DEFAULT,
) -> Dict[str, pd.DataFrame]:
    """
    Build the four household-type profiles as hourly load_kWh for 2022–2024.

    Household types:
        1. Retired couple (no work)
        2. Working couple (both at work)
        3. Family with 1 child (both adults at work)
        4. Family with 3 children (1 adult at work, 1 at home) – higher daytime demand

    Returns
    -------
    dict
        Keys:
            'retired'
            'working'
            'family_one_child'
            'family_two_children'  (note: now represents 3-children family)
        Each value is a DataFrame with columns ['time', 'load_kWh'].
    """
    # 1) Retired couple – already hourly SumProfiles_3600s
    retired_2016 = _parse_lpg_hourly(PATH_LPG_RETIRED)

    # 2) Working couple – hourly SumProfiles file
    working_2016 = _parse_lpg_hourly(PATH_LPG_WORKING_COUPLE)

    # 3) Family, 1 child both at work – minute resolution, aggregate to hourly
    family_one_child_2016 = _parse_lpg_minute_to_hourly(PATH_LPG_FAMILY_ONE_CHILD)

    # 4) Family with 3 children (1 adult at work, 1 at home) – represents households
    # with daytime occupancy and higher midday electricity consumption.
    # Variable name kept as 'family_two_children' for backward compatibility.
    family_two_children_2016 = _parse_lpg_hourly(path_family_two_children)

    # Extend each profile from 2016 → 2022–2024
    target_years = (2022, 2023)
    retired = extend_profile_years(retired_2016, target_years=target_years)
    working = extend_profile_years(working_2016, target_years=target_years)
    family_one_child = extend_profile_years(
        family_one_child_2016, target_years=target_years
    )
    family_two_children = extend_profile_years(
        family_two_children_2016, target_years=target_years
    )

    return {
        "retired": retired,
        "working": working,
        "family_one_child": family_one_child,
        "family_two_children": family_two_children,
    }


# -----------------------------
# APARTMENT ASSIGNMENTS (20 units)
# -----------------------------

ASSIGNMENTS: Mapping[str, str] = {
    # ap: household type key from build_household_profiles()
    "ap1": "retired",
    "ap2": "working",
    "ap3": "family_one_child",
    "ap4": "family_two_children",
    "ap5": "retired",
    "ap6": "working",
    "ap7": "family_one_child",
    "ap8": "family_two_children",
    "ap9": "retired",
    "ap10": "working",
    "ap11": "family_one_child",
    "ap12": "family_two_children",
    "ap13": "retired",
    "ap14": "working",
    "ap15": "family_one_child",
    "ap16": "family_two_children",
    "ap17": "retired",
    "ap18": "working",
    "ap19": "family_one_child",
    "ap20": "family_two_children",
}


def build_20_apartment_load_matrix(
    profiles: Mapping[str, pd.DataFrame],
    assignments: Mapping[str, str] = ASSIGNMENTS,
) -> pd.DataFrame:
    """
    Build the apartment-level load matrix for 20 apartments.

    Parameters
    ----------
    profiles : dict
        Output of build_household_profiles().
    assignments : dict
        Mapping from 'ap1'..'ap20' -> household type string.

    Returns
    -------
    DataFrame
        Columns:
            time, load_ap1, ..., load_ap20, total_load
    """
    # IMPORTANT: Some LPG exports may cover different time horizons (e.g. the
    # 'two_children' file ends earlier). We never fill missing values.
    # Instead, we restrict to the intersection of timestamps available in ALL
    # household profiles, ensuring every apartment has real (non-NaN) load.
    time_sets = []
    for name, prof in profiles.items():
        if "time" not in prof.columns:
            raise ValueError(f"Profile '{name}' is missing a 'time' column.")
        time_sets.append(set(pd.to_datetime(prof["time"])))

    common_times = set.intersection(*time_sets) if time_sets else set()
    if not common_times:
        raise ValueError("No common timestamps found across household profiles.")

    common_times = sorted(common_times)
    master = pd.DataFrame({"time": common_times})

    for ap, family in assignments.items():
        if family not in profiles:
            raise KeyError(f"Unknown family type '{family}' for apartment '{ap}'.")
        df_family = profiles[family].copy()
        df_family["time"] = pd.to_datetime(df_family["time"])
        df_family = df_family[df_family["time"].isin(common_times)]
        df_family = df_family.rename(columns={"load_kWh": f"load_{ap}"})
        master = master.merge(df_family[["time", f"load_{ap}"]], on="time", how="left")

    load_cols = [f"load_ap{i}" for i in range(1, 21)]
    master["total_load"] = master[load_cols].sum(axis=1)
    return master


# -----------------------------
# PVGIS + OPENWEATHER + CALENDAR
# -----------------------------

def build_pv_and_weather_features() -> pd.DataFrame:
    """
    Build hourly PV + irradiance + OpenWeather features for 2022–2024.

    Returns
    -------
    DataFrame
        Indexed by time with columns:
            PV_true, Gb, Gd, Gr, H_sun, T2m, WS10m,
            temp, humidity, wind_speed, clouds
    """
    def _find_pvgis_skiprows(filepath: Path) -> int:
        with filepath.open("r") as f:
            skiprows = 0
            for line in f:
                if line.startswith("time,"):
                    break
                skiprows += 1
        return skiprows

    def _read_pvgis_hourly_chunked(
        filepath: Path,
        start: str,
        end: str,
        chunksize: int = 200_000,
    ) -> pd.DataFrame:
        """
        Memory-friendly PVGIS reader.

        Reads PVGIS in chunks, filters to [start, end], then aggregates
        10-minute values to hourly means.
        """
        skiprows = _find_pvgis_skiprows(filepath)
        usecols = ["time", "P", "Gb(i)", "Gd(i)", "Gr(i)", "H_sun", "T2m", "WS10m"]
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        # incremental sums and counts per hour
        sum_df: pd.DataFrame | None = None
        cnt_df: pd.DataFrame | None = None

        for chunk in pd.read_csv(
            filepath,
            skiprows=skiprows,
            usecols=usecols,
            low_memory=False,
            chunksize=chunksize,
        ):
            # Parse PVGIS time.
            # Your 2022–2023 export is already ISO-like: "2022-01-01 00:10:00".
            # Other PVGIS exports can be "YYYYMMDD:HHMM". We support both.
            raw = chunk["time"].astype(str)
            # Fast path: try normal datetime parsing
            parsed = pd.to_datetime(raw, errors="coerce")
            # Fallback: handle "YYYYMMDD:HHMM" by stripping ":" then parsing
            needs_fallback = parsed.isna() & raw.str.contains(":") & raw.str.match(r"^\d{8}:\d{4}$", na=False)
            if needs_fallback.any():
                fb = raw[needs_fallback].str.replace(":", "", regex=False)
                parsed.loc[needs_fallback] = pd.to_datetime(fb, format="%Y%m%d%H%M", errors="coerce")
            chunk["time"] = parsed
            chunk = chunk.dropna(subset=["time"])

            # Filter to time window before doing any heavy work
            chunk = chunk[(chunk["time"] >= start_ts) & (chunk["time"] <= end_ts)]
            if chunk.empty:
                continue

            # Convert numerics
            for col in usecols:
                if col == "time":
                    continue
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

            # P is in W for 1 kWp; convert to kW and later hourly-mean it
            chunk["P_kW"] = chunk["P"] / 1000.0

            # Group by hour
            hour = chunk["time"].dt.floor("h")
            cols_for_agg = ["P_kW", "Gb(i)", "Gd(i)", "Gr(i)", "H_sun", "T2m", "WS10m"]
            grouped_sum = chunk[cols_for_agg].groupby(hour).sum(min_count=1)
            grouped_cnt = chunk[cols_for_agg].groupby(hour).count()

            if sum_df is None:
                sum_df = grouped_sum
                cnt_df = grouped_cnt
            else:
                sum_df = sum_df.add(grouped_sum, fill_value=0.0)
                cnt_df = cnt_df.add(grouped_cnt, fill_value=0.0)

        if sum_df is None or cnt_df is None:
            raise ValueError(
                "No PVGIS rows found in the requested time window. "
                "Check the PVGIS file coverage vs the requested 2022–2024 period."
            )

        hourly_mean = sum_df.divide(cnt_df.where(cnt_df != 0))
        hourly_mean = hourly_mean.sort_index()

        hourly_mean = hourly_mean.rename(
            columns={
                "P_kW": "PV_true",
                "Gb(i)": "Gb",
                "Gd(i)": "Gd",
                "Gr(i)": "Gr",
            }
        )
        return hourly_mean

    # OpenWeather hourly dataset (already hourly)
    df_ow = pd.read_csv(PATH_OPENWEATHER, parse_dates=["time"])
    df_ow = df_ow.set_index("time").sort_index()
    df_ow = df_ow[["temp", "humidity", "wind_speed", "clouds"]]

    # Use OpenWeather coverage as the authoritative time window for alignment
    start = df_ow.index.min().floor("h")
    end = df_ow.index.max().ceil("h")

    # PVGIS hourly features (chunked to avoid macOS killing the process)
    pvgis_hourly = _read_pvgis_hourly_chunked(PATH_PVGIS, start=str(start), end=str(end))

    # Merge PVGIS + OpenWeather.
    # Use left join to keep the full OpenWeather range (2022–2024) even if PVGIS
    # doesn't cover all those years. We'll report missingness later.
    df = df_ow.join(pvgis_hourly, how="left")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hour, day-of-week, weekend flag, month, and season to a dataframe
    indexed by DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must use a DatetimeIndex to add calendar features.")

    out = df.copy()
    idx = out.index

    out["hour"] = idx.hour
    out["dow"] = idx.dayofweek
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
    out["month"] = idx.month

    # 0=winter, 1=spring, 2=summer, 3=autumn
    season_map = {
        12: 0,
        1: 0,
        2: 0,
        3: 1,
        4: 1,
        5: 1,
        6: 2,
        7: 2,
        8: 2,
        9: 3,
        10: 3,
        11: 3,
    }
    out["season"] = out["month"].map(season_map).astype("Int64")
    return out


# -----------------------------
# MASTER PIPELINE
# -----------------------------

def build_master_dataset(
    output_path: Path = PATH_OUTPUT_MASTER,
    require_complete_features: bool = True,
    also_write_local_copy: bool = True,
    path_family_two_children: Path = PATH_LPG_FAMILY_TWO_CHILDREN_DEFAULT,
) -> pd.DataFrame:
    """
    Build the full master dataset for 20 apartments + PV + weather + calendar.

    Steps:
        1) Build four household-type profiles and extend them to 2022–2024.
        2) Assemble the 20-apartment load matrix (load_ap1..load_ap20, total_load).
        3) Build PV + irradiance + OpenWeather features at 1-hour resolution.
        4) Inner-join loads with PV+weather on time.
        5) Add calendar features.
        6) Save to CSV.

    Returns
    -------
    DataFrame
        The final master dataset, also written to `output_path`.
    """
    print("[1/5] Building household profiles and 20-apartment load matrix...")
    profiles = build_household_profiles(path_family_two_children=path_family_two_children)
    loads = build_20_apartment_load_matrix(profiles)
    loads = loads.set_index("time").sort_index()

    print("[2/5] Building PVGIS + OpenWeather hourly features...")
    pv_weather = build_pv_and_weather_features()

    print("[3/5] Merging loads with PV/weather...")
    # Start from the full load horizon, then optionally restrict to the
    # intersection where PVGIS + OpenWeather are fully available.
    df = loads.join(pv_weather, how="left")

    print("[4/5] Adding calendar features...")
    df = add_calendar_features(df)

    # Report missingness in PV/weather features (helps diagnose PVGIS coverage)
    feature_cols = ["PV_true", "Gb", "Gd", "Gr", "H_sun", "T2m", "WS10m", "temp", "humidity", "wind_speed", "clouds"]
    present_cols = [c for c in feature_cols if c in df.columns]
    if present_cols:
        missing_frac = df[present_cols].isna().mean().sort_values(ascending=False)
        worst = missing_frac.iloc[0]
        if worst > 0:
            print("[info] Missing fractions (top 5):")
            print(missing_frac.head(5).to_string())

    if require_complete_features:
        required = [c for c in feature_cols if c in df.columns]
        before = len(df)
        df = df.dropna(subset=required)
        after = len(df)
        if before != after:
            print(
                f"[info] Dropped {before - after} rows to keep only timestamps where "
                f"all PV/weather features are present."
            )

    if len(df) == 0:
        raise ValueError(
            "After aligning sources, there are 0 rows left. This usually means the "
            "time ranges do not overlap (e.g., PVGIS file doesn't cover 2022–2024)."
        )

    print(f"[info] Final time range: {df.index.min()}  →  {df.index.max()} (rows={len(df)})")

    print(f"[5/5] Saving CSV to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(names="time").to_csv(output_path, index=False)

    if also_write_local_copy:
        local_path = PATH_OUTPUT_MASTER_LOCAL_COPY
        local_path.parent.mkdir(parents=True, exist_ok=True)
        df.reset_index(names="time").to_csv(local_path, index=False)
        print(f"[info] Also wrote a copy here: {local_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a master dataset for 20 apartments (2022–2023) by merging LPG loads, PVGIS, and OpenWeather."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PATH_OUTPUT_MASTER),
        help="Where to write the master CSV (absolute path recommended).",
    )
    parser.add_argument(
        "--keep-missing",
        action="store_true",
        help="If set, do not drop rows with missing PV/weather features (keeps full load timeline).",
    )
    parser.add_argument(
        "--lpg-family-two-children",
        type=str,
        default=str(PATH_LPG_FAMILY_TWO_CHILDREN_DEFAULT),
        help=(
            "Absolute path to the LPG SumProfiles file for the 'family_two_children' profile. "
            "Current default: Family with 3 children (1 adult at work, 1 at home) representing "
            "higher daytime electricity consumption patterns."
        ),
    )
    args = parser.parse_args()

    out_path = Path(args.output).expanduser()
    family_two_children_path = Path(args.lpg_family_two_children).expanduser()
    df_master = build_master_dataset(
        output_path=out_path,
        require_complete_features=not args.keep_missing,
        also_write_local_copy=True,
        path_family_two_children=family_two_children_path,
    )
    print(f"Master dataset written to: {out_path}")
    print(f"Shape: {df_master.shape[0]} rows x {df_master.shape[1]} columns")


