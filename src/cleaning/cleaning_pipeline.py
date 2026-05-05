import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Area unit conversions to sqft ───────────────────────────────────────────
AREA_CONVERSIONS = {
    "bigha":  27_225.0,
    "biswa":   1_361.25,
    "sq.mt":      10.7639, "sq.m":  10.7639, "sqm":  10.7639, "sq m": 10.7639,
    "sq.yd":       9.0,    "sqyd":   9.0,    "sq yd": 9.0,    "yard":  9.0,
    "acre":   43_560.0,
    "marla":     272.25,
    "kanal":   5_445.0,
}

# ─── City-specific price/sqft sanity bounds (₹/sqft) ─────────────────────────
PRICE_PER_SQFT_BOUNDS = {
    "gurgaon":    (1_500,  45_000),
    "noida":      (2_000,  28_000),
    "chandigarh": (1_500,  22_000),
    "kota":         (800,  12_000),
}

# ─── Area bounds by property type (sqft) ─────────────────────────────────────
MIN_AREA_BY_TYPE = {"flat": 200,  "house": 500, "independent_floor": 300, "plot": 50}
MAX_AREA_BY_TYPE = {"flat": 8_000, "house": 20_000, "independent_floor": 5_000, "plot": 5_000}

# ─── Columns that need non-breaking space stripping ───────────────────────────
STRING_COLS = ["floor", "age", "furnish", "sector", "amenities",
               "nearbylocations", "facing", "locality"]





def load_all_cities(cities: list = None, data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load and concatenate the latest parquet for each city.

    Picks the most recent magicbricks_{city}_YYYYMMDD.parquet from each
    city's folder. If a city folder has no parquet, it is skipped with
    a warning (not a crash) so the pipeline still runs for other cities.
    """
    if cities is None:
        cities = ["gurgaon", "noida", "chandigarh", "kota"]

    logger.info("=" * 60)
    logger.info("Loading raw data from all cities...")

    frames = []
    for city in cities:
        city_dir = Path(data_dir) / city
        parquets = sorted(city_dir.glob("magicbricks_*.parquet"), reverse=True)

        if not parquets:
            logger.warning(f"  {city}: no parquet found in {city_dir} — skipping")
            continue

        path = parquets[0]
        df = pd.read_parquet(path)
        df["city"] = city   # Ensure city column is correct
        frames.append(df)
        logger.info(f"  {city}: {len(df)} rows ← {path.name}")

    if not frames:
        logger.error("No data loaded from any city. Run the scraper first.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Total loaded: {len(combined)} rows from {len(frames)} cities")
    logger.info(f"  Cities: {combined['city'].value_counts().to_dict()}")
    return combined










def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and rename MagicBricks-specific fields.

    Why: scraper outputs 'bedrooms', cleaning pipeline expects 'bhk'.
    All renames happen here — never change column names in individual functions.
    """
    # Lowercase and strip all column names
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    rename_map = {
        "bedroom":          "bhk",
        "bedrooms":         "bhk",
        "bed":              "bhk",
        "bath":             "bathroom",
        "baths":            "bathroom",
        "washrooms":        "bathroom",
        "sq_ft":            "area",
        "area_sqft":        "area",
        "sqft":             "area",
        "carpet_area":      "area",
        "ressize":          "area",
        "price_in_rs":      "price",
        "total_price":      "price",
        "nearbylocation":   "nearbylocations",
        "nearby":           "nearbylocations",
        "furnishing":       "furnish",
        "furnish_status":   "furnish",
        "furnishstatus":    "furnish",
        "age_of_property":  "age",
        "ageproperty":      "age",
        "possession":       "age",
        "localityname":     "locality",
        "societyname":      "sector",
    }

    for old_col, new_col in rename_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    # Ensure all expected columns exist (fill with NaN if absent)
    required_after_rename = [
        "bhk", "bathroom", "balcony", "floor", "total_floors",
        "age", "furnish", "amenities", "sector", "nearbylocations",
        "parking", "facing", "is_near_coaching",
    ]
    for col in required_after_rename:
        if col not in df.columns:
            df[col] = np.nan

    logger.info(f"Columns after standardization: {len(df.columns)}")
    return df











def parse_price(val) -> Optional[float]:
    """
    Parse price to Crores from any MagicBricks string format.

    Handles: "₹ 1.25 Cr", "₹ 85 L", raw int rupees (4500000), already-Crore floats.
    Returns None for "Price on Request" and empty values.
    """
    if pd.isna(val):
        return np.nan

    val_str = str(val).lower().replace("₹", "").replace(",", "").strip()

    if any(kw in val_str for kw in ["request", "call", "contact"]):
        return np.nan

    match = re.search(r"(\d+\.?\d*)", val_str)
    if not match:
        return np.nan

    num = float(match.group(1))

    if "cr" in val_str or "crore" in val_str:
        return num
    if "lac" in val_str or "lakh" in val_str or "lak" in val_str:
        return round(num / 100.0, 4)
    if re.search(r"\d\s*l\b", val_str) and "lac" not in val_str:
        return round(num / 100.0, 4)
    if "k" in val_str or "thousand" in val_str:
        return round(num / 10_000.0, 4)
    if num >= 100_000:
        return round(num / 10_000_000.0, 4)    # Raw rupees → Cr
    if num < 1_000:
        return round(num, 4)                    # Already in Cr

    return np.nan


def parse_area(val) -> Optional[float]:
    """Parse area to sqft from any MagicBricks format including unit conversions."""
    if pd.isna(val):
        return np.nan

    val_str = str(val).lower().replace(",", "").strip()

    # Handle range "1200 - 1500 sqft" → midpoint
    range_m = re.search(r"(\d+\.?\d*)\s*[-–to]+\s*(\d+\.?\d*)", val_str)
    if range_m:
        num = (float(range_m.group(1)) + float(range_m.group(2))) / 2
    else:
        match = re.search(r"(\d+\.?\d*)", val_str)
        if not match:
            return np.nan
        num = float(match.group(1))

    if num <= 0:
        return np.nan

    for unit, factor in AREA_CONVERSIONS.items():
        if unit in val_str:
            return round(num * factor, 2)

    return round(num, 2)  # Default: sqft


def parse_floor(val) -> tuple:
    """
    Parse floor string → (floor_pos: int, total_floors: int).

    Handles non-breaking spaces \\xa0, "Ground", "6+", "5 out of 12".
    """
    if pd.isna(val):
        return 0, 5

    val_str = str(val).replace("\xa0", " ").replace("\u00a0", " ").lower().strip()

    m = re.search(r"(ground|\d+)\s*(?:out of|of|/)\s*(\d+)", val_str)
    if m:
        fp = 0 if m.group(1) == "ground" else int(m.group(1))
        tf = int(m.group(2))
        return fp, max(tf, fp + 1)

    if re.fullmatch(r"(ground|g|gf|ground floor|gr)", val_str):
        return 0, 1

    if "+" in val_str:
        n = re.search(r"(\d+)", val_str)
        if n:
            fp = int(n.group(1))
            return fp, max(fp + 4, 10)

    n = re.search(r"(\d+)", val_str)
    if n:
        fp = int(n.group(1))
        return fp, max(fp + 2, 5)

    return 0, 5


def parse_bhk(val) -> int:
    """Parse BHK value to int. Handles '3 BHK', '3', 3."""
    if pd.isna(val):
        return 2
    m = re.search(r"(\d+)", str(val))
    return int(m.group(1)) if m else 2









def parse_raw_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all string parsers to convert raw columns to numeric types.

    Drops rows where price or area cannot be parsed — these are
    fundamentally unusable for training.
    """
    logger.info("Parsing raw strings...")
    initial = len(df)

    df["price"] = df["price"].apply(parse_price)
    df["area"]  = df["area"].apply(parse_area)

    # Parse floor string → two numeric columns
    floor_parsed = df["floor"].apply(parse_floor)
    df["floor_pos"]           = floor_parsed.apply(lambda x: x[0])
    df["total_floors_parsed"] = floor_parsed.apply(lambda x: x[1])

    # Use existing total_floors if valid, otherwise use parsed value
    df["total_floors"] = pd.to_numeric(df["total_floors"], errors="coerce")
    df["total_floors"] = df["total_floors"].fillna(df["total_floors_parsed"]).astype(int)
    df.drop(columns=["total_floors_parsed"], inplace=True)

    df["bhk"] = df["bhk"].apply(parse_bhk)

    # Drop rows with unparseable price or area
    before_drop = len(df)
    df = df.dropna(subset=["price", "area"])
    df = df[df["price"] > 0]
    df = df[df["area"]  > 0]

    dropped = before_drop - len(df)
    logger.info(f"  Dropped {dropped} rows with unparseable price/area")
    logger.info(f"  Remaining: {len(df)} rows")
    return df









def apply_business_logic_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that violate real-estate domain constraints.

    Why this before IQR outlier removal:
      IQR is statistical — it operates on the distribution.
      Business logic is deterministic — a flat with 21 bedrooms
      is ALWAYS wrong regardless of the distribution.
      Business logic runs first, then IQR operates on a cleaner set.

    Filters applied:
      1. BHK range: 0–10 (flats), 0–15 (houses)
      2. Area range: per property_type (see MIN/MAX_AREA_BY_TYPE)
      3. Price/sqft: city-specific bounds on residential properties
         Plots are excluded from price/sqft check (no per-sqft pricing)
    """
    initial = len(df)

    # ── BHK range ────────────────────────────────────────────────────
    df = df[df["bhk"].between(0, 10)]

    # ── Area range by property type ──────────────────────────────────
    for prop_type, min_area in MIN_AREA_BY_TYPE.items():
        max_area = MAX_AREA_BY_TYPE[prop_type]
        mask = df["property_type"] == prop_type
        df = df[~mask | df["area"].between(min_area, max_area)]

    # ── Price/sqft sanity (residential types only) ───────────────────
    residential = ["flat", "house", "independent_floor"]
    df_res   = df[df["property_type"].isin(residential)].copy()
    df_plots = df[~df["property_type"].isin(residential)].copy()

    if len(df_res) > 0:
        df_res["_ppsf"] = df_res["price"] * 1e7 / df_res["area"].clip(lower=1)
        valid_mask = pd.Series(True, index=df_res.index)

        for city, (lo, hi) in PRICE_PER_SQFT_BOUNDS.items():
            city_mask   = df_res["city"] == city
            bounds_mask = df_res["_ppsf"].between(lo, hi)
            valid_mask[city_mask] = bounds_mask[city_mask]

        removed_ppsf = (~valid_mask).sum()
        df_res = df_res[valid_mask].drop(columns=["_ppsf"])
        logger.info(f"  Price/sqft filter removed {removed_ppsf} rows")

    df = pd.concat([df_res, df_plots], ignore_index=True)

    removed = initial - len(df)
    logger.info(
        f"Business filters: removed {removed} rows ({removed / initial * 100:.1f}%). "
        f"Remaining: {len(df)}"
    )
    return df