"""
src/cleaning/cleaning_pipeline.py
PropML — Data cleaning pipeline (Layer 5).

Input:  data/raw/{city}/magicbricks_{city}_YYYYMMDD.parquet  (13 cities)
Output: data/cleaned/combined_cleaned.parquet

Phase 3 updates:
  - standardize_columns: lat, lng, rera_approved passthrough added
  - fix_data_types: lat/lng cast to float, rera_approved cast to int
  - fix_data_types: plot bhk=0 enforced (not filled to 2)
  - validate_output: Phase 3 column checks added
  - STRING_COLS: no changes needed (lat/lng are numeric)
  - PRICE_PER_SQFT_BOUNDS: all 13 cities already present

DVC stage (dvc.yaml):
  clean:
    cmd: python src/cleaning/cleaning_pipeline.py
    deps: [src/cleaning/cleaning_pipeline.py, data/raw/]
    outs: [data/cleaned/combined_cleaned.parquet]
    params: [params.yaml: cleaning.iqr_multiplier, cleaning.min_price_cr, cleaning.max_price_cr]
"""

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


# ── Area unit conversions to sqft ─────────────────────────────────────────────
AREA_CONVERSIONS = {
    "bigha":  27_225.0,
    "biswa":   1_361.25,
    "sq.mt":      10.7639, "sq.m":  10.7639, "sqm":  10.7639, "sq m": 10.7639,
    "sq.yd":       9.0,    "sqyd":   9.0,    "sq yd": 9.0,    "yard":  9.0,
    "sq-yrd":      9.0,
    "acre":   43_560.0,
    "marla":     272.25,
    "kanal":   5_445.0,
}

# ── City-specific price/sqft sanity bounds (Rs/sqft) ─────────────────────────
PRICE_PER_SQFT_BOUNDS = {
    # NCR / North India
    "gurgaon":    (1_500, 55_000),
    "noida":      (2_000, 30_000),
    "chandigarh": (1_500, 22_000),
    "kota":       (800,   12_000),
    "delhi":      (3_000, 60_000),
    # West India
    "mumbai":     (5_000, 1_20_000),
    "pune":       (3_000, 35_000),
    "ahmedabad":  (1_500, 20_000),
    # South India
    "bangalore":  (3_000, 40_000),
    "chennai":    (2_500, 35_000),
    "hyderabad":  (2_500, 35_000),
    # Other
    "jaipur":     (1_200, 18_000),
    "indore":     (1_000, 15_000),
}

# ── Area bounds by property type (sqft) ──────────────────────────────────────
MIN_AREA_BY_TYPE = {
    "flat":              200,
    "house":             500,
    "independent_floor": 300,
    "plot":               50,   # plots can be small (50 sqft = ~5 sq-yrd minimum)
}
MAX_AREA_BY_TYPE = {
    "flat":              8_000,
    "house":            20_000,
    "independent_floor": 5_000,
    "plot":             50_000,  # larger than flats — 1 acre = 43560 sqft
}

# ── String columns (need non-breaking space stripping) ────────────────────────
# lat/lng/rera_approved are numeric — NOT in this list
STRING_COLS = [
    "floor", "age", "furnish", "sector", "amenities",
    "nearbylocations", "facing", "locality",
]

# ── All supported cities ──────────────────────────────────────────────────────
ALL_CITIES = [
    "gurgaon", "noida", "chandigarh", "kota",
    "delhi", "mumbai", "bangalore", "chennai",
    "pune", "hyderabad", "ahmedabad", "jaipur", "indore",
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_cities(cities: list = None, data_dir: str = "data/raw") -> pd.DataFrame: # type: ignore
    """
    Load and concatenate the latest parquet for each city.

    Picks the most recent magicbricks_{city}_YYYYMMDD.parquet from each
    city's folder. If a city folder has no parquet, it is skipped with
    a warning (not a crash) so the pipeline still runs for other cities.
    """
    if cities is None:
        cities = ALL_CITIES

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
        df   = pd.read_parquet(path)
        df["city"] = city   # Ensure city column is correct
        frames.append(df)
        logger.info(f"  {city}: {len(df)} rows <- {path.name}")

    if not frames:
        logger.error("No data loaded from any city. Run the scraper first.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Total loaded: {len(combined)} rows from {len(frames)} cities")
    logger.info(f"  Cities: {combined['city'].value_counts().to_dict()}")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN STANDARDISATION
# ═══════════════════════════════════════════════════════════════════════════════

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names and rename MagicBricks-specific fields.

    Why: scraper outputs SERVER_PRELOADED_STATE_ field names.
    Cleaning pipeline uses canonical names (bhk, bathroom, etc.).
    All renames happen here — never rename columns in individual functions.

    Phase 3 additions:
      lat, lng         → passed through as-is (float coordinates)
      rera_approved    → passed through as-is (binary int)
    """
    # Lowercase and strip all column names
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    rename_map = {
        # BHK variants
        "bedroom":          "bhk",
        "bedrooms":         "bhk",
        "bedroomd":         "bhk",   # MagicBricks bedroomD field
        "bed":              "bhk",
        # Bathroom variants
        "bath":             "bathroom",
        "baths":            "bathroom",
        "bathd":            "bathroom",   # MagicBricks bathD field
        "washrooms":        "bathroom",
        # Area variants
        "sq_ft":            "area",
        "area_sqft":        "area",
        "sqft":             "area",
        "carpet_area":      "area",
        "ressize":          "area",
        "casqft":           "area",   # MagicBricks caSqFt field
        # Price variants
        "price_in_rs":      "price",
        "total_price":      "price",
        "priced":           "price",   # MagicBricks priceD field
        # Nearby variants
        "nearbylocation":   "nearbylocations",
        "nearby":           "nearbylocations",
        "landmarkdetails":  "nearbylocations",   # MagicBricks landmarkDetails
        # Furnish variants
        "furnishing":       "furnish",
        "furnish_status":   "furnish",
        "furnishstatus":    "furnish",
        "furnishedd":       "furnish",   # MagicBricks furnishedD field
        # Age variants
        "age_of_property":  "age",
        "ageproperty":      "age",
        "possession":       "age",
        "acd":              "age",   # MagicBricks acD field
        # Location variants
        "localityname":     "locality",
        "lmtdname":         "locality",   # MagicBricks lmtDName field
        "societyname":      "sector",
        "prjname":          "sector",   # MagicBricks prjname field
        # Balcony
        "balconiesd":       "balcony",   # MagicBricks balconiesD field
        # Coordinates (Phase 3)
        "latitude":         "lat",
        "longitude":        "lng",
        "lon":              "lng",
    }

    for old_col, new_col in rename_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    # Ensure all expected columns exist (fill with NaN/0 if absent)
    required_after_rename = [
        "bhk", "bathroom", "balcony", "floor", "total_floors",
        "age", "furnish", "amenities", "sector", "nearbylocations",
        "parking", "facing", "is_near_coaching",
    ]
    for col in required_after_rename:
        if col not in df.columns:
            df[col] = np.nan

    # Phase 3 columns — initialize if not present
    if "lat" not in df.columns:
        df["lat"] = 0.0
    if "lng" not in df.columns:
        df["lng"] = 0.0
    if "rera_approved" not in df.columns:
        df["rera_approved"] = 0

    logger.info(f"Columns after standardization: {len(df.columns)}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PARSERS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_price(val) -> Optional[float]:
    """
    Parse price to Crores from any MagicBricks string or numeric format.

    Phase 3: SERVER_PRELOADED_STATE_ sends:
      priceD = "73.41 Cr"  (string, already in Crores)
      price  = 734187000   (int, in paisa)

    Handles: "73.41 Cr", "85 L", raw int rupees (4500000), already-Crore floats.
    Returns None for "Price on Request" and empty values.
    """
    if pd.isna(val):
        return np.nan

    # Numeric: assume paisa if > 1_000_000, else already Crores
    if isinstance(val, (int, float)):
        if val <= 0:
            return np.nan
        if val >= 100_000:
            return round(val / 10_000_000.0, 4)   # paisa to Crores
        return round(float(val), 4)                 # already Crores

    val_str = str(val).lower().replace(",", "").strip()
    # Remove currency symbols
    val_str = re.sub(r"[₹rs\s]", " ", val_str).strip()

    if any(kw in val_str for kw in ["request", "call", "contact"]):
        return np.nan

    match = re.search(r"(\d+\.?\d*)", val_str)
    if not match:
        return np.nan

    num = float(match.group(1))

    if "cr" in val_str or "crore" in val_str:
        return round(num, 4)
    if "lac" in val_str or "lakh" in val_str or "lak" in val_str:
        return round(num / 100.0, 4)
    if re.search(r"\d\s*l\b", val_str) and "lac" not in val_str:
        return round(num / 100.0, 4)
    if "k" in val_str or "thousand" in val_str:
        return round(num / 10_000.0, 4)
    if num >= 100_000:
        return round(num / 10_000_000.0, 4)    # Raw rupees to Cr
    if num < 1_000:
        return round(num, 4)                    # Already in Cr

    return np.nan


def parse_area(val) -> Optional[float]:
    """
    Parse area to sqft from any MagicBricks format including unit conversions.

    Phase 3: plots use sq-yrd (la field). Scraper already converts la to sqft
    using parse_area(la, unit_hint="sq-yrd"), so by cleaning time area is sqft.
    But if raw sq-yrd strings slip through, conversion handles them.
    """
    if pd.isna(val):
        return np.nan

    # Numeric already (from scraper)
    if isinstance(val, (int, float)):
        return round(float(val), 2) if val > 0 else np.nan

    val_str = str(val).lower().replace(",", "").strip()

    # Handle range "1200 - 1500 sqft" -> midpoint
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
    Parse floor string -> (floor_pos: int, total_floors: int).

    Handles non-breaking spaces \xa0, "Ground", "6+", "5 out of 12".
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


def parse_bhk(val, property_type: str = "flat") -> int:
    """
    Parse BHK value to int.

    Phase 3: plots must have bhk=0. Plots should already have bhk=0
    from scraper, but this ensures it regardless of input.
    """
    if property_type == "plot":
        return 0
    if pd.isna(val):
        return 2
    # If already 0 (plot), keep it
    try:
        int_val = int(float(str(val)))
        if int_val == 0:
            return 0
    except (ValueError, TypeError):
        pass
    m = re.search(r"(\d+)", str(val))
    return int(m.group(1)) if m else 2


# ═══════════════════════════════════════════════════════════════════════════════
# PARSING STAGE
# ═══════════════════════════════════════════════════════════════════════════════

def parse_raw_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all string parsers to convert raw columns to numeric types.

    Drops rows where price or area cannot be parsed — these are
    fundamentally unusable for training.

    Phase 3: parse_bhk now receives property_type to handle plot bhk=0.
    """
    logger.info("Parsing raw strings...")
    initial = len(df)

    df["price"] = df["price"].apply(parse_price)
    df["area"]  = df["area"].apply(parse_area)

    # Parse floor string -> two numeric columns
    floor_parsed          = df["floor"].apply(parse_floor)
    df["floor_pos"]       = floor_parsed.apply(lambda x: x[0])
    df["total_floors_parsed"] = floor_parsed.apply(lambda x: x[1])

    # Use existing total_floors if valid, otherwise use parsed value
    df["total_floors"] = pd.to_numeric(df["total_floors"], errors="coerce")
    df["total_floors"] = df["total_floors"].fillna(df["total_floors_parsed"]).astype(int)
    df.drop(columns=["total_floors_parsed"], inplace=True)

    # Phase 3: pass property_type to parse_bhk so plots get bhk=0
    if "property_type" in df.columns:
        df["bhk"] = df.apply(
            lambda row: parse_bhk(row["bhk"], row.get("property_type", "flat")),
            axis=1
        )
    else:
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


# ═══════════════════════════════════════════════════════════════════════════════
# BUSINESS LOGIC FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

def apply_business_logic_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that violate real-estate domain constraints.

    Why this before IQR outlier removal:
      IQR is statistical — it operates on the distribution.
      Business logic is deterministic — a flat with 21 bedrooms
      is ALWAYS wrong regardless of the distribution.
      Business logic runs first, then IQR operates on a cleaner set.

    Filters applied:
      1. BHK range: 0-10 (0 is valid for plots)
      2. Area range: per property_type (see MIN/MAX_AREA_BY_TYPE)
      3. Price/sqft: city-specific bounds on residential properties
         Plots are excluded from price/sqft check (no per-sqft pricing)

    Phase 3: plot area bounds updated (MIN=50, MAX=50000 sqft).
    """
    initial = len(df)

    # ── BHK range ─────────────────────────────────────────────────────
    # 0 is valid (plots), max is 10
    df = df[df["bhk"].between(0, 10)]

    # ── Area range by property type ───────────────────────────────────
    for prop_type, min_area in MIN_AREA_BY_TYPE.items():
        max_area = MAX_AREA_BY_TYPE[prop_type]
        mask = df["property_type"] == prop_type
        df   = df[~mask | df["area"].between(min_area, max_area)]

    # ── Price/sqft sanity (residential types only, NOT plots) ─────────
    residential = ["flat", "house", "independent_floor"]
    df_res      = df[df["property_type"].isin(residential)].copy()
    df_plots    = df[~df["property_type"].isin(residential)].copy()

    if len(df_res) > 0:
        df_res["_ppsf"] = df_res["price"] * 1e7 / df_res["area"].clip(lower=1)
        valid_mask      = pd.Series(True, index=df_res.index)

        for city, (lo, hi) in PRICE_PER_SQFT_BOUNDS.items():
            city_mask   = df_res["city"] == city
            bounds_mask = df_res["_ppsf"].between(lo, hi)
            valid_mask[city_mask] = bounds_mask[city_mask]

        removed_ppsf = (~valid_mask).sum()
        df_res       = df_res[valid_mask].drop(columns=["_ppsf"])
        logger.info(f"  Price/sqft filter removed {removed_ppsf} rows")

    df = pd.concat([df_res, df_plots], ignore_index=True)

    removed = initial - len(df)
    logger.info(
        f"Business filters: removed {removed} rows ({removed / initial * 100:.1f}%). "
        f"Remaining: {len(df)}"
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# CITY TIER
# ═══════════════════════════════════════════════════════════════════════════════

def create_city_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each property into Tier-1 / Tier-2 / Tier-3 within its city.

    Method: within-city price tertiles
      Tier-1 = top 33% by price (luxury)
      Tier-2 = middle 33% (mid-market)
      Tier-3 = bottom 33% (budget)

    Why within-city and not global:
      A Rs1.5Cr flat is Tier-1 in Kota but Tier-3 in Gurgaon.
      Global classification conflates different markets.

    Why this exists:
      Layer 7 assigns higher sample weights to Tier-1 properties.
      Layer 6 uses city_tier_num as a feature.
      Grouped IQR uses tiers to avoid removing luxury as "outliers".
    """
    logger.info("Creating city tier classification...")
    df["city_tier"] = "Tier-2"  # Default

    for city in df["city"].unique():
        mask   = df["city"] == city
        prices = df.loc[mask, "price"]
        q33    = prices.quantile(0.33) # type: ignore
        q67    = prices.quantile(0.67) # type: ignore

        df.loc[mask & (df["price"] <= q33),                           "city_tier"] = "Tier-3"
        df.loc[mask & (df["price"] >  q33) & (df["price"] <= q67),   "city_tier"] = "Tier-2"
        df.loc[mask & (df["price"] >  q67),                           "city_tier"] = "Tier-1"

        dist = df.loc[mask, "city_tier"].value_counts().to_dict() # type: ignore
        logger.info(f"  {city.upper()}: {dist}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL OUTLIER REMOVAL
# ═══════════════════════════════════════════════════════════════════════════════

def remove_structural_outliers(df: pd.DataFrame, iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove statistical outliers using grouped IQR.

    Groups: city x property_type x city_tier
    For each group, compute IQR bounds on price and area separately.
    Remove rows outside bounds IN THAT GROUP ONLY.

    Why grouped, not global:
      Global IQR on merged data where flats outnumber houses 3:1
      pulls the upper price fence DOWN, falsely flagging Rs10Cr luxury
      houses as outliers. Grouped IQR applies separate bounds per sub-market.

    Minimum group size: 10 rows (smaller groups skipped).
    """
    logger.info("Removing structural outliers (grouped IQR)...")
    initial    = len(df)
    valid_mask = pd.Series(True, index=df.index)
    stats_log  = []

    for city in df["city"].unique():
        for prop_type in df["property_type"].unique():
            for tier in df["city_tier"].unique():

                group_mask = (
                    (df["city"]          == city) &
                    (df["property_type"] == prop_type) &
                    (df["city_tier"]     == tier)
                )

                subset = df[group_mask]
                if len(subset) < 10:
                    continue

                # ── Price IQR bounds ──────────────────────────────────
                q1_p, q3_p = subset["price"].quantile([0.25, 0.75])
                iqr_p  = q3_p - q1_p
                lo_p   = max(q1_p - iqr_multiplier * iqr_p, 0.01)
                hi_p   = q3_p + iqr_multiplier * iqr_p

                # ── Area IQR bounds ───────────────────────────────────
                q1_a, q3_a = subset["area"].quantile([0.25, 0.75])
                iqr_a  = q3_a - q1_a
                min_a  = MIN_AREA_BY_TYPE.get(prop_type, 200)
                lo_a   = max(q1_a - iqr_multiplier * iqr_a, min_a)
                hi_a   = q3_a + iqr_multiplier * iqr_a

                # ── Apply bounds ──────────────────────────────────────
                keep = (
                    (df["price"] >= lo_p) & (df["price"] <= hi_p) &
                    (df["area"]  >= lo_a) & (df["area"]  <= hi_a)
                )
                valid_mask[group_mask] = keep[group_mask]

                removed = len(subset) - keep[group_mask].sum()
                if removed > 0:
                    stats_log.append(
                        f"  {city.upper()} | {tier} | {prop_type}: "
                        f"-{removed}/{len(subset)}"
                    )

    for line in stats_log:
        logger.info(line)

    df_clean      = df[valid_mask].copy()
    total_removed = initial - len(df_clean)
    pct           = total_removed / initial * 100
    logger.info(
        f"Grouped IQR: removed {total_removed} ({pct:.1f}%). "
        f"Remaining: {len(df_clean)}"
    )
    return df_clean


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE FIXING
# ═══════════════════════════════════════════════════════════════════════════════

def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final type casting and cleaning after all filter stages.

    Handles:
    - Non-breaking spaces in string columns (\xa0 from MagicBricks HTML)
    - 'nan' string values that should be NaN
    - Numeric columns clipped to valid ranges and cast to correct dtype
    - locality column aliased from sector for synthetic cities

    Phase 3 additions:
    - lat, lng: cast to float, 0.0 for missing
    - rera_approved: cast to int binary (0 or 1)
    - plot bhk: enforce 0 (not filled to 2 like other property types)
    """
    # ── String columns: strip non-breaking spaces ──────────────────────
    for col in STRING_COLS:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                       .str.replace("\xa0", " ", regex=False)
                       .str.strip()
            )
            df[col] = df[col].replace("nan", np.nan)

    # ── BHK — plots stay at 0, others fill NaN with 2 ─────────────────
    df["bhk"] = pd.to_numeric(df["bhk"], errors="coerce")
    if "property_type" in df.columns:
        plot_mask          = df["property_type"] == "plot"
        df.loc[plot_mask,  "bhk"] = 0
        df.loc[~plot_mask, "bhk"] = df.loc[~plot_mask, "bhk"].fillna(2)
    else:
        df["bhk"] = df["bhk"].fillna(2)
    df["bhk"] = df["bhk"].clip(0, 10).astype(int)

    # ── Other numeric columns ──────────────────────────────────────────
    df["bathroom"] = pd.to_numeric(df["bathroom"], errors="coerce")
    df["bathroom"] = df["bathroom"].fillna(df["bhk"]).clip(0, 8).astype(int)

    df["balcony"]  = pd.to_numeric(df["balcony"],  errors="coerce").fillna(0).clip(0, 5).astype(int)
    df["parking"]  = pd.to_numeric(df["parking"],  errors="coerce").fillna(0).clip(0, 5).astype(int)

    df["floor_pos"]    = pd.to_numeric(df["floor_pos"],    errors="coerce").fillna(0).clip(0, 60).astype(int)
    df["total_floors"] = pd.to_numeric(df["total_floors"], errors="coerce").fillna(5).clip(1, 60).astype(int)

    df["is_near_coaching"] = pd.to_numeric(
        df.get("is_near_coaching", 0), errors="coerce"
    ).fillna(0).astype(int) # type: ignore

    # ── Phase 3: lat/lng ───────────────────────────────────────────────
    df["lat"] = pd.to_numeric(df.get("lat", 0), errors="coerce").fillna(0.0).astype(float) # type: ignore
    df["lng"] = pd.to_numeric(df.get("lng", 0), errors="coerce").fillna(0.0).astype(float) # type: ignore

    # Sanity: lat must be between 8 and 37 (India range), else 0.0
    bad_lat = ~df["lat"].between(8.0, 37.0) & (df["lat"] != 0.0)
    bad_lng = ~df["lng"].between(68.0, 97.0) & (df["lng"] != 0.0)
    df.loc[bad_lat, "lat"] = 0.0
    df.loc[bad_lng, "lng"] = 0.0

    lat_filled = (df["lat"] != 0.0).sum()
    logger.info(f"  lat/lng: {lat_filled}/{len(df)} rows have valid coordinates")

    # ── Phase 3: rera_approved ────────────────────────────────────────
    df["rera_approved"] = pd.to_numeric(
        df.get("rera_approved", 0), errors="coerce"
    ).fillna(0).clip(0, 1).astype(int) # type: ignore

    # ── locality column — alias for sector if missing ─────────────────
    if "locality" not in df.columns:
        df["locality"] = df.get("sector", "Unknown")

    logger.info(f"Data types fixed. Shape: {df.shape}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TARGET CREATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create log-transformed price target column.

    log_price = log1p(price)  where price is in Crores.

    Why log transform:
      Price range: Rs0.15Cr - Rs35Cr = 233x range
      log1p(0.15)=0.14, log1p(35)=3.58 -> 25x range
      Log-space makes the target approximately Gaussian.
      Also makes MAPE and RMSE comparable across price segments.

    At inference time:
      predicted_price_cr = expm1(model.predict(X))
    """
    df["log_price"] = np.log1p(df["price"])
    logger.info(
        f"Target created: log_price — "
        f"mean={df['log_price'].mean():.3f}, "
        f"std={df['log_price'].std():.3f}, "
        f"range=[{df['log_price'].min():.2f}, {df['log_price'].max():.2f}]"
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_output(df: pd.DataFrame) -> bool:
    """
    Post-cleaning sanity check. Runs AFTER all cleaning stages.

    Returns True if output is valid. Logs all issues found.
    Does not stop the pipeline — cleaning may have partial data
    that is still useful for training.

    Phase 3 additions:
    - Checks lat/lng are float
    - Checks rera_approved is binary
    - Checks plot bhk=0
    """
    logger.info("\nValidating cleaned output...")
    issues = []

    # Required columns
    required = [
        "city", "property_type", "price", "area", "bhk",
        "log_price", "city_tier", "floor_pos", "total_floors",
    ]
    for col in required:
        if col not in df.columns:
            issues.append(f"MISSING column: {col}")

    # No NaN in critical columns
    for col in ["price", "area", "log_price"]:
        n = df[col].isna().sum() if col in df.columns else -1
        if n > 0:
            issues.append(f"NaN in {col}: {n} rows")

    # Price range
    if "price" in df.columns:
        if df["price"].min() < 0.01:
            issues.append(f"Price below Rs1L: {(df['price'] < 0.01).sum()} rows")
        if df["price"].max() > 150:
            issues.append(f"Price above Rs150Cr: {(df['price'] > 150).sum()} rows")

    # Property types
    known_types = {"flat", "house", "independent_floor", "plot"}
    if "property_type" in df.columns:
        unknown = set(df["property_type"].unique()) - known_types
        if unknown:
            issues.append(f"Unknown property_type values: {unknown}")

    # Phase 3: plot bhk must be 0
    if "property_type" in df.columns and "bhk" in df.columns:
        plots         = df[df["property_type"] == "plot"]
        if len(plots) > 0:
            nonzero_bhk = (plots["bhk"] != 0).sum()
            if nonzero_bhk > 0:
                issues.append(f"Plots with bhk != 0: {nonzero_bhk} rows")

    # Phase 3: lat/lng dtype
    if "lat" in df.columns and df["lat"].dtype not in [np.float32, np.float64, float]:
        issues.append(f"lat column dtype is {df['lat'].dtype} (expected float)")
    if "lng" in df.columns and df["lng"].dtype not in [np.float32, np.float64, float]:
        issues.append(f"lng column dtype is {df['lng'].dtype} (expected float)")

    # Phase 3: rera_approved binary
    if "rera_approved" in df.columns:
        unique_vals = set(df["rera_approved"].unique())
        if not unique_vals.issubset({0, 1}):
            issues.append(f"rera_approved has non-binary values: {unique_vals}")

    if issues:
        for issue in issues:
            logger.error(f"  VALIDATION FAIL: {issue}")
        return False

    logger.info(f"  All validation checks passed. {len(df)} clean rows.")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def clean_pipeline(
    cities: list = None, # type: ignore
    output_path: str = "data/cleaned/combined_cleaned.parquet",
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Full cleaning pipeline — runs all stages in order.

    Reads params.yaml for iqr_multiplier if available.
    Saves output to data/cleaned/combined_cleaned.parquet.

    Phase 3: passes lat/lng and rera_approved through all stages.
    """
    # Read params from params.yaml if available
    params_path = Path("params.yaml")
    if params_path.exists():
        with open(params_path) as f:
            params = yaml.safe_load(f)
        iqr_multiplier = params.get("cleaning", {}).get("iqr_multiplier", iqr_multiplier)

    logger.info("=" * 60)
    logger.info("Layer 5: Data Cleaning Pipeline")
    logger.info(f"  IQR multiplier: {iqr_multiplier}")
    logger.info("=" * 60)

    df = load_all_cities(cities)
    df = standardize_columns(df)
    df = parse_raw_strings(df)
    df = apply_business_logic_filters(df)
    df = create_city_tier(df)
    df = remove_structural_outliers(df, iqr_multiplier)
    df = fix_data_types(df)
    df = create_target(df)

    is_valid = validate_output(df)
    if not is_valid:
        logger.warning("Output validation issues found. Proceeding anyway.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    logger.info(f"\nSaved {len(df)} clean rows -> {output_path}")
    logger.info(f"   Cities:    {df['city'].value_counts().to_dict()}")
    logger.info(f"   Types:     {df['property_type'].value_counts().to_dict()}")
    logger.info(f"   Tiers:     {df['city_tier'].value_counts().to_dict()}")
    logger.info(f"   Price:     Rs{df['price'].min():.2f}Cr - Rs{df['price'].max():.2f}Cr")
    logger.info(f"   Area:      {df['area'].min():.0f} - {df['area'].max():.0f} sqft")

    # Phase 3 summary
    lat_filled  = (df["lat"] != 0.0).sum()
    rera_filled = df["rera_approved"].sum()
    logger.info(f"   Coords:    {lat_filled}/{len(df)} rows have lat/lng")
    logger.info(f"   RERA:      {rera_filled}/{len(df)} rows RERA approved ({rera_filled/len(df)*100:.1f}%)")

    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PropML Cleaning Pipeline")
    parser.add_argument("--cities", nargs="+", default=None,
                        help=f"Cities to process (default: all 13). Available: {ALL_CITIES}")
    parser.add_argument("--output", default="data/cleaned/combined_cleaned.parquet",
                        help="Output parquet path")
    args = parser.parse_args()
    clean_pipeline(cities=args.cities, output_path=args.output)


if __name__ == "__main__":
    main()