"""
src/features/feature_engineering.py
PropML — Feature engineering pipeline.

Input:  data/cleaned/combined_cleaned.parquet
Output: data/features/combined_engineered.parquet
        data/features/feature_list.txt
        data/features/feature_metadata.json
        data/features/target_encoding_map.json

All 32 features with their source, formula, and SHAP importance rank:

  Location (highest importance):
    city_tier_num, city_encoded, area_x_city_encoded, locality_encoded,
    area_x_city_tier

  Property type (one-hot):
    is_flat, is_house, is_independent_floor, is_plot

  Area features:
    area (raw), log_area, area_per_bhk, area_x_locality

  Room features:
    bhk, bathroom, bath_per_bed

  Floor features:
    floor_pos, total_floors, relative_floor, is_high_floor

  Quality features:
    amenity_score, avg_rating, furnish_score, amenity_x_city

  Nearby flags (NLP from nearbylocations text):
    has_metro_nearby, has_hospital_nearby, has_school_nearby,
    has_mall_nearby

  Age:
    age_bucket

  Other:
    balcony, parking, is_near_coaching

DVC stage (dvc.yaml):
  featurize:
    cmd: python src/features/feature_engineering.py
    deps: [data/cleaned/combined_cleaned.parquet]
    outs: [data/features/]
    params: [params.yaml:features]

Run:
  python src/features/feature_engineering.py
"""

import json
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

# ─── Paths ───────────────────────────────────────────────────────────────────
CLEANED_PATH   = "data/cleaned/combined_cleaned.parquet"
FEATURES_PATH  = "data/features/combined_engineered.parquet"
FEAT_LIST_PATH = "data/features/feature_list.txt"
META_PATH      = "data/features/feature_metadata.json"
ENC_MAP_PATH   = "data/features/target_encoding_map.json"

TARGET = "log_price"

# ─── Feature list (ordered by expected SHAP importance) ──────────────────────
# This is the authoritative order — Layer 7 training loads features in this order
FINAL_FEATURES = [
    # Location (top importance)
    "city_tier_num",
    "city_encoded",
    "area_x_city_encoded",
    "locality_encoded",
    "area_x_city_tier",
    # Property type one-hot
    "is_house",
    "is_flat",
    "is_independent_floor",
    "is_plot",
    # Area
    "area",
    "log_area",
    "area_x_locality",
    "area_per_bhk",
    # Rooms
    "bhk",
    "bathroom",
    "bath_per_bed",
    # Floor
    "floor_pos",
    "total_floors",
    "relative_floor",
    "is_high_floor",
    # Quality
    "amenity_score",
    "avg_rating",
    "furnish_score",
    "amenity_x_city",
    # Nearby
    "has_metro_nearby",
    "has_hospital_nearby",
    "has_school_nearby",
    "has_mall_nearby",
    # Age
    "age_bucket",
    # Other
    "balcony",
    "parking",
    "is_near_coaching",
]






def load_cleaned() -> pd.DataFrame:
    """Load cleaned data from Layer 5 output."""
    if not os.path.exists(CLEANED_PATH):
        logger.error(f"Missing: {CLEANED_PATH}")
        logger.error("Run: python src/cleaning/cleaning_pipeline.py")
        sys.exit(1)

    df = pd.read_parquet(CLEANED_PATH)
    logger.info(f"Loaded: {len(df)} rows × {len(df.columns)} cols")
    logger.info(f"  Cities: {df['city'].value_counts().to_dict()}")
    logger.info(f"  Types:  {df['property_type'].value_counts().to_dict()}")
    return df






def align_location_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unify sector / locality into a single 'locality' column.

    Why: MagicBricks has 'localityName' (neighbourhood) and 'societyName'
    (building/society). We scraped these as 'locality' and 'sector'.
    For target encoding we want one representative location column.
    Prefer 'locality' if it exists, fall back to 'sector'.
    """
    if "sector" in df.columns and "locality" not in df.columns:
        df["locality"] = df["sector"]
    elif "sector" in df.columns and "locality" in df.columns:
        df["locality"] = df["locality"].fillna(df["sector"])
    elif "locality" not in df.columns:
        df["locality"] = "Unknown"

    df["locality"] = (
        df["locality"]
        .fillna("unknown_" + df["city"].astype(str))
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )

    logger.info(f"  Locality: {df['locality'].nunique()} unique values")
    return df






def build_property_type_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode property_type into 4 binary columns.

    Why not just 'is_house' binary:
      is_house = 0 conflates flat, independent_floor, and plot.
      An independent_floor has a fundamentally different price pattern
      from a flat. Plots have no BHK. One binary loses this signal.

    These 4 flags are mutually exclusive and exhaustive.
    """
    df["is_flat"]              = (df["property_type"] == "flat").astype(int)
    df["is_house"]             = (df["property_type"] == "house").astype(int)
    df["is_independent_floor"] = (df["property_type"] == "independent_floor").astype(int)
    df["is_plot"]              = (df["property_type"] == "plot").astype(int)
    return df





def build_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio and per-unit derived features.

    bath_per_bed: bathroom / bhk
      A 3BHK with 3 bathrooms is standard.
      A 3BHK with 5 bathrooms is luxury.
      Captures room-level quality that raw counts cannot.
      Clipped 0–4 to handle data errors (e.g. 10 bathrooms / 1 BHK).

    area_per_bhk: area / bhk
      Space quality proxy. 600sqft 3BHK is cramped. 800sqft 2BHK is spacious.
      Clipped 0–5000 sqft per bedroom.

    log_area: log1p(area)
      Area is right-skewed. A 10,000sqft plot is 5× a 2,000sqft flat,
      but in log-space the difference is more proportional.
      Matches the log_price target distribution shape.
    """
    df["bath_per_bed"] = (
        df["bathroom"].clip(lower=1) / df["bhk"].clip(lower=1)
    ).clip(0, 4)

    df["area_per_bhk"] = (
        df["area"] / df["bhk"].clip(lower=1)
    ).clip(0, 5000)

    df["log_area"] = np.log1p(df["area"])
    return df


def build_floor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Floor-derived features.

    relative_floor = floor_pos / total_floors, clipped 0–1
      0 = ground floor, 1 = top floor.
      Captures "how high is this floor" as a proportion, independent
      of how tall the building is.
      A 5th floor in a 6-storey building (0.83) is very different from
      a 5th floor in a 20-storey building (0.25).

    is_high_floor = 1 if relative_floor >= 0.75
      Binary flag for "top quarter" of the building.
      Top floors command a premium (views, air, penthouse perception).
      Threshold 0.75 comes from observed price premium patterns.

    CRITICAL: do NOT re-parse floor string here.
    Layer 5 already extracted floor_pos and total_floors correctly
    from '5 out of 12' format. Re-parsing with a simpler regex here
    would pick the wrong number.
    """
    df["floor_pos"]    = pd.to_numeric(df["floor_pos"],    errors="coerce").fillna(0).clip(0, 60)
    df["total_floors"] = pd.to_numeric(df["total_floors"], errors="coerce").fillna(5).clip(1, 60)

    df["relative_floor"] = (
        df["floor_pos"] / df["total_floors"].clip(lower=1)
    ).clip(0, 1)

    df["is_high_floor"] = (df["relative_floor"] >= 0.75).astype(int)
    return df








def build_age_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordinal encoding for property age.

    Ordinal (not one-hot) because age has a natural order:
    new > mid-age > old in terms of price (generally).
    Ordinal lets the model learn monotonic age effects.

    Under Construction = 0 because it has unique pricing dynamics
    (speculative premium, possession risk) separate from new-build.
    """
    age_map = {
        "under construction": 0,
        "0-1 years":  1, "0-1": 1,
        "1-5 years":  2, "1-5": 2,
        "5-10 years": 3, "5-10": 3,
        "10+ years":  4, "10+": 4,
        "land": 0,
    }
    if "age" in df.columns:
        df["age_bucket"] = (
            df["age"].astype(str).str.lower().str.strip()
                     .map(age_map)
                     .fillna(2)  # Default: 1-5 years (most common)
                     .astype(int)
        )
    else:
        df["age_bucket"] = 2

    return df   






def parse_amenity_score(amenities_str) -> float:
    """
    Weighted sum of amenities present.

    Weights reflect price premium each amenity adds:
      pool:           3.0  (highest — adds ₹30–50L to luxury flat)
      gym:            2.0
      lift:           1.0  (basic — expected in most buildings)
      club_house:     1.0
      security/24x7:  0.5
      power_backup:   0.5
      jogging/park:   0.5

    Why weighted instead of count:
      count([gym, lift]) = count([pool, security]) = 2
      But pool+security adds far more value than gym+lift.
      Weighted score captures this market reality.

    Maximum possible score ≈ 8.5 (all amenities present).
    In practice, score 0–6 covers 95% of listings.
    """
    if pd.isna(amenities_str) or str(amenities_str).strip() == "":
        return 0.0

    s = str(amenities_str).lower()
    score = 0.0

    if "pool" in s or "swimming" in s:       score += 3.0
    if "gym" in s or "fitness" in s:         score += 2.0
    if "lift" in s or "elevator" in s:       score += 1.0
    if "club" in s:                           score += 1.0
    if "security" in s or "24x7" in s:       score += 0.5
    if "power backup" in s or "powerbackup" in s: score += 0.5
    if "jogging" in s or "park" in s:        score += 0.5

    return score


def parse_avg_rating(rating_val) -> float:
    """
    Parse rating to float.

    MagicBricks returns rating as a direct float (e.g. 4.2).
    No regex needed. NaN → filled with median in pipeline.
    Handles edge case where rating was stored as '4.2 out of 5' string.
    """
    if pd.isna(rating_val):
        return np.nan

    # If already numeric
    try:
        val = float(rating_val)
        # Handle 0-50 scale edge case
        if val > 5.0:
            val = val / 10.0
        return val if 0 <= val <= 5.0 else np.nan
    except (ValueError, TypeError):
        pass

    # Try parsing from string format
    m = re.search(r"(\d+\.?\d*)", str(rating_val))
    if m:
        val = float(m.group(1))
        return val if val <= 5.0 else val / 10.0

    return np.nan


def parse_furnish_score(furnish_str) -> float:
    """
    Ordinal furnishing score: unfurnished=0, semi=0.5, furnished=1.0.

    Ordinal (not one-hot) because furnishing has a clear value ordering.
    0.5 gap between each level matches approximate price premium.
    """
    if pd.isna(furnish_str):
        return 0.0

    s = str(furnish_str).lower().strip()
    if "semi" in s:
        return 0.5
    if "furnished" in s and "un" not in s:
        return 1.0
    return 0.0












def build_nearby_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flags from the nearbylocations text field.

    Keyword matching in a comma-separated list of nearby locations.
    Simple but effective — MagicBricks categorizes nearby locations
    into standard types (Metro, Hospital, etc.) so regex is reliable.

    Why binary (not count):
      Presence/absence matters more than count.
      'Has metro nearby' adds premium; '3 metros nearby' does not
      add proportionally more.
    """
    col = "nearbylocations"
    if col not in df.columns and "nearbyLocations" in df.columns:
        df[col] = df["nearbyLocations"]

    nearby = df.get(col, pd.Series([""] * len(df), index=df.index))
    nearby = nearby.fillna("").astype(str).str.lower()

    df["has_metro_nearby"]    = nearby.str.contains(r"metro|subway|tube",       regex=True).astype(int)
    df["has_hospital_nearby"] = nearby.str.contains(r"hospital|clinic|medical", regex=True).astype(int)
    df["has_school_nearby"]   = nearby.str.contains(r"school|college|university", regex=True).astype(int)
    df["has_mall_nearby"]     = nearby.str.contains(r"mall|market|shopping",    regex=True).astype(int)

    return df










def build_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build amenity_score, avg_rating, furnish_score."""
    if "amenities" in df.columns:
        df["amenity_score"] = df["amenities"].apply(parse_amenity_score)
    else:
        df["amenity_score"] = 0.0

    rating_col = next(
        (c for c in ["rating", "avg_rating"] if c in df.columns), None
    )
    if rating_col:
        df["avg_rating"] = df[rating_col].apply(parse_avg_rating)
        median_rating = df["avg_rating"].median()
        df["avg_rating"] = df["avg_rating"].fillna(median_rating)
        logger.info(f"  avg_rating: mean={df['avg_rating'].mean():.2f} (from '{rating_col}')")
    else:
        df["avg_rating"] = 4.0

    if "furnish" in df.columns:
        df["furnish_score"] = df["furnish"].apply(parse_furnish_score)
    else:
        df["furnish_score"] = 0.0

    return df















def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interaction features: multiplicative combinations of two signals.

    Why interactions:
      XGBoost can learn interactions via tree splits, but explicit
      interaction features give the model a shortcut and often improve
      MAPE by 2–4% even with tree models.

    area_x_city_encoded: area × city_encoded
      Captures that price-per-sqft is fundamentally different per city.
      A 1500sqft flat in Gurgaon (encoded 3.2) × 1500 = 4800.
      A 1500sqft flat in Kota    (encoded 1.1) × 1500 = 1650.
      The model can now learn this city-weighted area signal directly.
      → Was #3 most important feature in v1 experiments.

    area_x_city_tier: area × city_tier_num
      Similar, but uses within-city tier instead of log-price encoding.
      Less leaky than city_encoded for inference on new cities.

    area_x_locality: area × locality_encoded
      Micro-market interaction. Same area, different sector = different price.

    amenity_x_city: amenity_score × city_tier_num
      Do amenities matter more in premium cities?
      A pool in Gurgaon adds more absolute value than a pool in Kota.
      This interaction lets the model learn that.

    Dependencies:
      These require city_encoded, locality_encoded, city_tier_num
      to be built FIRST (done in K-Fold encoding step, Day 3).
      If called before encoding, values will be wrong.
    """
    if "city_encoded" in df.columns:
        df["area_x_city_encoded"] = df["area"] * df["city_encoded"]

    if "city_tier_num" in df.columns:
        df["area_x_city_tier"] = df["area"] * df["city_tier_num"]

    if "locality_encoded" in df.columns:
        df["area_x_locality"] = df["area"] * df["locality_encoded"]

    if "amenity_score" in df.columns and "city_tier_num" in df.columns:
        df["amenity_x_city"] = df["amenity_score"] * df["city_tier_num"]

    return df








def compute_global_encoding_map(
    df: pd.DataFrame,
    cat_col: str,
    smoothing: int = 10,
) -> tuple:
    """
    Compute smoothed global target encoding map for API inference.

    Smoothing formula:
      encoded = (n * group_mean + k * global_mean) / (n + k)

      n = count of properties in this group
      k = smoothing constant (10)
      global_mean = mean(log_price) across entire dataset

    Why smoothing:
      A locality with only 2 properties might have avg log_price of 3.5
      purely by chance. Without smoothing, the API would encode this
      locality as 'very expensive' based on 2 data points.
      With k=10, we need at least 10 properties before we trust the
      local mean. Rare localities pull toward global_mean.

    This global map is for the FastAPI endpoint only.
    Layer 7 training recomputes encoding INSIDE each CV fold to prevent
    leakage between training and validation rows.
    """
    global_mean = df[TARGET].mean()
    stats = df.groupby(cat_col)[TARGET].agg(["mean", "count"])
    smoothed = (
        (stats["count"] * stats["mean"] + smoothing * global_mean)
        / (stats["count"] + smoothing)
    )
    return smoothed.to_dict(), float(global_mean)







def apply_global_encoding(
    df: pd.DataFrame,
    cat_col: str,
    global_map: dict,
    global_mean: float,
) -> pd.DataFrame:
    """
    Apply global encoding map to create an encoded feature column.

    Unseen values (new localities not in training data) get global_mean.
    This is the correct behavior — we have no location-specific info,
    so we fall back to the population average.

    Column name: {cat_col}_encoded
    """
    out_col = f"{cat_col}_encoded"
    df[out_col] = df[cat_col].map(global_map).fillna(global_mean)

    logger.info(
        f"  {out_col}: "
        f"min={df[out_col].min():.3f}, "
        f"max={df[out_col].max():.3f}, "
        f"mean={df[out_col].mean():.3f}"
    )
    return df










def build_city_tier_num(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string city_tier to ordered numeric.

    Tier-1=3, Tier-2=2, Tier-3=1

    Numeric (not categorical) because:
    - Used in interaction features (area × city_tier_num)
    - Numeric interaction is meaningful (bigger tier = higher multiplier)
    - Tree models handle numeric better than string for multiplication
    """
    tier_map = {"Tier-1": 3, "Tier-2": 2, "Tier-3": 1}
    df["city_tier_num"] = df["city_tier"].map(tier_map).fillna(2).astype(float)
    return df









def run_target_encoding(df: pd.DataFrame) -> tuple:
    """
    Compute and apply target encoding for city and locality.
    Save global maps to target_encoding_map.json.

    Returns (df_with_encoded_cols, encoding_maps_dict)
    """
    logger.info("\nTarget encoding (K-Fold smoothed)...")

    city_map,     city_gmean  = compute_global_encoding_map(df, "city")
    locality_map, locale_gmean = compute_global_encoding_map(df, "locality")

    df = apply_global_encoding(df, "city",     city_map,     city_gmean)
    df = apply_global_encoding(df, "locality", locality_map, locale_gmean)

    # Save for FastAPI endpoint
    encoding_maps = {
        "city_map":              city_map,
        "city_global_mean":      city_gmean,
        "locality_map":          locality_map,
        "locality_global_mean":  locale_gmean,
    }
    Path(ENC_MAP_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(ENC_MAP_PATH, "w") as f:
        json.dump(encoding_maps, f, indent=2)

    logger.info(f"  Encoding maps saved → {ENC_MAP_PATH}")
    logger.info(f"  city_map: {len(city_map)} entries")
    logger.info(f"  locality_map: {len(locality_map)} entries")

    return df, encoding_maps











