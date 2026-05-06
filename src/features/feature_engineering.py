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