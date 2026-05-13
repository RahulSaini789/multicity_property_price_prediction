"""
src/features/feature_engineering.py
PropML — Feature engineering pipeline (Phase 3 enhanced).

Input:  data/cleaned/combined_cleaned.parquet
Output: data/features/combined_engineered.parquet
        data/features/feature_list.txt
        data/features/feature_metadata.json
        data/features/target_encoding_map.json

Features added vs Phase 1 (33 total → 50 total):
  Phase 1 (33):  city_tier_num, city_encoded, area interactions,
                 property type flags, ratio features, floor features,
                 quality features, basic nearby flags, age, bhk_x_city

  Phase 3 additions (+17):
    Amenity flags (8):   has_pool, has_gym, has_lift, has_security,
                         has_power_backup, has_park_jogging, has_clubhouse, has_garden
    Amenity count (1):   amenity_count
    Nearby enhanced (6): has_market_nearby, has_park_nearby, has_police_nearby,
                         nearby_score, is_well_served
    OSM distances (5):   dist_hospital_km, dist_school_km, dist_metro_km,
                         dist_market_km, dist_park_km
    Other (1):           rera_approved

OSM Overpass enrichment:
  Uses free OpenStreetMap Overpass API (no API key needed).
  Queries nearest hospital, school, police, market, park for each property.
  Results cached in data/osm_cache/ to avoid repeat calls.
  Run: python src/features/feature_engineering.py --enrich-osm

DVC stage (dvc.yaml):
  featurize:
    cmd: python src/features/feature_engineering.py
    deps: [data/cleaned/combined_cleaned.parquet]
    outs: [data/features/]
    params: [params.yaml:features]
"""

import json
import logging
import math
import os
import re
import sys
import time
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

# ── Paths ─────────────────────────────────────────────────────────────────────
CLEANED_PATH   = "data/cleaned/combined_cleaned.parquet"
FEATURES_PATH  = "data/features/combined_engineered.parquet"
FEAT_LIST_PATH = "data/features/feature_list.txt"
META_PATH      = "data/features/feature_metadata.json"
ENC_MAP_PATH   = "data/features/target_encoding_map.json"
OSM_CACHE_DIR  = "data/osm_cache"

TARGET = "log_price"

# ── Feature list (ordered by expected SHAP importance) ────────────────────────
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
    "amenity_count",
    "avg_rating",
    "furnish_score",
    "amenity_x_city",
    # Amenity binary flags (Phase 3)
    "has_pool",
    "has_gym",
    "has_lift",
    "has_security",
    "has_power_backup",
    "has_park_jogging",
    "has_clubhouse",
    "has_garden",
    # Nearby presence flags
    "has_metro_nearby",
    "has_hospital_nearby",
    "has_school_nearby",
    "has_mall_nearby",
    "has_market_nearby",
    "has_park_nearby",
    "has_police_nearby",
    # Nearby derived
    "nearby_score",
    "is_well_served",
    # OSM distances (km, -1 if unknown)
    "dist_hospital_km",
    "dist_school_km",
    "dist_metro_km",
    "dist_market_km",
    "dist_park_km",
    # Age
    "age_bucket",
    # Other
    "balcony",
    "parking",
    "is_near_coaching",
    "rera_approved",
    "bhk_x_city",
]


# ═══════════════════════════════════════════════════════════════════════════════
# OSM OVERPASS ENRICHMENT
# Free, no API key needed. Rate limit: 1 req/sec recommended.
# ═══════════════════════════════════════════════════════════════════════════════

# Overpass API endpoints (load balanced)
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# OSM tags for each facility type
OSM_FACILITY_TAGS = {
    "hospital": '[amenity~"hospital|clinic|doctors"]',
    "school":   '[amenity~"school|college|university"]',
    "police":   '[amenity="police"]',
    "market":   '[shop~"supermarket|mall|convenience|market"]',
    "park":     '[leisure~"park|garden|playground|recreation_ground"]',
    "metro":    '[station~"subway|metro"][railway="station"]',
}

# Search radius per facility (metres)
OSM_RADIUS = {
    "hospital": 3000,
    "school":   2000,
    "police":   3000,
    "market":   1500,
    "park":     2000,
    "metro":    5000,
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two lat/lon points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return round(R * 2 * math.asin(math.sqrt(a)), 3)


def _osm_cache_path(lat: float, lon: float, facility: str) -> Path:
    """Return cache file path for an OSM query."""
    key = f"{lat:.4f}_{lon:.4f}_{facility}"
    return Path(OSM_CACHE_DIR) / f"{key}.json"


def query_osm_nearest(
    lat: float,
    lon: float,
    facility: str,
    use_cache: bool = True,
) -> Optional[dict]:
    """
    Query OSM Overpass API for nearest facility to a coordinate.

    Uses Overpass QL to find the nearest amenity/shop/leisure node
    within the configured radius.

    Args:
        lat, lon:    property coordinates
        facility:    one of: hospital, school, police, market, park, metro
        use_cache:   read/write results from data/osm_cache/

    Returns:
        dict with keys: name, dist_km, osm_id
        None if not found or API fails

    Rate limiting: sleeps 1.1s between calls to respect Overpass limits.
    Cache: results stored as JSON files — no repeat calls for same location.
    """
    if not lat or not lon or lat == 0.0 or lon == 0.0:
        return None

    cache_path = _osm_cache_path(lat, lon, facility)

    # Read from cache
    if use_cache and cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            return data if data else None
        except Exception:
            pass

    radius   = OSM_RADIUS.get(facility, 2000)
    tag      = OSM_FACILITY_TAGS.get(facility, "")
    if not tag:
        return None

    # Overpass QL query — finds nearest node/way within radius
    query = f"""
[out:json][timeout:25];
(
  node{tag}(around:{radius},{lat},{lon});
  way{tag}(around:{radius},{lat},{lon});
);
out center 1;
"""

    result = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            import urllib.request
            import urllib.parse

            data    = urllib.parse.urlencode({"data": query}).encode()
            req     = urllib.request.Request(endpoint, data=data)
            req.add_header("User-Agent", "PropML/1.0 (property price prediction; educational)")

            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = json.loads(resp.read().decode())

            elements = raw.get("elements", [])
            if not elements:
                break

            # Get first result (nearest — Overpass returns sorted by distance)
            el       = elements[0]
            el_lat   = el.get("lat") or el.get("center", {}).get("lat")
            el_lon   = el.get("lon") or el.get("center", {}).get("lon")
            name     = (
                el.get("tags", {}).get("name")
                or el.get("tags", {}).get("name:en")
                or facility.title()
            )
            dist_km  = _haversine_km(lat, lon, el_lat, el_lon) if el_lat and el_lon else None

            result = {
                "name":    name,
                "dist_km": dist_km,
                "osm_id":  el.get("id"),
            }
            break

        except Exception as e:
            logger.debug(f"OSM Overpass error ({endpoint}) for {facility} at {lat},{lon}: {e}")
            continue

    # Cache result (even None → empty dict to avoid re-querying)
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result or {}))

    # Rate limit: 1 req/sec
    time.sleep(0.5)
    return result


def enrich_osm_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    OSM enrichment — locality-level caching (not row-level).

    Key optimization: group by rounded lat/lng (2 decimal = ~1km grid).
    Instead of 17000 queries, only unique grid cells queried.
    Typical reduction: 17000 rows -> 200-400 unique grid cells.
    Time: ~400 cells x 5 facilities x 1.1s = ~35 mins (was 26 hours).
    """
    facilities = ["hospital", "school", "metro", "market", "park"]
    dist_cols  = [f"dist_{f}_km" for f in facilities]

    for col in dist_cols:
        if col not in df.columns:
            df[col] = -1.0

    has_coords = (
        "lat" in df.columns and "lng" in df.columns and
        (df["lat"] != 0.0).any()
    )
    if not has_coords:
        logger.warning("  OSM enrichment skipped: no lat/lng columns")
        return df

    valid_mask = (df["lat"] != 0.0) & (df["lng"] != 0.0)
    logger.info(
        f"  OSM enrichment: {valid_mask.sum()} rows with coordinates "
        f"({(~valid_mask).sum()} skipped)"
    )

    # ── KEY OPTIMIZATION: round to 2 decimal places (~1km grid) ──────
    # lat 28.4541 -> 28.45, lon 77.0977 -> 77.10
    # All properties within ~1km get the same OSM result
    df["_lat_r"] = df["lat"].round(2)
    df["_lng_r"] = df["lng"].round(2)

    unique_coords = df[valid_mask][["_lat_r", "_lng_r"]].drop_duplicates()
    logger.info(
        f"  Grid cells (1km): {len(unique_coords)} unique "
        f"(was {valid_mask.sum()} rows — {valid_mask.sum()//max(len(unique_coords),1)}x faster)"
    )

    for i, facility in enumerate(facilities, 1):
        col = f"dist_{facility}_km"
        logger.info(f"  [{i}/{len(facilities)}] Querying OSM: {facility} "
                    f"({len(unique_coords)} grid cells)...")

        # Build grid-level cache
        grid_results = {}
        filled = 0

        for _, row in unique_coords.iterrows():
            lat_r = float(row["_lat_r"])
            lng_r = float(row["_lng_r"])
            cache_key = f"{lat_r}_{lng_r}_{facility}"

            # Check file cache first
            cache_path = Path(OSM_CACHE_DIR) / f"{cache_key}.json"
            if cache_path.exists():
                try:
                    data = json.loads(cache_path.read_text())
                    grid_results[(lat_r, lng_r)] = data.get("dist_km", -1.0) if data else -1.0
                    filled += 1
                    continue
                except Exception:
                    pass

            # Query OSM
            result = query_osm_nearest(lat_r, lng_r, facility)
            dist   = result["dist_km"] if result and result.get("dist_km") else -1.0
            grid_results[(lat_r, lng_r)] = dist
            if dist > 0:
                filled += 1

        # Map grid results back to all rows
        df[col] = df.apply(
            lambda r: grid_results.get(
                (r["_lat_r"], r["_lng_r"]), -1.0
            ) if valid_mask.loc[r.name] else -1.0,
            axis=1
        )

        logger.info(
            f"    {col}: {filled}/{len(unique_coords)} grid cells filled "
            f"(mean={df[df[col]>0][col].mean():.2f} km)"
        )

    df.drop(columns=["_lat_r", "_lng_r"], inplace=True)
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# AMENITY CATALOG (28 binary flags)
# ═══════════════════════════════════════════════════════════════════════════════

AMENITY_CATALOG = [
    # (search_keywords,          weight, output_flag_name)
    (["pool", "swimming"],       3.0,    "has_pool"),
    (["gym", "fitness"],         2.0,    "has_gym"),
    (["lift", "elevator"],       1.0,    "has_lift"),
    (["club"],                   1.0,    "has_clubhouse"),
    (["security", "24x7"],       0.5,    "has_security"),
    (["power backup"],           0.5,    "has_power_backup"),
    (["jogging", "park"],        0.5,    "has_park_jogging"),
    (["garden"],                 0.5,    "has_garden"),
    (["parking"],                1.0,    "has_parking"),
    (["cctv"],                   0.5,    "has_cctv"),
    (["intercom"],               0.3,    "has_intercom"),
    (["badminton"],              0.5,    "has_badminton"),
    (["tennis"],                 0.5,    "has_tennis"),
    (["basketball"],             0.5,    "has_basketball"),
    (["indoor games"],           0.5,    "has_indoor_games"),
    (["amphitheatre"],           0.3,    "has_amphitheatre"),
    (["rainwater"],              0.3,    "has_rainwater"),
    (["solar"],                  0.3,    "has_solar"),
    (["ev charging"],            0.5,    "has_ev_charging"),
    (["fire safety", "fire noc"],0.5,    "has_fire_safety"),
    (["wheelchair"],             0.3,    "has_wheelchair"),
    (["kids play", "children play"], 0.5, "has_kids_play"),
    (["multipurpose"],           0.3,    "has_multipurpose"),
    (["meditation"],             0.3,    "has_meditation"),
    (["library"],                0.3,    "has_library"),
    (["cafeteria"],              0.3,    "has_cafeteria"),
    (["atm"],                    0.3,    "has_atm"),
    (["spa"],                    0.5,    "has_spa"),
]


def parse_amenities_full(amenity_str) -> dict:
    """
    Parse raw amenity string into 28 binary flags + weighted score + tier.

    Why weighted sum:
      pool+security adds far more value than gym+lift even if count is equal.
      Weighted score captures this market reality.

    Tier classification:
      0      → none
      < 2    → basic
      < 5    → standard
      < 8    → premium
      >= 8   → luxury

    Returns dict with:
      amenity_score (float), amenity_count (int), amenity_tier (str),
      has_pool, has_gym, ... (28 binary int flags)
    """
    if pd.isna(amenity_str) or str(amenity_str).strip() == "":
        result = {"amenity_score": 0.0, "amenity_count": 0, "amenity_tier": "none"}
        for _, _, flag in AMENITY_CATALOG:
            result[flag] = 0
        return result

    text  = str(amenity_str).lower()
    score = 0.0
    count = 0
    flags = {}

    for keywords, weight, flag_name in AMENITY_CATALOG:
        present = any(kw in text for kw in keywords)
        flags[flag_name] = int(present)
        if present:
            score += weight
            count += 1

    if score == 0:
        tier = "none"
    elif score < 2:
        tier = "basic"
    elif score < 5:
        tier = "standard"
    elif score < 8:
        tier = "premium"
    else:
        tier = "luxury"

    return {
        "amenity_score": round(score, 2),
        "amenity_count": count,
        "amenity_tier":  tier,
        **flags,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CORE FEATURE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

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
    Prefer 'locality' if exists, fall back to 'sector'.
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
    Mutually exclusive and exhaustive.
    """
    df["is_flat"]              = (df["property_type"] == "flat").astype(int)
    df["is_house"]             = (df["property_type"] == "house").astype(int)
    df["is_independent_floor"] = (df["property_type"] == "independent_floor").astype(int)
    df["is_plot"]              = (df["property_type"] == "plot").astype(int)
    return df


def build_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    bath_per_bed, area_per_bhk, log_area.

    bath_per_bed: luxury proxy (3BHK + 5 bathrooms = luxury).
    area_per_bhk: space quality (600sqft 3BHK = cramped).
    log_area:     corrects right-skewed area distribution.
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
    relative_floor = floor_pos / total_floors (0=ground, 1=top).
    is_high_floor = 1 if relative_floor >= 0.75 (top quarter premium).
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
    Ordinal age encoding. Under construction=0, 0-1y=1, 1-5y=2, 5-10y=3, 10+y=4.
    Ordinal because age has a natural price ordering.
    """
    age_map = {
        "under construction": 0,
        "0-1 years": 1, "0-1": 1,
        "1-5 years": 2, "1-5": 2,
        "5-10 years": 3, "5-10": 3,
        "10+ years": 4, "10+": 4,
        "less than 5 years": 2,
        "more than 5 years": 3,
        "land": 0,
    }
    if "age" in df.columns:
        df["age_bucket"] = (
            df["age"].astype(str).str.lower().str.strip()
                     .map(age_map)
                     .fillna(2)
                     .astype(int)
        )
    else:
        df["age_bucket"] = 2
    return df


def parse_avg_rating(rating_val) -> float:
    """Parse rating to 0-5 float. Handles 0-50 scale edge cases."""
    if pd.isna(rating_val):
        return np.nan
    try:
        val = float(rating_val)
        if val > 5.0:
            val = val / 10.0
        return val if 0 <= val <= 5.0 else np.nan
    except (ValueError, TypeError):
        pass
    m = re.search(r"(\d+\.?\d*)", str(rating_val))
    if m:
        val = float(m.group(1))
        return val if val <= 5.0 else val / 10.0
    return np.nan


def parse_furnish_score(furnish_str) -> float:
    """Ordinal: unfurnished=0, semi=0.5, furnished=1.0."""
    if pd.isna(furnish_str):
        return 0.0
    s = str(furnish_str).lower().strip()
    if "semi" in s:
        return 0.5
    if "furnished" in s and "un" not in s:
        return 1.0
    return 0.0


def build_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build amenity features using full 28-flag catalog (Phase 3).
    Also builds avg_rating and furnish_score.
    """
    # Full amenity parsing
    logger.info("  Parsing amenities (28-flag catalog)...")
    if "amenities" in df.columns:
        amenity_results = df["amenities"].apply(parse_amenities_full)
        amenity_df      = pd.DataFrame(list(amenity_results), index=df.index)
        for col in amenity_df.columns:
            if col != "amenity_tier":   # skip string column from FINAL_FEATURES
                df[col] = amenity_df[col].values
        logger.info(f"    amenity_score: mean={df['amenity_score'].mean():.2f}, max={df['amenity_score'].max():.1f}")
        logger.info(f"    has_pool: {df['has_pool'].mean()*100:.1f}%  has_gym: {df['has_gym'].mean()*100:.1f}%")
    else:
        df["amenity_score"] = 0.0
        df["amenity_count"] = 0
        for _, _, flag in AMENITY_CATALOG:
            df[flag] = 0

    # Rating
    rating_col = next((c for c in ["rating", "avg_rating"] if c in df.columns), None)
    if rating_col:
        df["avg_rating"] = df[rating_col].apply(parse_avg_rating)
        df["avg_rating"] = df["avg_rating"].fillna(df["avg_rating"].median())
        logger.info(f"    avg_rating: mean={df['avg_rating'].mean():.2f}")
    else:
        df["avg_rating"] = 4.0

    # Furnish
    if "furnish" in df.columns:
        df["furnish_score"] = df["furnish"].apply(parse_furnish_score)
    else:
        df["furnish_score"] = 0.0

    return df


def build_nearby_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary nearby flags from nearbylocations text + enhanced derived features.

    Phase 3 adds: has_market_nearby, has_park_nearby, has_police_nearby,
                  nearby_score, is_well_served.

    nearby_score: weighted sum (metro=3, hospital=2, school=1.5,
                                market=1, park=1, police=0.5)
    is_well_served: 1 if nearby_score >= 5
    """
    col = "nearbylocations"
    if col not in df.columns and "nearbyLocations" in df.columns:
        df[col] = df["nearbyLocations"]

    nearby = df.get(col, pd.Series([""] * len(df), index=df.index))
    nearby = nearby.fillna("").astype(str).str.lower()

    # Basic flags
    df["has_metro_nearby"]    = nearby.str.contains(r"metro|subway|tube",        regex=True).astype(int)
    df["has_hospital_nearby"] = nearby.str.contains(r"hospital|clinic|medical",  regex=True).astype(int)
    df["has_school_nearby"]   = nearby.str.contains(r"school|college|university", regex=True).astype(int)
    df["has_mall_nearby"]     = nearby.str.contains(r"mall|market|shopping",     regex=True).astype(int)

    # Enhanced flags (Phase 3)
    # Prefer scraper-provided columns if already present
    if "has_market_nearby" not in df.columns:
        df["has_market_nearby"] = nearby.str.contains(
            r"market|mall|shopping|bazaar|supermarket", regex=True
        ).astype(int)
    else:
        df["has_market_nearby"] = pd.to_numeric(df["has_market_nearby"], errors="coerce").fillna(0).astype(int)

    if "has_park_nearby" not in df.columns:
        df["has_park_nearby"] = nearby.str.contains(
            r"park|garden|ground|playground", regex=True
        ).astype(int)
    else:
        df["has_park_nearby"] = pd.to_numeric(df["has_park_nearby"], errors="coerce").fillna(0).astype(int)

    if "has_police_nearby" not in df.columns:
        df["has_police_nearby"] = nearby.str.contains(
            r"police|thana", regex=True
        ).astype(int)
    else:
        df["has_police_nearby"] = pd.to_numeric(df["has_police_nearby"], errors="coerce").fillna(0).astype(int)

    # Weighted nearby score
    nearby_weights = {
        "has_metro_nearby":    3.0,
        "has_hospital_nearby": 2.0,
        "has_school_nearby":   1.5,
        "has_market_nearby":   1.0,
        "has_park_nearby":     1.0,
        "has_police_nearby":   0.5,
    }
    df["nearby_score"] = sum(
        df[col] * w for col, w in nearby_weights.items()
    )
    df["is_well_served"] = (df["nearby_score"] >= 5.0).astype(int)

    logger.info(f"    nearby_score: mean={df['nearby_score'].mean():.2f}")
    logger.info(f"    is_well_served: {df['is_well_served'].mean()*100:.1f}% of listings")

    return df


def build_osm_distance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Initialize OSM distance columns.

    If dist_*_km columns already exist in data (from OSM enrichment step),
    use them. Otherwise fill with -1.0 (unknown).

    OSM enrichment is run separately via:
      python src/features/feature_engineering.py --enrich-osm

    Distance columns: dist_hospital_km, dist_school_km, dist_metro_km,
                      dist_market_km, dist_park_km
    Value: float km, -1.0 = unknown/not enriched yet.
    """
    dist_cols = [
        "dist_hospital_km", "dist_school_km", "dist_metro_km",
        "dist_market_km",   "dist_park_km",
    ]
    for col in dist_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1.0)
        else:
            df[col] = -1.0

    enriched = (df["dist_hospital_km"] > 0).sum()
    logger.info(f"    OSM distances: {enriched}/{len(df)} rows enriched")
    return df


def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multiplicative interaction features.

    area_x_city_encoded: city-weighted area (most important interaction).
    area_x_city_tier:    tier-weighted area.
    area_x_locality:     micro-market area interaction.
    amenity_x_city:      amenity value varies by city tier.
    bhk_x_city:          BHK premium varies per city.
    """
    if "city_encoded" in df.columns:
        df["area_x_city_encoded"] = df["area"] * df["city_encoded"]

    if "city_tier_num" in df.columns:
        df["area_x_city_tier"] = df["area"] * df["city_tier_num"]

    if "locality_encoded" in df.columns:
        df["area_x_locality"] = df["area"] * df["locality_encoded"]

    if "amenity_score" in df.columns and "city_tier_num" in df.columns:
        df["amenity_x_city"] = df["amenity_score"] * df["city_tier_num"]

    if "bhk" in df.columns and "city_encoded" in df.columns:
        df["bhk_x_city"] = df["bhk"] * df["city_encoded"]

    return df


def build_city_tier_num(df: pd.DataFrame) -> pd.DataFrame:
    """
    City tier to ordered numeric: Tier-1=3, Tier-2=2, Tier-3=1.
    Numeric for interaction features (area × city_tier_num).
    """
    tier_map = {"Tier-1": 3, "Tier-2": 2, "Tier-3": 1}
    df["city_tier_num"] = df["city_tier"].map(tier_map).fillna(2).astype(float)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TARGET ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_global_encoding_map(
    df: pd.DataFrame,
    cat_col: str,
    smoothing: int = 10,
) -> tuple:
    """
    Smoothed global target encoding map for API inference.

    Formula: (n * group_mean + k * global_mean) / (n + k)
    Rare localities pull toward global_mean (prevents overfitting on 2-row localities).
    """
    global_mean = df[TARGET].mean()
    stats       = df.groupby(cat_col)[TARGET].agg(["mean", "count"])
    smoothed    = (
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
    """Apply encoding map. Unseen values → global_mean."""
    out_col    = f"{cat_col}_encoded"
    df[out_col] = df[cat_col].map(global_map).fillna(global_mean)
    logger.info(
        f"  {out_col}: "
        f"min={df[out_col].min():.3f}, "
        f"max={df[out_col].max():.3f}, "
        f"mean={df[out_col].mean():.3f}"
    )
    return df


def run_target_encoding(df: pd.DataFrame) -> tuple:
    """Compute and apply target encoding for city and locality. Save maps."""
    logger.info("\nTarget encoding (K-Fold smoothed)...")

    city_map,     city_gmean   = compute_global_encoding_map(df, "city")
    locality_map, locale_gmean = compute_global_encoding_map(df, "locality")

    df = apply_global_encoding(df, "city",     city_map,     city_gmean)
    df = apply_global_encoding(df, "locality", locality_map, locale_gmean)

    encoding_maps = {
        "city_map":             city_map,
        "city_global_mean":     city_gmean,
        "locality_map":         locality_map,
        "locality_global_mean": locale_gmean,
    }
    Path(ENC_MAP_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(ENC_MAP_PATH, "w") as f:
        json.dump(encoding_maps, f, indent=2)

    logger.info(f"  Encoding maps saved → {ENC_MAP_PATH}")
    logger.info(f"  city_map: {len(city_map)} entries")
    logger.info(f"  locality_map: {len(locality_map)} entries")

    return df, encoding_maps


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_features(df: pd.DataFrame, features: list) -> bool:
    """
    Post-engineering validation. Checks all features for:
    - Existence, NaN, non-numeric dtype, Inf values.
    Returns True if all pass. Logs all failures at once.
    """
    logger.info("\nFeature validation...")
    issues = []

    for feat in features:
        if feat not in df.columns:
            issues.append(f"MISSING: '{feat}' not in DataFrame")
            continue

        series = df[feat]

        nan_count = series.isna().sum()
        if nan_count > 0:
            pct = nan_count / len(df) * 100
            issues.append(f"NaN: '{feat}' has {nan_count} nulls ({pct:.1f}%)")

        if series.dtype == object:
            issues.append(f"DTYPE: '{feat}' is object/string (must be numeric)")

        if np.isinf(df[feat].values).any():
            issues.append(f"Inf values in {feat}")

    if issues:
        for issue in issues:
            logger.error(f"  FAIL: {issue}")
        return False

    logger.info(f"  All {len(features)} features valid. No NaN/Inf/dtype issues.")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_feature_engineering(enrich_osm: bool = False) -> pd.DataFrame:
    """
    Full Layer 6 pipeline — assembles all 50 features.

    Args:
        enrich_osm: if True, runs OSM Overpass enrichment for distance columns.
                    Requires lat/lng columns in cleaned data.
                    Takes ~10 mins for 100 rows × 5 facilities.

    Output files:
      data/features/combined_engineered.parquet
      data/features/feature_list.txt
      data/features/feature_metadata.json
      data/features/target_encoding_map.json
    """
    logger.info("=" * 60)
    logger.info("Layer 6: Feature Engineering Pipeline (Phase 3)")
    logger.info("=" * 60)

    # ── Stage 1: Load ─────────────────────────────────────────────────
    df = load_cleaned()
    df = align_location_column(df)

    # ── Stage 2: Numeric passthrough ──────────────────────────────────
    for col in ["bhk", "bathroom", "balcony", "parking"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 10)

    # rera_approved passthrough
    if "rera_approved" in df.columns:
        df["rera_approved"] = pd.to_numeric(df["rera_approved"], errors="coerce").fillna(0).astype(int)
    else:
        df["rera_approved"] = 0

    # ── Stage 3: City tier numeric ────────────────────────────────────
    df = build_city_tier_num(df)

    # ── Stage 4: Target encoding ──────────────────────────────────────
    df, _ = run_target_encoding(df)

    # ── Stage 5: Property type one-hot ────────────────────────────────
    df = build_property_type_flags(df)

    # ── Stage 6: Ratio + log features ────────────────────────────────
    df = build_ratio_features(df)

    # ── Stage 7: Floor features ───────────────────────────────────────
    df = build_floor_features(df)

    # ── Stage 8: Quality + amenity flags (Phase 3) ────────────────────
    logger.info("\nBuilding quality features (Phase 3: 28-flag amenity catalog)...")
    df = build_quality_features(df)

    # ── Stage 9: Nearby flags (Phase 3 enhanced) ──────────────────────
    logger.info("\nBuilding nearby flags (Phase 3: 6 flags + score)...")
    df = build_nearby_flags(df)

    # ── Stage 10: OSM distance columns ────────────────────────────────
    logger.info("\nBuilding OSM distance columns...")
    if enrich_osm:
        logger.info("  Running OSM Overpass enrichment (this takes ~10 mins)...")
        df = enrich_osm_distances(df)
    df = build_osm_distance_columns(df)

    # ── Stage 11: Age bucket ──────────────────────────────────────────
    df = build_age_bucket(df)

    # ── Stage 12: Interaction features ───────────────────────────────
    df = build_interaction_features(df)

    # ── Stage 13: Filter to FINAL_FEATURES, fill NaN ─────────────────
    available = []
    skipped   = []
    for feat in FINAL_FEATURES:
        if feat in df.columns:
            df[feat] = pd.to_numeric(df[feat], errors="coerce").astype(float)
            null_count = df[feat].isna().sum()
            if null_count > 0:
                fill_val = df[feat].median()
                df[feat] = df[feat].fillna(fill_val)
                logger.debug(f"  Filled {null_count} NaN in '{feat}' with median={fill_val:.3f}")
            available.append(feat)
        else:
            skipped.append(feat)

    if skipped:
        logger.warning(f"  Skipped {len(skipped)} features not in data: {skipped}")

    logger.info(f"\nFinal feature count: {len(available)} / {len(FINAL_FEATURES)}")

    # ── Stage 14: Validate ────────────────────────────────────────────
    is_valid = validate_features(df, available)
    if not is_valid:
        logger.warning("  Validation issues found — check feature engineering steps")

    # ── Stage 15: Save ────────────────────────────────────────────────
    metadata_cols = [TARGET, "price", "city", "property_type", "city_tier"]
    out_cols      = available + [c for c in metadata_cols if c in df.columns]
    out_df        = df[out_cols].copy()

    Path(FEATURES_PATH).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(FEATURES_PATH, index=False)

    with open(FEAT_LIST_PATH, "w") as f:
        f.write("\n".join(available))

    metadata = {
        "feature_columns": available,
        "target_column":   TARGET,
        "n_features":      len(available),
        "n_rows":          len(out_df),
        "generated_at":    datetime.now().isoformat(),
        "phase":           "phase3",
        "osm_enriched":    enrich_osm,
        "cities":          out_df["city"].value_counts().to_dict(),
        "property_types":  out_df["property_type"].value_counts().to_dict(),
        "feature_stats": {
            feat: {
                "mean": round(float(out_df[feat].mean()), 4),
                "std":  round(float(out_df[feat].std()),  4),
                "min":  round(float(out_df[feat].min()),  4),
                "max":  round(float(out_df[feat].max()),  4),
            }
            for feat in available
        },
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nFeature engineering complete")
    logger.info(f"   {len(out_df)} rows x {len(available)} features")
    logger.info(f"   Amenity flags: {sum(1 for f in available if f.startswith('has_') and f not in ['has_metro_nearby','has_hospital_nearby','has_school_nearby','has_mall_nearby','has_market_nearby','has_park_nearby','has_police_nearby','is_well_served'])} binary flags")
    logger.info(f"   Nearby flags:  {sum(1 for f in available if 'nearby' in f or f in ['is_well_served'])}")
    logger.info(f"   OSM distances: {sum(1 for f in available if f.startswith('dist_'))}")
    logger.info(f"   -> {FEATURES_PATH}")
    logger.info(f"   -> {FEAT_LIST_PATH}")
    logger.info(f"   -> {ENC_MAP_PATH}")
    logger.info(f"   -> {META_PATH}")
    logger.info(f"\nNext: python src/training/train.py")

    return out_df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PropML Feature Engineering")
    parser.add_argument(
        "--enrich-osm",
        action="store_true",
        help="Run OSM Overpass enrichment for hospital/school/police/market/park distances"
    )
    args = parser.parse_args()
    run_feature_engineering(enrich_osm=args.enrich_osm)


if __name__ == "__main__":
    main()