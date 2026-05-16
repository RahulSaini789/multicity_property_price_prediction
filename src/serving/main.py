"""
src/serving/main.py
PropML FastAPI — Property Price Prediction API (Phase 3, 13 cities).

Endpoints:
  GET  /              — API info
  GET  /health        — model health + latency stats
  GET  /model-info    — feature list, ensemble weights
  POST /predict       — price prediction + SHAP top-3
  POST /recommend     — content-based property recommendations
  GET  /analytics     — list available analytics reports
  GET  /analytics/{name} — serve pre-computed analytics JSON

Run:
  uvicorn src.serving.main:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional
import pathlib

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR    = pathlib.Path(os.getenv("MODELS_DIR", "models"))
ANALYTICS_DIR = pathlib.Path(os.getenv("ANALYTICS_DIR", "reports/analytics"))
ENC_MAP_PATH  = MODELS_DIR / "target_encoding_map.json"
ENC_MAP_PATH_FALLBACK = pathlib.Path("data/features/target_encoding_map.json")


# ── CatBoost optional ─────────────────────────────────────────────────────────
CATBOOST_AVAILABLE = False
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    pass


# ── Shared state ──────────────────────────────────────────────────────────────
STATE: dict = {
    "models_xgb":        None,
    "models_lgb":        None,
    "models_cat":        None,
    "ensemble_weights":  (0.48, 0.16, 0.36),
    "feature_list":      [],
    "shap_explainer":    None,
    "shap_mode":         "disabled",
    "encoding_map":      {},
    "start_time":        time.time(),
    "predictions_count": 0,
    "latencies":         [],
}


# ── Domain constants (Phase 3: 13 cities) ────────────────────────────────────
VALID_CITIES = {
    "gurgaon", "noida", "chandigarh", "kota",
    "delhi", "mumbai", "bangalore", "chennai",
    "pune", "hyderabad", "ahmedabad", "jaipur", "indore",
}

VALID_TYPES = {"flat", "house", "independent_floor", "plot"}

# city_tier_num: Tier-1=3, Tier-2=2, Tier-3=1
# Based on median property prices across India
TIER_MAP = {
    "mumbai":     3,
    "gurgaon":    3,
    "delhi":      3,
    "bangalore":  3,
    "noida":      2,
    "pune":       2,
    "hyderabad":  2,
    "chennai":    2,
    "ahmedabad":  2,
    "chandigarh": 2,
    "jaipur":     1,
    "indore":     1,
    "kota":       1,
}

FURNISH_MAP = {
    "unfurnished":    0.0,
    "semi-furnished": 0.5,
    "furnished":      1.0,
}

AGE_MAP = {
    "under construction": 0,
    "0-1 years":          1,
    "1-5 years":          2,
    "less than 5 years":  2,
    "5-10 years":         3,
    "more than 5 years":  3,
    "10+ years":          4,
}

# CV MAPE by city — used for confidence intervals
CITY_CV_MAPE = {
    "pune":       0.155,
    "ahmedabad":  0.161,
    "hyderabad":  0.171,
    "mumbai":     0.173,
    "bangalore":  0.190,
    "noida":      0.207,
    "gurgaon":    0.211,
    "indore":     0.218,
    "chennai":    0.222,
    "delhi":      0.226,
    "kota":       0.243,
    "chandigarh": 0.263,
    "jaipur":     0.289,
}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_pkl(filename: str):
    """Load a joblib pkl from MODELS_DIR. Returns None if missing."""
    path = MODELS_DIR / filename
    if os.path.exists(path):
        try:
            obj = joblib.load(path)
            logger.info(f"  Loaded: {path}")
            return obj
        except Exception as e:
            logger.error(f"  Load failed {path}: {e}")
    else:
        logger.warning(f"  Not found: {path}")
    return None


def _init_shap(models_xgb: list, feature_list: list) -> tuple:
    """
    Initialize SHAP KernelExplainer.

    KernelExplainer used instead of TreeExplainer because this XGBoost
    version stores leaf values as JSON arrays. SHAP's C parser fails on
    these. KernelExplainer only calls model.predict() — always works.

    Background: 50 rows from training data. Falls back to zeros if
    training data not available (Render/Docker deployment).
    """
    try:
        import shap

        model = models_xgb[0]
        bg_df = None

        for p in [
            pathlib.Path("data/features/combined_engineered.parquet"),
            pathlib.Path("data/cleaned/combined_cleaned.parquet"),
        ]:
            if p.exists():
                raw  = pd.read_parquet(p)
                cols = [f for f in feature_list if f in raw.columns]
                if cols:
                    sample = (
                        raw[cols]
                        .fillna(0)
                        .sample(min(50, len(raw)), random_state=42)
                    )
                    for feat in feature_list:
                        if feat not in sample.columns:
                            sample[feat] = 0.0
                    bg_df = sample[feature_list]
                break

        if bg_df is None:
            bg_df = pd.DataFrame(
                np.zeros((1, len(feature_list))), columns=feature_list
            )

        def predict_fn(X_arr):
            X_df = pd.DataFrame(X_arr, columns=feature_list)
            return model.predict(X_df)

        explainer = shap.KernelExplainer(predict_fn, bg_df)
        logger.info(f"  SHAP: KernelExplainer OK (bg={len(bg_df)} rows)")
        return explainer, "kernel"

    except Exception as e:
        logger.warning(f"  SHAP disabled: {e}")
        return None, "disabled"


def download_models_from_hf():
    """
    Download model pkl files from HuggingFace Hub at startup.

    Called when running on Render (HF_MODEL_REPO env var is set).
    Skipped when running locally (models/ folder already has pkl files).

    Why download at startup instead of baking into Docker image:
      Model files are 200–500MB each. Including in Docker image would:
      a) Make the image 2GB+ (Render free tier limit is 512MB RAM)
      b) Require Docker rebuild on every model update
      HF Hub download: model update = just run upload_to_hf.py + redeploy.
    """
    hf_repo = os.getenv("HF_MODEL_REPO")
    if not hf_repo:
        logger.info("HF_MODEL_REPO not set — skipping HF download (local mode)")
        return

    models_dir = pathlib.Path(MODELS_DIR)
    models_dir.mkdir(exist_ok=True)

    # Check if models already exist (cached from previous warm request)
    required_files = ["models_xgb_ensemble.pkl", "ensemble_weights.pkl", "feature_list.pkl"]
    if all((models_dir / f).exists() for f in required_files):
        logger.info("Models already present — skipping HF download")
        return

    logger.info(f"Downloading models from HuggingFace: {hf_repo}")

    try:
        from huggingface_hub import hf_hub_download

        files = [
            "models_xgb_ensemble.pkl",
            "models_lgb_ensemble.pkl",
            "models_cat_ensemble.pkl",
            "ensemble_weights.pkl",
            "feature_list.pkl",
            "version.txt",
            "target_encoding_map.json",
            "combined_engineered.parquet",
            "combined_cleaned.parquet",
        ]

        for filename in files:
            try:
                local_path = hf_hub_download(
                    repo_id=hf_repo,
                    filename=filename,
                    local_dir=str(models_dir),
                    repo_type="model",
                )
                logger.info(f"  Downloaded: {filename}")

                # Move target_encoding_map.json to correct location
                if filename == "target_encoding_map.json":
                    import shutil
                    pathlib.Path(ENC_MAP_PATH).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(local_path, ENC_MAP_PATH)
                # Move parquet files to correct data/ locations
                if filename == "combined_engineered.parquet":
                    pathlib.Path("data/features").mkdir(parents=True, exist_ok=True)
                    shutil.copy(local_path, "data/features/combined_engineered.parquet")
                if filename == "combined_cleaned.parquet":
                    pathlib.Path("data/cleaned").mkdir(parents=True, exist_ok=True)
                    shutil.copy(local_path, "data/cleaned/combined_cleaned.parquet")

            except Exception as e:
                logger.warning(f"  Could not download {filename}: {e}")

        logger.info("HF download complete")

    except ImportError:
        logger.error("huggingface-hub not installed. Add to requirements-render.txt")
    except Exception as e:
        logger.error(f"HF download failed: {e}")


def load_models():
    """
    Load all models into STATE at startup.

    Loading order:
    1. XGBoost (required)
    2. LightGBM (optional)
    3. CatBoost (optional)
    4. Ensemble weights
    5. Feature list (required)
    6. SHAP explainer (optional)
    7. Encoding map (required for inference)
    """
    logger.info("=" * 55)
    logger.info("PropML API — Model Loading (Phase 3, 13 cities)")

    download_models_from_hf()

    STATE["models_xgb"] = _load_pkl("models_xgb_ensemble.pkl")
    STATE["models_lgb"] = _load_pkl("models_lgb_ensemble.pkl")
    STATE["models_cat"] = _load_pkl("models_cat_ensemble.pkl")

    weights = _load_pkl("ensemble_weights.pkl")
    if weights is not None:
        STATE["ensemble_weights"] = tuple(weights)

    feat_list = _load_pkl("feature_list.pkl")
    if feat_list is not None:
        STATE["feature_list"] = list(feat_list)
        logger.info(f"  feature_list: {len(STATE['feature_list'])} features")

    if STATE["models_xgb"] and STATE["feature_list"]:
        STATE["shap_explainer"], STATE["shap_mode"] = _init_shap(
            STATE["models_xgb"], STATE["feature_list"]
        )

  # Direct path to the Hugging Face models folder
    real_enc_path = pathlib.Path("models/target_encoding_map.json")

    # Check if the real file exists, otherwise try your old fallbacks
    if real_enc_path.exists():
        final_path = real_enc_path
    elif ENC_MAP_PATH.exists():
        final_path = ENC_MAP_PATH
    elif ENC_MAP_PATH_FALLBACK.exists():
        final_path = ENC_MAP_PATH_FALLBACK
    else:
        final_path = None

    # Load the JSON safely
    if final_path:
        with open(final_path, "r") as f:
            STATE["encoding_map"] = json.load(f)  # <--- Sirf ek baar load karna hai
            
        n_loc = len(STATE["encoding_map"].get("locality_map", {}))
        logger.info(f"  Encoding map: {n_loc} localities loaded from {final_path}")
    else:
        logger.warning("  Encoding map NOT FOUND! API might fail on categorical inputs.")

    logger.info(
        f"  xgb={STATE['models_xgb'] is not None} "
        f"lgb={STATE['models_lgb'] is not None} "
        f"cat={STATE['models_cat'] is not None} "
        f"shap={STATE['shap_mode']} "
        f"features={len(STATE['feature_list'])}"
    )
    logger.info("=" * 55)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: load_models() once at startup.

    Why lifespan (not @app.on_event): on_event is deprecated in
    FastAPI 0.93+. lifespan is the recommended pattern.
    """
    load_models()
    yield
    logger.info("PropML API — Shutdown")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PropML Property Price API",
    description=(
        "Multi-city Indian real estate price prediction. "
        "13 cities: Gurgaon, Noida, Delhi, Mumbai, Bangalore, "
        "Chennai, Pune, Hyderabad, Ahmedabad, Jaipur, Indore, "
        "Chandigarh, Kota."
    ),
    version="3.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Ya fir apne frontend ka exact domain dalein
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    city:             str   = Field(...,   description=f"One of: {sorted(VALID_CITIES)}")
    property_type:    str   = Field("flat", description="flat | house | independent_floor | plot")
    area_sqft:        float = Field(...,   ge=100, le=50000)
    bhk:              int   = Field(...,   ge=0, le=10)
    bathroom:         int   = Field(2,     ge=0, le=8)
    balcony:          int   = Field(1,     ge=0, le=5)
    floor_pos:        int   = Field(3,     ge=0, le=60)
    total_floors:     int   = Field(10,    ge=1, le=60)
    age:              str   = Field("1-5 years")
    furnish:          str   = Field("semi-furnished")
    amenity_score:    float = Field(2.0,   ge=0, le=15)
    avg_rating:       float = Field(4.0,   ge=1.0, le=5.0)
    parking:          int   = Field(1,     ge=0, le=5)
    locality:         Optional[str] = None
    # Phase 3: nearby flags (optional, default 0)
    has_metro_nearby:    int   = Field(0, ge=0, le=1)
    has_hospital_nearby: int   = Field(0, ge=0, le=1)
    has_school_nearby:   int   = Field(0, ge=0, le=1)
    has_mall_nearby:     int   = Field(0, ge=0, le=1)
    has_market_nearby:   int   = Field(0, ge=0, le=1)
    has_park_nearby:     int   = Field(0, ge=0, le=1)
    has_police_nearby:   int   = Field(0, ge=0, le=1)
    # Phase 3: amenity binary flags (optional)
    has_pool:            int   = Field(0, ge=0, le=1)
    has_gym:             int   = Field(0, ge=0, le=1)
    has_lift:            int   = Field(0, ge=0, le=1)
    has_security:        int   = Field(0, ge=0, le=1)
    has_power_backup:    int   = Field(0, ge=0, le=1)
    # Phase 3: OSM distances (-1 = unknown)
    dist_hospital_km:    float = Field(-1.0)
    dist_school_km:      float = Field(-1.0)
    dist_metro_km:       float = Field(-1.0)
    dist_market_km:      float = Field(-1.0)
    dist_park_km:        float = Field(-1.0)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "city": "mumbai", "property_type": "flat",
                "area_sqft": 950, "bhk": 2, "bathroom": 2,
                "balcony": 1, "floor_pos": 8, "total_floors": 20,
                "age": "1-5 years", "furnish": "semi-furnished",
                "amenity_score": 3.5, "avg_rating": 4.2,
                "parking": 1, "locality": "Andheri West",
                "has_metro_nearby": 1, "has_mall_nearby": 1,
            }]
        }
    }


class RecommendRequest(BaseModel):
    city:          str             = Field(..., description=f"One of: {sorted(VALID_CITIES)}")
    budget_cr:     float           = Field(..., gt=0, description="Budget in Crores e.g. 1.5")
    bhk:           int             = Field(..., ge=1, le=10)
    property_type: str             = Field("flat")
    area_min:      Optional[float] = None
    area_max:      Optional[float] = None
    amenity_score: Optional[float] = None
    top_n:         int             = Field(5, ge=1, le=20)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "city": "bangalore", "budget_cr": 1.2, "bhk": 2,
                "property_type": "flat", "area_min": 900.0, "area_max": 1400.0,
            }]
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (INFERENCE)
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_df(req: PredictRequest) -> pd.DataFrame:
    """
    Convert PredictRequest into model-ready feature DataFrame.

    Must produce EXACTLY the same features in EXACTLY the same order
    as feature_list.txt from Layer 6.

    Phase 3 additions:
    - 28 amenity binary flags
    - 7 nearby flags + nearby_score + is_well_served
    - 5 OSM distance columns (dist_*_km)
    - amenity_count
    """
    city = req.city.lower().strip()
    pt   = req.property_type.lower().strip()
    enc  = STATE["encoding_map"]

    city_map     = enc.get("city_map", {})
    city_gmean   = enc.get("city_global_mean", 0.75)
    loc_map      = enc.get("locality_map", {})

    city_encoded = city_map.get(city, city_gmean)
    loc_key      = (req.locality or "").lower().replace(" ", "_")
    loc_encoded  = loc_map.get(loc_key, city_gmean)

    city_tier_num = float(TIER_MAP.get(city, 2))
    furnish_score = FURNISH_MAP.get(req.furnish.lower().strip(), 0.5)
    age_bucket    = float(AGE_MAP.get(req.age.lower().strip(), 2))
    rel_floor     = min(req.floor_pos / max(req.total_floors, 1), 1.0)

    # Nearby score (weighted)
    nearby_score = (
        req.has_metro_nearby    * 3.0 +
        req.has_hospital_nearby * 2.0 +
        req.has_school_nearby   * 1.5 +
        req.has_market_nearby   * 1.0 +
        req.has_park_nearby     * 1.0 +
        req.has_police_nearby   * 0.5
    )
    is_well_served = float(nearby_score >= 5.0)

    # amenity_count from binary flags
    amenity_count = sum([
        req.has_pool, req.has_gym, req.has_lift,
        req.has_security, req.has_power_backup,
    ])

    fv = {
        # Location
        "city_tier_num":           city_tier_num,
        "city_encoded":            city_encoded,
        "area_x_city_encoded":     req.area_sqft * city_encoded,
        "locality_encoded":        loc_encoded,
        "area_x_city_tier":        req.area_sqft * city_tier_num,
        # Property type
        "is_house":                float(pt == "house"),
        "is_flat":                 float(pt == "flat"),
        "is_independent_floor":    float(pt == "independent_floor"),
        "is_plot":                 float(pt == "plot"),
        # Area
        "area":                    req.area_sqft,
        "log_area":                float(np.log1p(req.area_sqft)),
        "area_x_locality":         req.area_sqft * loc_encoded,
        "area_per_bhk":            req.area_sqft / max(req.bhk, 1),
        # Rooms
        "bhk":                     float(req.bhk),
        "bathroom":                float(req.bathroom),
        "bath_per_bed":            req.bathroom / max(req.bhk, 1),
        # Floor
        "floor_pos":               float(req.floor_pos),
        "total_floors":            float(req.total_floors),
        "relative_floor":          rel_floor,
        "is_high_floor":           float(rel_floor >= 0.75),
        # Quality
        "amenity_score":           req.amenity_score,
        "amenity_count":           float(amenity_count),
        "avg_rating":              req.avg_rating,
        "furnish_score":           furnish_score,
        "amenity_x_city":          req.amenity_score * city_tier_num,
        # Phase 3: amenity binary flags
        "has_pool":                float(req.has_pool),
        "has_gym":                 float(req.has_gym),
        "has_lift":                float(req.has_lift),
        "has_security":            float(req.has_security),
        "has_power_backup":        float(req.has_power_backup),
        "has_park_jogging":        0.0,
        "has_clubhouse":           0.0,
        "has_garden":              0.0,
        # Phase 3: nearby flags
        "has_metro_nearby":        float(req.has_metro_nearby),
        "has_hospital_nearby":     float(req.has_hospital_nearby),
        "has_school_nearby":       float(req.has_school_nearby),
        "has_mall_nearby":         float(req.has_mall_nearby),
        "has_market_nearby":       float(req.has_market_nearby),
        "has_park_nearby":         float(req.has_park_nearby),
        "has_police_nearby":       float(req.has_police_nearby),
        "nearby_score":            nearby_score,
        "is_well_served":          is_well_served,
        # Phase 3: OSM distances
        "dist_hospital_km":        req.dist_hospital_km,
        "dist_school_km":          req.dist_school_km,
        "dist_metro_km":           req.dist_metro_km,
        "dist_market_km":          req.dist_market_km,
        "dist_park_km":            req.dist_park_km,
        # Age
        "age_bucket":              age_bucket,
        # Other
        "balcony":                 float(req.balcony),
        "parking":                 float(req.parking),
        "is_near_coaching":        0.0,
        "rera_approved":           0.0,
        # Interactions
        "bhk_x_city":              float(req.bhk) * city_encoded,
    }

    feat_list = STATE["feature_list"]
    if feat_list:
        return pd.DataFrame(
            [{f: fv.get(f, 0.0) for f in feat_list}],
            columns=feat_list,
        )
    return pd.DataFrame([fv])


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def ensemble_predict(X_df: pd.DataFrame) -> float:
    """
    Weighted ensemble prediction.

    prediction = expm1(
        w_xgb * mean(xgb_fold_preds) +
        w_lgb * mean(lgb_fold_preds) +
        w_cat * mean(cat_fold_preds)
    )

    expm1 converts from log_price space back to Crores.
    """
    w_xgb, w_lgb, w_cat = STATE["ensemble_weights"]
    weighted_sum = 0.0
    total_weight = 0.0

    if STATE["models_xgb"]:
        p = float(np.mean([m.predict(X_df)[0] for m in STATE["models_xgb"]]))
        weighted_sum += p * w_xgb
        total_weight += w_xgb

    if STATE["models_lgb"]:
        p = float(np.mean([m.predict(X_df)[0] for m in STATE["models_lgb"]]))
        weighted_sum += p * w_lgb
        total_weight += w_lgb

    if STATE["models_cat"] and CATBOOST_AVAILABLE:
        p = float(np.mean([m.predict(X_df)[0] for m in STATE["models_cat"]]))
        weighted_sum += p * w_cat
        total_weight += w_cat

    if total_weight == 0:
        raise HTTPException(503, "No models loaded")

    return float(np.expm1(weighted_sum / total_weight))


def get_shap_top3(X_df: pd.DataFrame) -> list:
    """
    Top-3 SHAP feature contributions for this prediction.

    Returns list of dicts: [{feature, shap_value, direction, impact}]
    Empty list if SHAP disabled or fails.
    """
    exp = STATE["shap_explainer"]
    if exp is None:
        return []

    try:
        sv    = exp.shap_values(X_df)
        vals  = np.array(sv).flatten()
        cols  = list(X_df.columns)

        if len(vals) != len(cols):
            return []

        top   = np.argsort(np.abs(vals))[::-1][:3]
        total = sum(abs(v) for v in vals) + 1e-9

        return [
            {
                "feature":    cols[i],
                "shap_value": round(float(vals[i]), 4),
                "direction":  "positive" if vals[i] > 0 else "negative",
                "impact":     round(abs(float(vals[i])) / total, 3),
            }
            for i in top
        ]
    except Exception as e:
        logger.debug(f"SHAP error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "name":      "PropML API",
        "version":   "3.0",
        "docs":      "/docs",
        "cities":    sorted(VALID_CITIES),
        "n_cities":  len(VALID_CITIES),
        "endpoints": ["/predict", "/recommend", "/analytics", "/health", "/model-info"],
    }


@app.get("/health")
def health():
    lat     = STATE["latencies"][-100:]
    version = "unknown"
    version_path = os.path.join(MODELS_DIR, "version.txt")
    if os.path.exists(version_path):
        version = open(version_path).read().strip()

    return {
        "status":       "healthy",
        "model_version": version,
        "models_loaded": {
            "xgb":  STATE["models_xgb"]      is not None,
            "lgb":  STATE["models_lgb"]      is not None,
            "cat":  STATE["models_cat"]      is not None,
            "shap": STATE["shap_explainer"]  is not None,
        },
        "uptime_seconds":     round(time.time() - STATE["start_time"], 1),
        "predictions_served": STATE["predictions_count"],
        "avg_latency_ms":     round(float(np.mean(lat)), 1) if lat else 0.0,
        "shap_mode":          STATE["shap_mode"],
        "n_features":         len(STATE["feature_list"]),
        "n_cities":           len(VALID_CITIES),
    }


@app.get("/model-info")
def model_info():
    return {
        "features":    STATE["feature_list"],
        "n_features":  len(STATE["feature_list"]),
        "weights": {
            "xgb": STATE["ensemble_weights"][0],
            "lgb": STATE["ensemble_weights"][1],
            "cat": STATE["ensemble_weights"][2],
        },
        "shap_mode":   STATE["shap_mode"],
        "n_xgb_folds": len(STATE["models_xgb"]) if STATE["models_xgb"] else 0,
        "n_lgb_folds": len(STATE["models_lgb"]) if STATE["models_lgb"] else 0,
        "n_cat_folds": len(STATE["models_cat"]) if STATE["models_cat"] else 0,
        "cities":      sorted(VALID_CITIES),
        "city_mapes":  CITY_CV_MAPE,
    }


@app.post("/predict")
def predict(req: PredictRequest, explain: bool = False):
    city = req.city.lower().strip()
    pt   = req.property_type.lower().strip()
    

    if city not in VALID_CITIES:
        raise HTTPException(422, f"city must be one of {sorted(VALID_CITIES)}")
    if pt not in VALID_TYPES:
        raise HTTPException(422, f"property_type must be one of {sorted(VALID_TYPES)}")
    if not STATE["models_xgb"] and not STATE["models_lgb"]:
        raise HTTPException(503, "Models not loaded — check /health")

    t0       = time.time()
    X_df     = build_feature_df(req)
    price_cr = ensemble_predict(X_df)
    shap3 = get_shap_top3(X_df) if explain else []
    lat      = round((time.time() - t0) * 1000, 1)

    STATE["predictions_count"] += 1
    STATE["latencies"].append(lat)

    # City-specific confidence interval from CV MAPE
    ci = CITY_CV_MAPE.get(city, 0.22)

    return {
        "prediction_cr": round(price_cr, 3),
        "confidence_interval": {
            "low":  round(price_cr * (1 - ci), 3),
            "high": round(price_cr * (1 + ci), 3),
            "note": f"±{int(ci*100)}% band based on CV MAPE for {city}",
        },
        "price_per_sqft":    int(price_cr * 1e7 / req.area_sqft),
        "shap_top_features": shap3,
        "shap_available":    len(shap3) > 0,
        "request_id":        str(uuid.uuid4())[:8],
        "latency_ms":        lat,
        "city":              city,
        "property_type":     pt,
    }


@app.post("/recommend")
def recommend(req: RecommendRequest):
    """
    Content-based property recommendations using cosine similarity.

    Finds properties similar to user's preference vector.
    Hard filters: city match, budget ±50%, BHK exact match.

    Phase 3: filters and scores use amenity_score, nearby_score
    if available in the dataset.

    Note: brute-force cosine similarity works up to ~30k properties.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    enc       = STATE["encoding_map"]
    city      = req.city.lower().strip()
    pt        = req.property_type.lower().strip()
    feat_list = STATE["feature_list"]
    gmean     = enc.get("city_global_mean", 0.75)

    if city not in VALID_CITIES:
        raise HTTPException(422, f"city must be one of {sorted(VALID_CITIES)}")
    if not feat_list:
        raise HTTPException(503, "Feature list empty — retrain model")

    # Build preference vector
    city_encoded = enc.get("city_map", {}).get(city, gmean)
    city_tier    = float(TIER_MAP.get(city, 2))
    area_mid     = float(((req.area_min or 1000) + (req.area_max or 1000)) / 2)

    pref = {f: 0.0 for f in feat_list}
    pref.update({
        "city_encoded":         city_encoded,
        "city_tier_num":        city_tier,
        "area_x_city_encoded":  area_mid * city_encoded,
        "area_x_city_tier":     area_mid * city_tier,
        "bhk":                  float(req.bhk),
        "area":                 area_mid,
        "area_per_bhk":         area_mid / max(req.bhk, 1),
        "log_area":             float(np.log1p(area_mid)),
        "amenity_score":        req.amenity_score or 1.0,
        "avg_rating":           4.0,
        "bathroom":             float(req.bhk),
        "bath_per_bed":         1.0,
        "is_flat":              float(pt == "flat"),
        "is_house":             float(pt == "house"),
        "is_independent_floor": float(pt == "independent_floor"),
        "is_plot":              float(pt == "plot"),
        "floor_pos":            3.0,
        "total_floors":         10.0,
        "relative_floor":       0.3,
        "furnish_score":        0.5,
        "locality_encoded":     gmean,
        "area_x_locality":      area_mid * gmean,
        "balcony":              1.0,
        "parking":              1.0,
        "bhk_x_city":           float(req.bhk) * city_encoded,
    })

    # Load property data
    df = None
    for path in [
        "data/features/combined_engineered.parquet",
        "data/cleaned/combined_cleaned.parquet",
    ]:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            break

    if df is None:
        raise HTTPException(503, "Property data not found. Run the pipeline first.")

    available_feats = [f for f in feat_list if f in df.columns]
    if not available_feats:
        raise HTTPException(503, "No matching features in dataset.")

    pref_vec = np.array([[pref.get(f, 0.0) for f in available_feats]])
    prop_mat = df[available_feats].fillna(0).values
    sims     = cosine_similarity(pref_vec, prop_mat)[0]

    # Hard filters
    if "city" in df.columns:
        sims[df["city"].str.lower().str.strip().values != city] = 0.0
    if "price" in df.columns:
        outside = ~df["price"].between(req.budget_cr * 0.5, req.budget_cr * 1.5)
        sims[outside.values] = 0.0
    if "bhk" in df.columns:
        sims[df["bhk"].values != req.bhk] = 0.0
    if req.area_min and "area" in df.columns:
        sims[df["area"].values < req.area_min] = 0.0 # type: ignore
    if req.area_max and "area" in df.columns:
        sims[df["area"].values > req.area_max] = 0.0 # type: ignore

    results = []
    for idx in np.argsort(sims)[::-1][:req.top_n]:
        if sims[idx] < 0.01:
            continue
        row = df.iloc[int(idx)]
        results.append({
            "property_idx":    int(idx),
            "similarity_score": round(float(sims[idx]), 4),
            "price_cr":        round(float(row.get("price", 0)), 3),
            "area_sqft":       round(float(row.get("area",  0)), 0),
            "bhk":             int(row.get("bhk", 0)),
            "city":            str(df["city"].iloc[int(idx)]) if "city" in df.columns else city,
            "property_type":   str(df["property_type"].iloc[int(idx)]) if "property_type" in df.columns else pt,
            "amenity_score":   round(float(row.get("amenity_score", 0)), 1) if "amenity_score" in df.columns else 0.0,
            "nearby_score":    round(float(row.get("nearby_score", 0)), 1) if "nearby_score" in df.columns else 0.0,
        })

    return {
        "recommendations": results,
        "count":           len(results),
        "filters": {
            "city":         city,
            "bhk":          req.bhk,
            "budget_cr":    req.budget_cr,
            "budget_range": f"Rs{req.budget_cr*0.5:.2f}Cr - Rs{req.budget_cr*1.5:.2f}Cr",
        },
    }


@app.get("/analytics")
def list_analytics():
    if not os.path.exists(ANALYTICS_DIR):
        return {
            "reports": [],
            "hint":    "Run: python src/analytics/analytics_engine.py",
        }
    reports = sorted(
        f[:-5] for f in os.listdir(ANALYTICS_DIR) if f.endswith(".json")
    )
    return {
        "reports":  reports,
        "base_url": "/analytics/{report_name}",
        "example":  "/analytics/market_summary",
    }


@app.get("/analytics/{report_name}")
def analytics(report_name: str):
    safe = report_name.replace("/", "").replace("..", "").replace("\\", "")
    if not safe.endswith(".json"):
        safe += ".json"

    path = os.path.join(ANALYTICS_DIR, safe)
    if not os.path.exists(path):
        available = []
        if os.path.exists(ANALYTICS_DIR):
            available = [f[:-5] for f in os.listdir(ANALYTICS_DIR) if f.endswith(".json")]
        raise HTTPException(
            404,
            {"error": f"'{report_name}' not found", "available": available},
        )

    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.serving.main:app", host="0.0.0.0", port=port, reload=False)