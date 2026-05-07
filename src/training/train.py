import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─── CatBoost optional import ─────────────────────────────────────────────────
CATBOOST_AVAILABLE = False
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    logger.warning("CatBoost not installed. Run: pip install catboost==1.2.2")

# ─── Paths ───────────────────────────────────────────────────────────────────
FEAT_PATH      = "data/features/combined_engineered.parquet"
FEAT_LIST_PATH = "data/features/feature_list.txt"
MODELS_DIR     = "models"
REPORTS_DIR    = "reports"

# ─── Config from params.yaml ─────────────────────────────────────────────────
def load_params() -> dict:
    path = Path("params.yaml")
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}

PARAMS = load_params()

TARGET              = "log_price"
CV_FOLDS            = PARAMS.get("training", {}).get("cv_folds", 5)
OPTUNA_TRIALS       = PARAMS.get("training", {}).get("optuna_trials", 100)
ENSEMBLE_TRIALS     = PARAMS.get("training", {}).get("optuna_ensemble_trials", 25)
RANDOM_STATE        = PARAMS.get("training", {}).get("random_state", 42)
PRODUCTION_GATE_MAPE = PARAMS.get("gate", {}).get("max_mape_pct", 22.0)
PRODUCTION_GATE_R2   = PARAMS.get("gate", {}).get("min_r2", 0.82)

TIER_WEIGHTS = {
    "Tier-1": 2.0,   # luxury — harder to predict, weight higher
    "Tier-2": 1.0,
    "Tier-3": 0.8,
}















def load_features() -> tuple:
    """
    Load feature matrix and ordered feature list.

    Returns (df, feature_names_list)

    Why load feature_list.txt:
      XGBoost/LGB use column order from training for prediction.
      feature_list.txt guarantees the API uses the EXACT same order.
      Auto-detecting features from the parquet is fragile —
      column order can change between pandas versions.
    """
    if not os.path.exists(FEAT_PATH):
        logger.error(f"Missing: {FEAT_PATH}")
        logger.error("Run: python src/features/feature_engineering.py")
        sys.exit(1)

    df = pd.read_parquet(FEAT_PATH)

    if os.path.exists(FEAT_LIST_PATH):
        with open(FEAT_LIST_PATH) as f:
            features = [l.strip() for l in f if l.strip()]
        # Validate all features exist in the parquet
        missing = [f for f in features if f not in df.columns]
        if missing:
            logger.warning(f"Features in list but missing from parquet: {missing}")
            features = [f for f in features if f in df.columns]
    else:
        # Auto-detect as fallback
        exclude = {TARGET, "price", "city", "property_type", "city_tier",
                   "sector", "age", "furnish", "nearbylocations", "floor", "facing"}
        features = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]
        logger.warning(f"feature_list.txt not found — auto-detected {len(features)} features")

    logger.info(f"Loaded: {len(df)} rows × {len(features)} features")
    logger.info(f"  Cities: {df['city'].value_counts().to_dict()}")
    return df, features
















def prepare_data(df: pd.DataFrame, features: list) -> tuple:
    """
    Prepare X, y, sample weights, and stratification labels.

    Sample weights (city tier):
      Tier-1 (luxury) properties are harder to predict — higher variance,
      fewer training examples. Weight them 2× to compensate.
      Normalized to mean=1 so overall loss scale stays comparable.

    Stratification (city × property_type × price_quartile):
      Simple price quantile stratification misses cross-city imbalance.
      Composite label ensures each fold has all city/type combinations.
      Prevents a fold where all Kota properties land in one split.

    Returns: (X, y, weights, strata, df_full)
    """
    X = df[features].copy().fillna(df[features].median(numeric_only=True))
    y = df[TARGET].copy()

    # Sample weights
    sample_weights = df["city_tier"].map(TIER_WEIGHTS).fillna(1.0).values
    sample_weights = sample_weights / sample_weights.mean()  # type: ignore # normalize to mean=1

    # Composite strata for stratified CV
    price_q = pd.cut(y, bins=3, labels=["low", "mid", "high"])
    strata   = (
        df["city"].astype(str)
        + "_" + df["property_type"].astype(str)
        + "_" + price_q.astype(str)
    ).values

    logger.info(f"X: {X.shape}  |  y: {y.shape}")
    logger.info(f"Sample weights: mean={sample_weights.mean():.3f}, std={sample_weights.std():.3f}")

    return X, y, sample_weights, strata, df












def mape_by_city(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df_full: pd.DataFrame,
    val_idx: np.ndarray,
) -> dict:
    """
    Compute MAPE per city for this validation fold.

    Why per-city MAPE:
      Overall MAPE 20% could mean Gurgaon 15% + Kota 35%.
      The per-city breakdown tells you which city needs more data
      or better features — actionable for the next sprint.
    """
    if "city" not in df_full.columns:
        return {}

    val_cities = df_full.iloc[val_idx]["city"]
    result = {}
    for city in val_cities.unique():
        mask = val_cities == city
        if mask.sum() < 5:
            continue
        city_mape = mean_absolute_percentage_error(
            np.expm1(y_true[mask.values]),
            np.expm1(y_pred[mask.values]),
        )
        result[city] = round(city_mape * 100, 2)

    return result













def ridge_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    strata: np.ndarray,
) -> tuple:
    """
    Ridge regression baseline — establishes the floor to beat.

    Why Ridge as baseline:
      Ridge is a strong linear baseline. If XGBoost cannot beat Ridge,
      something is wrong with the features or pipeline — not the model.
      'Our XGBoost beats Ridge by 13 percentage points' is a concrete
      number that communicates feature quality to stakeholders.

    Why StandardScaler inside each fold:
      Scaler fit on training fold only. Applying scaler fit on full
      dataset before CV leaks val distribution into training.
    """
    logger.info("\n" + "=" * 50)
    logger.info("BASELINE: Ridge Regression")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mapes, r2s = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, strata)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        scaler = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model = Ridge(alpha=10.0)
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_val_s)

        mapes.append(mean_absolute_percentage_error(np.expm1(y_val), np.expm1(y_pred)))
        r2s.append(r2_score(y_val, y_pred))

    cv_mape = np.mean(mapes) * 100
    cv_r2   = np.mean(r2s)
    logger.info(f"Ridge CV MAPE: {cv_mape:.2f}%  |  R²: {cv_r2:.4f}")
    return cv_mape, cv_r2













def xgboost_default(
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
    strata: np.ndarray,
) -> tuple:
    """
    XGBoost with sensible defaults — no tuning.

    objective='reg:squaredlogerror':
      Better than 'reg:squarederror' for log-transformed targets.
      Directly minimizes squared log error, which is proportionally
      aligned with how MAPE penalizes errors.

    Why this before Optuna:
      Default XGB should substantially beat Ridge.
      If it does not, the issue is in features — not hyperparameters.
      Stop and fix features before spending time on HPO.
    """
    logger.info("\n" + "=" * 50)
    logger.info("XGBoost (Default Parameters)")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mapes, r2s = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, strata)):
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squaredlogerror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            sample_weight=weights[train_idx],
        )
        y_pred = model.predict(X.iloc[val_idx])

        fold_mape = mean_absolute_percentage_error(np.expm1(y.iloc[val_idx]), np.expm1(y_pred))
        fold_r2   = r2_score(y.iloc[val_idx], y_pred)
        mapes.append(fold_mape)
        r2s.append(fold_r2)
        logger.info(f"  Fold {fold+1}: MAPE={fold_mape*100:.2f}%  R²={fold_r2:.4f}")

    cv_mape = np.mean(mapes) * 100
    cv_r2   = np.mean(r2s)
    logger.info(f"XGBoost Default CV MAPE: {cv_mape:.2f}%  |  R²: {cv_r2:.4f}")
    return cv_mape, cv_r2