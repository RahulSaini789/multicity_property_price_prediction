"""
src/training/train.py
PropML — Model training pipeline (Layer 7, Ensemble v2).

Input:  data/features/combined_engineered.parquet
        data/features/feature_list.txt

Output: models/models_xgb_ensemble.pkl
        models/models_lgb_ensemble.pkl
        models/models_cat_ensemble.pkl
        models/ensemble_weights.pkl
        models/feature_list.pkl
        models/target_encoding_map.json   (copied from features/ for API)
        models/version.txt
        reports/metrics.json
        reports/feature_importance.json
        reports/optuna_study.json

Phase 3 updates:
  - TIER_WEIGHTS loaded from params.yaml (was hardcoded)
  - prepare_data: strata includes all 13 cities
  - save_metrics_json: n_cities, n_rows, feature_count added
  - save_models: target_encoding_map.json copied to models/ for API
  - training_pipeline: per-city MAPE summary in final log

Run:
  python src/training/train.py
  python src/training/train.py --trials 200
"""

import argparse
import json
import logging
import os
import shutil
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


# ── CatBoost optional import ──────────────────────────────────────────────────
CATBOOST_AVAILABLE = False
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    logger.warning("CatBoost not installed. Run: pip install catboost==1.2.2")


# ── Paths ─────────────────────────────────────────────────────────────────────
FEAT_PATH      = "data/features/combined_engineered.parquet"
FEAT_LIST_PATH = "data/features/feature_list.txt"
ENC_MAP_PATH   = "data/features/target_encoding_map.json"
MODELS_DIR     = "models"
REPORTS_DIR    = "reports"


# ── Config from params.yaml ───────────────────────────────────────────────────
def load_params() -> dict:
    path = Path("params.yaml")
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}

PARAMS = load_params()

TARGET               = "log_price"
CV_FOLDS             = PARAMS.get("training", {}).get("cv_folds", 5)
OPTUNA_TRIALS        = PARAMS.get("training", {}).get("optuna_trials", 100)
ENSEMBLE_TRIALS      = PARAMS.get("training", {}).get("optuna_ensemble_trials", 25)
RANDOM_STATE         = PARAMS.get("training", {}).get("random_state", 42)
PRODUCTION_GATE_MAPE = PARAMS.get("gate", {}).get("max_mape_pct", 22.0)
PRODUCTION_GATE_R2   = PARAMS.get("gate", {}).get("min_r2", 0.82)

# Phase 3: TIER_WEIGHTS loaded from params.yaml (was hardcoded before)
# params.yaml: training.tier_weights: {Tier-1: 2.0, Tier-2: 1.5, Tier-3: 0.8}
_tier_weights_raw = PARAMS.get("training", {}).get("tier_weights", {})
TIER_WEIGHTS = {
    "Tier-1": float(_tier_weights_raw.get("Tier-1", 2.0)),
    "Tier-2": float(_tier_weights_raw.get("Tier-2", 1.5)),
    "Tier-3": float(_tier_weights_raw.get("Tier-3", 0.8)),
}
logger.debug(f"Tier weights: {TIER_WEIGHTS}")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

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
            features = [line.strip() for line in f if line.strip()]
        missing = [feat for feat in features if feat not in df.columns]
        if missing:
            logger.warning(f"Features in list but missing from parquet: {missing}")
            features = [feat for feat in features if feat in df.columns]
    else:
        exclude  = {TARGET, "price", "city", "property_type", "city_tier",
                    "sector", "age", "furnish", "nearbylocations", "floor", "facing"}
        features = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]
        logger.warning(f"feature_list.txt not found — auto-detected {len(features)} features")

    logger.info(f"Loaded: {len(df)} rows x {len(features)} features")
    logger.info(f"  Cities: {df['city'].value_counts().to_dict()}")
    return df, features


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_data(df: pd.DataFrame, features: list) -> tuple:
    """
    Prepare X, y, sample weights, and stratification labels.

    Sample weights (city tier):
      Tier-1 (luxury): harder to predict, fewer examples -> weight higher.
      Values from params.yaml training.tier_weights.
      Normalized to mean=1 so overall loss scale stays comparable.

    Stratification (city x property_type x price_quartile):
      Composite label ensures each fold has all city/type combinations.
      Prevents a fold where all Kota properties land in one split.
      Phase 3: works with all 13 cities automatically.

    Returns: (X, y, weights, strata, df_full)
    """
    X = df[features].copy().fillna(df[features].median(numeric_only=True))
    y = df[TARGET].copy()

    # Sample weights from params.yaml tier_weights
    sample_weights = df["city_tier"].map(TIER_WEIGHTS).fillna(1.0).values
    sample_weights = sample_weights / sample_weights.mean()   # type: ignore # normalize to mean=1

    # Composite strata for stratified CV
    price_q = pd.cut(y, bins=4, labels=["q1", "q2", "q3", "q4"])
    strata   = (
        df["city"].astype(str)
        + "_" + df["property_type"].astype(str)
        + "_" + price_q.astype(str)
    ).values

    logger.info(f"X: {X.shape}  |  y: {y.shape}")
    logger.info(f"Sample weights: mean={sample_weights.mean():.3f}, std={sample_weights.std():.3f}")

    return X, y, sample_weights, strata, df


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

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
      Per-city breakdown tells you which city needs more data
      or better features — actionable for the next sprint.

    Phase 3: works with all 13 cities automatically.
    """
    if "city" not in df_full.columns:
        return {}

    val_cities = df_full.iloc[val_idx]["city"]
    result     = {}
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


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

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
      'Our XGBoost beats Ridge by N percentage points' is a concrete
      number that communicates feature quality to stakeholders.

    Why StandardScaler inside each fold:
      Scaler fit on training fold only. Applying scaler fit on full
      dataset before CV leaks val distribution into training.
    """
    logger.info("\n" + "=" * 50)
    logger.info("BASELINE: Ridge Regression")

    skf        = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mapes, r2s = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, strata)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model  = Ridge(alpha=10.0)
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_val_s)

        mapes.append(mean_absolute_percentage_error(np.expm1(y_val), np.expm1(y_pred)))
        r2s.append(r2_score(y_val, y_pred))

    cv_mape = np.mean(mapes) * 100
    cv_r2   = np.mean(r2s)
    logger.info(f"Ridge CV MAPE: {cv_mape:.2f}%  |  R²: {cv_r2:.4f}")
    return cv_mape, cv_r2


# ═══════════════════════════════════════════════════════════════════════════════
# XGBOOST DEFAULT
# ═══════════════════════════════════════════════════════════════════════════════

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
      Directly minimizes squared log error, proportionally aligned
      with how MAPE penalizes errors.

    Why this before Optuna:
      Default XGB should substantially beat Ridge.
      If it does not, the issue is features — not hyperparameters.
    """
    logger.info("\n" + "=" * 50)
    logger.info("XGBoost (Default Parameters)")

    skf        = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
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


# ═══════════════════════════════════════════════════════════════════════════════
# OPTUNA TUNING
# ═══════════════════════════════════════════════════════════════════════════════

def _xgb_objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
    strata: np.ndarray,
) -> float:
    """
    Optuna objective for XGBoost hyperparameter search.

    Uses CV_FOLDS-fold CV (same as final evaluation) to prevent
    CV mismatch — Optuna used to use 3-fold while final used 5-fold,
    causing best params to appear worse at final evaluation.

    TPE sampler learns which parameter regions give good results
    and focuses sampling there.
    """
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 300, 1500),
        "max_depth":        trial.suggest_int("max_depth", 4, 12),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 15.0),
        "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
        "objective":        "reg:squaredlogerror",
        "random_state":     RANDOM_STATE,
        "n_jobs":           -1,
        "verbosity":        0,
    }

    # CV_FOLDS-fold — same as final evaluation to prevent CV mismatch
    skf   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mapes = []

    for train_idx, val_idx in skf.split(X, strata):
        model = xgb.XGBRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], sample_weight=weights[train_idx])
        y_pred = model.predict(X.iloc[val_idx])
        mapes.append(
            mean_absolute_percentage_error(np.expm1(y.iloc[val_idx]), np.expm1(y_pred))
        )

    return float(np.mean(mapes))


def run_optuna_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
    strata: np.ndarray,
    n_trials: int = OPTUNA_TRIALS,
) -> tuple:
    """
    Run Optuna TPE hyperparameter search for XGBoost.
    Returns (best_params_dict, best_mape_pct)
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Optuna Tuning - {n_trials} trials (TPE sampler)")

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        direction="minimize",
        study_name="propml_xgb_mape",
    )
    study.optimize(
        lambda trial: _xgb_objective(trial, X, y, weights, strata),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,   # n_jobs>1 with TPE causes race conditions
    )

    best_params = study.best_params
    best_mape   = study.best_value * 100

    logger.info(f"Best MAPE ({CV_FOLDS}-fold): {best_mape:.2f}%")
    logger.info(f"Best params: {best_params}")

    Path(REPORTS_DIR).mkdir(exist_ok=True)
    with open(f"{REPORTS_DIR}/optuna_study.json", "w") as f:
        json.dump({
            "best_mape_pct": round(best_mape, 3),
            "best_params":   best_params,
            "n_trials":      n_trials,
            "cv_folds":      CV_FOLDS,
            "timestamp":     datetime.now().isoformat(),
        }, f, indent=2)

    return best_params, best_mape


# ═══════════════════════════════════════════════════════════════════════════════
# LIGHTGBM
# ═══════════════════════════════════════════════════════════════════════════════

def train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
    strata: np.ndarray,
) -> tuple:
    """
    LightGBM CV — independent from XGBoost.

    Why LightGBM separately:
      LGB and XGB make different types of errors.
      LGB uses leaf-wise growth (faster, better on large data).
      XGB uses level-wise growth (more regularized).
      Their predictions are weakly correlated -> ensemble benefits.

    objective='regression' with metric='mae':
      Optimizes MAE in log space — closer to MAPE than MSE.
    """
    logger.info(f"\n{'='*50}")
    logger.info("LightGBM (Default Parameters)")

    skf        = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mapes, r2s = [], []

    lgb_params = {
        "num_leaves":       63,
        "learning_rate":    0.05,
        "n_estimators":     700,
        "min_child_samples": 10,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "objective":        "regression",
        "metric":           "mae",
        "random_state":     RANDOM_STATE,
        "verbose":          -1,
        "n_jobs":           -1,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, strata)):
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            sample_weight=weights[train_idx],
            callbacks=[lgb.log_evaluation(period=-1)],
        )
        y_pred = model.predict(X.iloc[val_idx])

        fold_mape = mean_absolute_percentage_error(np.expm1(y.iloc[val_idx]), np.expm1(y_pred)) # type: ignore
        fold_r2   = r2_score(y.iloc[val_idx], y_pred) # type: ignore
        mapes.append(fold_mape)
        r2s.append(fold_r2)
        logger.info(f"  Fold {fold+1}: MAPE={fold_mape*100:.2f}%  R²={fold_r2:.4f}")

    cv_mape = np.mean(mapes) * 100
    cv_r2   = np.mean(r2s)
    logger.info(f"LightGBM CV MAPE: {cv_mape:.2f}%  |  R²: {cv_r2:.4f}")
    return cv_mape, cv_r2


# ═══════════════════════════════════════════════════════════════════════════════
# CATBOOST
# ═══════════════════════════════════════════════════════════════════════════════

def train_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
    strata: np.ndarray,
) -> tuple:
    """
    CatBoost with direct MAPE loss optimization.

    Why CatBoost is different:
      XGBoost/LGB optimize MSE in log space as a proxy for MAPE.
      CatBoost with loss_function='MAPE' directly minimizes the metric
      we care about. Typically saves 1-2 MAPE points vs the proxy.

      CatBoost's symmetric tree growth is more resistant to overfitting
      on small city datasets (Kota, Chandigarh).
    """
    if not CATBOOST_AVAILABLE:
        logger.warning("CatBoost not installed — skipping")
        return 999.0, 0.0

    logger.info(f"\n{'='*50}")
    logger.info("CatBoost (Direct MAPE Optimization)")

    skf        = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mapes, r2s = [], []

    cb_params = {
        "iterations":          700,
        "learning_rate":       0.05,
        "depth":               8,
        "l2_leaf_reg":         3.0,
        "subsample":           0.8,
        "loss_function":       "MAPE",
        "eval_metric":         "MAPE",
        "random_seed":         RANDOM_STATE,
        "verbose":             False,
        "allow_writing_files": False,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, strata)):
        model = CatBoostRegressor(**cb_params) # type: ignore
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            sample_weight=weights[train_idx],
            eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
            early_stopping_rounds=50,
        )
        y_pred = model.predict(X.iloc[val_idx])

        fold_mape = mean_absolute_percentage_error(np.expm1(y.iloc[val_idx]), np.expm1(y_pred))
        fold_r2   = r2_score(y.iloc[val_idx], y_pred)
        mapes.append(fold_mape)
        r2s.append(fold_r2)
        logger.info(f"  Fold {fold+1}: MAPE={fold_mape*100:.2f}%  R²={fold_r2:.4f}")

    cv_mape = np.mean(mapes) * 100
    cv_r2   = np.mean(r2s)
    logger.info(f"CatBoost CV MAPE: {cv_mape:.2f}%  |  R²: {cv_r2:.4f}")
    return cv_mape, cv_r2


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE WEIGHT OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════════

def _ensemble_objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
    strata: np.ndarray,
    best_xgb_params: dict,
) -> float:
    """
    Optuna objective to find optimal ensemble blend weights.

    Treats (w_xgb, w_lgb, w_cat) as hyperparameters.
    w_cat = 1 - w_xgb - w_lgb is derived (must sum to 1).
    Constraint: each weight between 0.05 and 0.70.
    Prevents degenerate solutions (one model with weight 0.98).
    Uses 3-fold CV for speed (not final evaluation).
    """
    w_xgb = trial.suggest_float("w_xgb", 0.15, 0.65)
    w_lgb = trial.suggest_float("w_lgb", 0.15, 0.65)
    w_cat = 1.0 - w_xgb - w_lgb

    if w_cat < 0.05 or w_cat > 0.70:
        return 1.0  # Invalid — penalize heavily

    skf   = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    mapes = []

    for train_idx, val_idx in skf.split(X, strata):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_tr        = weights[train_idx]

        # XGBoost
        xgb_params = {
            **best_xgb_params,
            "objective":    "reg:squaredlogerror",
            "random_state": RANDOM_STATE,
            "n_jobs":       -1,
            "verbosity":    0,
        }
        m_xgb = xgb.XGBRegressor(**xgb_params)
        m_xgb.fit(X_tr, y_tr, sample_weight=w_tr)

        # LightGBM
        m_lgb = lgb.LGBMRegressor(
            num_leaves=63, learning_rate=0.05, n_estimators=700,
            objective="regression", verbose=-1, n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        m_lgb.fit(X_tr, y_tr, sample_weight=w_tr,
                  callbacks=[lgb.log_evaluation(period=-1)])

        # CatBoost or fallback to LGB
        if CATBOOST_AVAILABLE:
            m_cat    = CatBoostRegressor( # type: ignore
                iterations=700, learning_rate=0.05, depth=8,
                loss_function="MAPE", verbose=False,
                allow_writing_files=False, random_seed=RANDOM_STATE,
            )
            m_cat.fit(X_tr, y_tr, sample_weight=w_tr)
            pred_cat  = m_cat.predict(X_val)
            w_lgb_eff = w_lgb
            w_cat_eff = w_cat
        else:
            pred_cat  = m_lgb.predict(X_val)
            w_lgb_eff = w_lgb + w_cat
            w_cat_eff = 0.0

        pred = (
            w_xgb    * m_xgb.predict(X_val)
            + w_lgb_eff * m_lgb.predict(X_val) # type: ignore
            + w_cat_eff * pred_cat # type: ignore
        )
        mapes.append(
            mean_absolute_percentage_error(np.expm1(y_val), np.expm1(pred))
        )

    return float(np.mean(mapes))


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

def train_final_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
    strata: np.ndarray,
    df_full: pd.DataFrame,
    best_xgb_params: dict,
) -> dict:
    """
    Train the final ensemble with optimized weights.

    Steps:
    A. Find optimal (w_xgb, w_lgb, w_cat) via Optuna
    B. Run full CV_FOLDS-fold CV with those weights
    C. Log everything to MLflow
    D. Save per-city MAPE breakdown, feature importance

    Returns dict with models, metrics, weights for caller to save.
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Finding optimal ensemble weights ({ENSEMBLE_TRIALS} trials)...")

    weight_study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        direction="minimize",
    )
    weight_study.optimize(
        lambda trial: _ensemble_objective(
            trial, X, y, weights, strata, best_xgb_params
        ),
        n_trials=ENSEMBLE_TRIALS,
        show_progress_bar=True,
        n_jobs=1,
    )

    best_w = weight_study.best_params
    w_xgb  = best_w["w_xgb"]
    w_lgb  = best_w["w_lgb"]
    w_cat  = 1.0 - w_xgb - w_lgb
    logger.info(f"Optimal weights - XGB: {w_xgb:.3f}, LGB: {w_lgb:.3f}, CAT: {w_cat:.3f}")

    # ── Full CV_FOLDS-fold CV with final weights ────────────────────────
    logger.info(f"\nFinal {CV_FOLDS}-fold CV with ensemble weights...")
    mlflow.set_experiment("propml-v2-multi-city")

    with mlflow.start_run(
        run_name=f"ensemble-{datetime.now():%Y%m%d-%H%M}"
    ) as run:

        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        mapes, r2s           = [], []
        city_mapes_all       = {}
        models_xgb, models_lgb, models_cat = [], [], []
        feat_importance      = np.zeros(X.shape[1])

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, strata)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            w_tr        = weights[train_idx]

            # XGBoost
            xgb_p = {
                **best_xgb_params,
                "objective":    "reg:squaredlogerror",
                "random_state": RANDOM_STATE,
                "n_jobs":       -1,
                "verbosity":    0,
            }
            m_xgb = xgb.XGBRegressor(**xgb_p)
            m_xgb.fit(X_tr, y_tr, sample_weight=w_tr)
            models_xgb.append(m_xgb)
            feat_importance += np.array(m_xgb.feature_importances_)

            # LightGBM
            m_lgb = lgb.LGBMRegressor(
                num_leaves=63, learning_rate=0.05, n_estimators=700,
                min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                objective="regression", verbose=-1, n_jobs=-1,
                random_state=RANDOM_STATE,
            )
            m_lgb.fit(X_tr, y_tr, sample_weight=w_tr,
                      callbacks=[lgb.log_evaluation(period=-1)])
            models_lgb.append(m_lgb)

            # CatBoost
            if CATBOOST_AVAILABLE:
                m_cat = CatBoostRegressor( # type: ignore
                    iterations=700, learning_rate=0.05, depth=8,
                    loss_function="MAPE", verbose=False,
                    allow_writing_files=False, random_seed=RANDOM_STATE,
                )
                m_cat.fit(X_tr, y_tr, sample_weight=w_tr,
                          eval_set=(X_val, y_val), early_stopping_rounds=50)
                pred_cat  = m_cat.predict(X_val)
                models_cat.append(m_cat)
                w_lgb_eff = w_lgb
                w_cat_eff = w_cat
            else:
                pred_cat  = m_lgb.predict(X_val)
                w_lgb_eff = w_lgb + w_cat
                w_cat_eff = 0.0

            y_pred = (
                w_xgb    * m_xgb.predict(X_val)
                + w_lgb_eff * m_lgb.predict(X_val)
                + w_cat_eff * pred_cat # type: ignore
            )

            fold_mape = mean_absolute_percentage_error(np.expm1(y_val), np.expm1(y_pred))
            fold_r2   = r2_score(y_val, y_pred)
            mapes.append(fold_mape)
            r2s.append(fold_r2)

            city_breakdown = mape_by_city(y_val.values, y_pred, df_full, val_idx)
            for city, city_mape in city_breakdown.items():
                city_mapes_all.setdefault(city, []).append(city_mape)

            logger.info(
                f"  Fold {fold+1}: MAPE={fold_mape*100:.2f}%  R²={fold_r2:.4f}"
                f"  | {city_breakdown}"
            )

        # ── Final metrics ──────────────────────────────────────────────
        cv_mape        = np.mean(mapes) * 100
        cv_r2          = np.mean(r2s)
        feat_importance /= CV_FOLDS

        logger.info(f"\nFinal CV MAPE: {cv_mape:.2f}%")
        logger.info(f"Final CV R²:   {cv_r2:.4f}")

        logger.info("\nMAPE by city:")
        for city, vals in sorted(city_mapes_all.items(), key=lambda x: np.mean(x[1])):
            logger.info(f"  {city}: {np.mean(vals):.2f}%")

        # ── Feature importance ─────────────────────────────────────────
        feat_imp_dict = dict(sorted(
            zip(X.columns, feat_importance.tolist()),
            key=lambda x: x[1], reverse=True,
        ))
        logger.info("\nTop 10 Features:")
        for i, (feat, imp) in enumerate(list(feat_imp_dict.items())[:10]):
            bar = "█" * int(imp / feat_importance.max() * 20)
            logger.info(f"  {i+1:2d}. {feat:25s} {bar} {imp:.4f}")

        # ── MLflow logging ─────────────────────────────────────────────
        mlflow.log_params({
            "ensemble_type":      "xgb_lgb_catboost",
            "w_xgb":              round(w_xgb, 3),
            "w_lgb":              round(w_lgb, 3),
            "w_cat":              round(w_cat, 3),
            "catboost_available": CATBOOST_AVAILABLE,
            "cv_folds":           CV_FOLDS,
            "optuna_trials":      OPTUNA_TRIALS,
            "tier_w_tier1":       TIER_WEIGHTS.get("Tier-1"),
            "tier_w_tier2":       TIER_WEIGHTS.get("Tier-2"),
            "tier_w_tier3":       TIER_WEIGHTS.get("Tier-3"),
            **{k: round(v, 4) for k, v in best_xgb_params.items()},
        })
        mlflow.log_metrics({
            "cv_mape_pct": float(round(cv_mape, 4)),
            "cv_r2":       float(round(cv_r2, 4)),
            **{f"city_mape_{c}": float(round(float(np.mean(v)), 4))
               for c, v in city_mapes_all.items()},
        })

        # Feature importance artifact
        Path(REPORTS_DIR).mkdir(exist_ok=True)
        fi_path = f"{REPORTS_DIR}/feature_importance.json"
        with open(fi_path, "w") as f:
            json.dump({k: round(float(v), 6) for k, v in feat_imp_dict.items()}, f, indent=2)
        mlflow.log_artifact(fi_path)

        run_id = run.info.run_id

    return {
        "cv_mape":          cv_mape,
        "cv_r2":            cv_r2,
        "models_xgb":       models_xgb,
        "models_lgb":       models_lgb,
        "models_cat":       models_cat,
        "feat_importance":  feat_imp_dict,
        "ensemble_weights": (w_xgb, w_lgb, w_cat),
        "city_mapes":       {c: round(float(np.mean(v)), 4) for c, v in city_mapes_all.items()},
        "run_id":           run_id,
        "n_features":       X.shape[1],
        "n_rows":           len(X),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SHAP
# ═══════════════════════════════════════════════════════════════════════════════

def init_shap(models_xgb: list, features: list) -> tuple:
    """
    Initialize SHAP KernelExplainer for the ensemble.

    Why KernelExplainer and NOT TreeExplainer:
      This XGBoost version stores leaf values as JSON arrays
      (e.g. [3.8188547E-1]). SHAP's C parser calls float(str) on these
      -> TypeError. Even get_booster() does not bypass it.
      KernelExplainer only calls model.predict() — never inspects
      model internals. Always works regardless of XGBoost version.

    Returns (explainer, mode_string)
    """
    try:
        import shap

        model = models_xgb[0]
        bg_df = None

        for path in [
            "data/features/combined_engineered.parquet",
            "data/cleaned/combined_cleaned.parquet",
        ]:
            if os.path.exists(path):
                raw    = pd.read_parquet(path)
                cols   = [f for f in features if f in raw.columns]
                if cols:
                    sample = raw[cols].fillna(0).sample(min(50, len(raw)), random_state=42)
                    for feat in features:
                        if feat not in sample.columns:
                            sample[feat] = 0.0
                    bg_df = sample[features]
                break

        if bg_df is None:
            bg_df = pd.DataFrame(np.zeros((1, len(features))), columns=features)

        def predict_fn(X_arr):
            X_df = pd.DataFrame(X_arr, columns=features)
            return model.predict(X_df)

        explainer = shap.KernelExplainer(predict_fn, bg_df)
        logger.info(f"SHAP KernelExplainer initialized (background={len(bg_df)} rows)")
        return explainer, "kernel"

    except Exception as e:
        logger.warning(f"SHAP disabled: {e}")
        return None, "disabled"


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL SAVING
# ═══════════════════════════════════════════════════════════════════════════════

def save_models(results: dict, features: list) -> None:
    """
    Save all ensemble models + metadata to models/ directory.

    Files saved:
      models/models_xgb_ensemble.pkl    — list of CV_FOLDS XGBoost models
      models/models_lgb_ensemble.pkl    — list of CV_FOLDS LightGBM models
      models/models_cat_ensemble.pkl    — list of CV_FOLDS CatBoost models
      models/ensemble_weights.pkl       — (w_xgb, w_lgb, w_cat) tuple
      models/feature_list.pkl           — ordered feature names list
      models/target_encoding_map.json   — copied from features/ for API use
      models/version.txt                — semantic version string

    Phase 3: target_encoding_map.json copied to models/ so the API
    doesn't need to read from data/features/ at inference time.

    Why CV_FOLDS models per algorithm (not 1):
      At inference: prediction = mean(fold_1.predict, ..., fold_N.predict)
      Ensemble of folds reduces variance vs a single model.
    """
    Path(MODELS_DIR).mkdir(exist_ok=True)

    joblib.dump(results["models_xgb"],       f"{MODELS_DIR}/models_xgb_ensemble.pkl")
    joblib.dump(results["models_lgb"],       f"{MODELS_DIR}/models_lgb_ensemble.pkl")
    joblib.dump(results["ensemble_weights"], f"{MODELS_DIR}/ensemble_weights.pkl")
    joblib.dump(features,                   f"{MODELS_DIR}/feature_list.pkl")

    if results["models_cat"]:
        joblib.dump(results["models_cat"],   f"{MODELS_DIR}/models_cat_ensemble.pkl")

    # Phase 3: copy target_encoding_map.json to models/ for API
    if os.path.exists(ENC_MAP_PATH):
        shutil.copy2(ENC_MAP_PATH, f"{MODELS_DIR}/target_encoding_map.json")
        logger.info(f"  Encoding map copied -> {MODELS_DIR}/target_encoding_map.json")
    else:
        logger.warning(f"  ENC_MAP_PATH not found: {ENC_MAP_PATH}")

    # Semantic version: read current, increment patch
    version_file = f"{MODELS_DIR}/version.txt"
    if os.path.exists(version_file):
        curr          = open(version_file).read().strip()
        major, minor, patch = curr.split(".")
        new_version   = f"{major}.{minor}.{int(patch)+1}"
    else:
        new_version = "1.0.0"

    with open(version_file, "w") as f:
        f.write(new_version)

    logger.info(f"Models saved -> {MODELS_DIR}/  (version: {new_version})")


# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCTION GATE
# ═══════════════════════════════════════════════════════════════════════════════

def production_gate(cv_mape: float, cv_r2: float) -> bool:
    logger.info(f"\n{'='*50}")
    logger.info("Production Gate Check:")

    gate_pass = True

    if cv_mape > PRODUCTION_GATE_MAPE:
        logger.error(f"  FAIL: MAPE {cv_mape:.2f}% > {PRODUCTION_GATE_MAPE}%")
        gate_pass = False
    else:
        logger.info(f"  PASS: MAPE {cv_mape:.2f}% <= {PRODUCTION_GATE_MAPE}%")

    if cv_r2 < PRODUCTION_GATE_R2:
        logger.error(f"  FAIL: R2 {cv_r2:.4f} < {PRODUCTION_GATE_R2}")
        gate_pass = False
    else:
        logger.info(f"  PASS: R2 {cv_r2:.4f} >= {PRODUCTION_GATE_R2}")

    status = "APPROVED" if gate_pass else "REJECTED"
    logger.info(f"\n  Model: {status}")
    return gate_pass


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS JSON
# ═══════════════════════════════════════════════════════════════════════════════

def save_metrics_json(
    results: dict,
    baseline_mape: float,
    gate_pass: bool,
) -> None:
    """
    Save final metrics to reports/metrics.json.

    This file is read by:
    - GitHub Actions ci.yml (model gate step)
    - DVC metrics (dvc metrics show, dvc metrics diff)
    - Grafana dashboard (via /analytics endpoint)

    Phase 3 additions:
    - n_cities: number of cities in training data
    - n_rows: total training rows
    - feature_count: number of features
    - city_mapes: per-city MAPE breakdown (all 13 cities)
    - tier_weights: actual weights used (from params.yaml)

    DVC metrics format: flat dict of scalar values.
    """
    Path(REPORTS_DIR).mkdir(exist_ok=True)

    metrics = {
        # Core metrics
        "cv_mape":                  round(results["cv_mape"], 4),
        "cv_r2":                    round(results["cv_r2"], 4),
        "production_gate":          "PASS" if gate_pass else "FAIL",
        "mlflow_run_id":            results["run_id"],
        "timestamp":                datetime.now().isoformat(),
        "model_type":               "ensemble_xgb_lgb_catboost",
        # Baseline comparison
        "baseline_ridge_mape":      round(baseline_mape, 4),
        "improvement_vs_baseline":  round(baseline_mape - results["cv_mape"], 4),
        # Ensemble configuration
        "ensemble_weights": {
            "xgb": round(results["ensemble_weights"][0], 3),
            "lgb": round(results["ensemble_weights"][1], 3),
            "cat": round(results["ensemble_weights"][2], 3),
        },
        # Phase 3: data provenance
        "n_rows":        results.get("n_rows", 0),
        "n_cities":      len(results.get("city_mapes", {})),
        "feature_count": results.get("n_features", 0),
        # Phase 3: per-city breakdown
        "city_mapes":    results["city_mapes"],
        # Phase 3: tier weights used
        "tier_weights": {
            "Tier-1": TIER_WEIGHTS.get("Tier-1"),
            "Tier-2": TIER_WEIGHTS.get("Tier-2"),
            "Tier-3": TIER_WEIGHTS.get("Tier-3"),
        },
        # CV config
        "cv_folds":      CV_FOLDS,
        "optuna_trials": OPTUNA_TRIALS,
    }

    with open(f"{REPORTS_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved -> {REPORTS_DIR}/metrics.json")


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_analytics_engine() -> None:
    """
    Run pre-computed analytics after training completes.

    Generates JSON reports in reports/analytics/ for the FastAPI
    /analytics/{name} endpoint. Pre-computed so API response < 5ms.
    """
    analytics_script = Path("src/analytics/analytics_engine.py")
    if not analytics_script.exists():
        logger.warning("analytics_engine.py not found — skipping analytics generation")
        return

    logger.info("\nRunning analytics engine...")
    import subprocess
    result = subprocess.run(
        ["python", str(analytics_script)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        logger.info("Analytics engine complete")
    else:
        logger.warning(f"Analytics engine failed: {result.stderr[:200]}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def training_pipeline(n_optuna_trials: int = OPTUNA_TRIALS) -> dict:
    """
    Full training pipeline — Layer 7.

    Steps:
      1.  Load engineered features
      2.  Ridge baseline
      3.  XGBoost default
      4.  Optuna hyperparameter tuning
      5.  LightGBM
      6.  CatBoost
      7.  Ensemble weight optimization
      8.  Final CV_FOLDS-fold ensemble CV
      9.  SHAP initialization
      10. Model saving (+ encoding map copy)
      11. Production gate check
      12. metrics.json (Phase 3: n_cities, n_rows, city_mapes)
      13. Analytics engine
    """
    logger.info("=" * 60)
    logger.info("Layer 7: Model Training Pipeline (Ensemble v2)")
    logger.info("=" * 60)

    # ── 1. Load + prepare ──────────────────────────────────────────────
    df, features                    = load_features()
    X, y, weights, strata, df_full  = prepare_data(df, features)

    # ── 2-6. Performance ladder ────────────────────────────────────────
    baseline_mape, _              = ridge_baseline(X, y, strata)
    xgb_mape, _                   = xgboost_default(X, y, weights, strata)
    best_xgb_params, optuna_mape  = run_optuna_tuning(X, y, weights, strata, n_optuna_trials)
    lgb_mape, _                   = train_lightgbm(X, y, weights, strata)
    cat_mape, _                   = train_catboost(X, y, weights, strata) if CATBOOST_AVAILABLE else (999.0, 0.0)

    logger.info(f"\n{'='*55}")
    logger.info("Performance Ladder Summary:")
    logger.info(f"  Ridge baseline:        {baseline_mape:.2f}%")
    logger.info(f"  XGBoost default:       {xgb_mape:.2f}%")
    logger.info(f"  XGBoost Optuna:        {optuna_mape:.2f}%")
    logger.info(f"  LightGBM:              {lgb_mape:.2f}%")
    if CATBOOST_AVAILABLE:
        logger.info(f"  CatBoost (MAPE loss):  {cat_mape:.2f}%")
    best_single = min(xgb_mape, lgb_mape, cat_mape if CATBOOST_AVAILABLE else 999.0)
    logger.info(f"  Ensemble (expected):   < {best_single:.2f}%")

    # ── 7-8. Final ensemble ────────────────────────────────────────────
    results = train_final_ensemble(
        X, y, weights, strata, df_full, best_xgb_params
    )

    # ── 9. SHAP ────────────────────────────────────────────────────────
    shap_explainer, shap_mode = init_shap(results["models_xgb"], features)
    results["shap_explainer"] = shap_explainer
    results["shap_mode"]      = shap_mode

    # ── 10. Save models ────────────────────────────────────────────────
    save_models(results, features)

    # ── 11. Production gate ────────────────────────────────────────────
    gate_pass = production_gate(results["cv_mape"], results["cv_r2"])

    # ── 12. metrics.json ───────────────────────────────────────────────
    n_cities  = len(results.get("city_mapes", {}))
    n_rows    = len(X)
    n_feats   = X.shape[1]
    results["n_cities"]   = n_cities
    results["n_rows"]     = n_rows
    results["n_features"] = n_feats
    save_metrics_json(results, baseline_mape, gate_pass)

    # ── 13. Analytics engine ───────────────────────────────────────────
    run_analytics_engine()

    # ── Final summary ──────────────────────────────────────────────────
    logger.info(f"\n{'='*55}")
    logger.info(f"Training complete")
    logger.info(f"   Final MAPE:     {results['cv_mape']:.2f}%")
    logger.info(f"   Final R2:       {results['cv_r2']:.4f}")
    logger.info(f"   Improvement:    {baseline_mape - results['cv_mape']:.2f}pp over Ridge")
    logger.info(f"   Gate:           {'PASS' if gate_pass else 'FAIL'}")
    logger.info(f"   MLflow run:     {results['run_id']}")
    n_cities = len(results.get("city_mapes", {}))
    n_rows   = len(X)
    n_feats  = X.shape[1]
    logger.info(f"   Cities:         {n_cities} ({n_rows} rows)")
    logger.info(f"   Features:       {n_feats}")

    # Per-city MAPE in final summary
    logger.info(f"\nPer-city MAPE (final):")
    for city, mape in sorted(results["city_mapes"].items(), key=lambda x: x[1]):
        logger.info(f"   {city:<15} {mape:.2f}%")

    if not gate_pass:
        logger.error("Production gate FAILED. Exiting with code 1.")
        sys.exit(1)

    return results


def main():
    parser = argparse.ArgumentParser(description="PropML Training Pipeline")
    parser.add_argument(
        "--trials", type=int, default=OPTUNA_TRIALS,
        help=f"Optuna trials (default: {OPTUNA_TRIALS})"
    )
    args = parser.parse_args()
    training_pipeline(n_optuna_trials=args.trials)


if __name__ == "__main__":
    main()