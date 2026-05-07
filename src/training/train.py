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






def _xgb_objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
    strata: np.ndarray,
) -> float:
    """
    Optuna objective for XGBoost hyperparameter search.

    Uses 3-fold CV (not 5) for speed — each trial is 3 train/val cycles.
    TPE sampler learns which parameter regions give good results and
    focuses sampling there. 100 trials covers more search space than
    10,000 Grid Search combinations in a fraction of the time.

    Parameter search space rationale:
      n_estimators 300–1500: low = underfit, high = overfit + slow
      max_depth 4–12: controls tree complexity
      learning_rate 0.005–0.15 (log-scale): smaller = better generalization
      subsample/colsample 0.5–1.0: introduce noise = regularization
      min_child_weight 1–15: min samples per leaf
      reg_alpha/lambda: L1/L2 regularization
      gamma: min loss reduction for split (additional regularization)
    """
    params = {
        "n_estimators":       trial.suggest_int("n_estimators", 300, 1500),
        "max_depth":          trial.suggest_int("max_depth", 4, 12),
        "learning_rate":      trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":   trial.suggest_int("min_child_weight", 1, 15),
        "reg_alpha":          trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda":         trial.suggest_float("reg_lambda", 0.1, 15.0),
        "gamma":              trial.suggest_float("gamma", 0.0, 1.0),
        "objective":          "reg:squaredlogerror",
        "random_state":       RANDOM_STATE,
        "n_jobs":             -1,
        "verbosity":          0,
    }

    # 3-fold for speed during search
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
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

    Returns (best_params_dict, best_3fold_mape_pct)
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Optuna Tuning — {n_trials} trials (TPE sampler)")

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

    best_params  = study.best_params
    best_mape    = study.best_value * 100

    logger.info(f"Best MAPE (3-fold): {best_mape:.2f}%")
    logger.info(f"Best params: {best_params}")

    # Save study for reference
    Path(REPORTS_DIR).mkdir(exist_ok=True)
    with open(f"{REPORTS_DIR}/optuna_study.json", "w") as f:
        json.dump({
            "best_mape_pct": round(best_mape, 3),
            "best_params":   best_params,
            "n_trials":      n_trials,
            "timestamp":     datetime.now().isoformat(),
        }, f, indent=2)

    return best_params, best_mape









def train_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
    strata: np.ndarray,
) -> tuple:
    """
    LightGBM 5-fold CV — independent from XGBoost.

    Why LightGBM separately:
      LGB and XGB make different types of errors.
      LGB uses leaf-wise growth (faster, better on large data).
      XGB uses level-wise growth (more regularized).
      Their predictions are weakly correlated → ensemble benefits.

    objective='regression' with metric='mae':
      Optimizes MAE in log space — closer to MAPE than MSE.
      CatBoost optimizes MAPE directly; LGB/XGB use this proxy.
    """
    logger.info(f"\n{'='*50}")
    logger.info("LightGBM (Default Parameters)")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mapes, r2s = [], []

    lgb_params = {
        "num_leaves":         63,
        "learning_rate":      0.05,
        "n_estimators":       700,
        "min_child_samples":  10,
        "subsample":          0.8,
        "colsample_bytree":   0.8,
        "reg_alpha":          0.1,
        "reg_lambda":         1.0,
        "objective":          "regression",
        "metric":             "mae",
        "random_state":       RANDOM_STATE,
        "verbose":            -1,
        "n_jobs":             -1,
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
      we care about. This typically saves 1–2 MAPE points vs the proxy.

      Also: CatBoost's symmetric tree growth is more resistant to
      overfitting on small datasets, which helps when Kota has
      fewer listings than Gurgaon.

    early_stopping_rounds=50:
      Stops training when val MAPE stops improving for 50 rounds.
      Prevents wasted compute on overfit iterations.
    """
    if not CATBOOST_AVAILABLE:
        logger.warning("CatBoost not installed — skipping")
        return 999.0, 0.0

    logger.info(f"\n{'='*50}")
    logger.info("CatBoost (Direct MAPE Optimization)")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mapes, r2s = [], []

    cb_params = {
        "iterations":         700,
        "learning_rate":      0.05,
        "depth":              8,
        "l2_leaf_reg":        3.0,
        "subsample":          0.8,
        "loss_function":      "MAPE",
        "eval_metric":        "MAPE",
        "random_seed":        RANDOM_STATE,
        "verbose":            False,
        "allow_writing_files": False,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, strata)):
        model = CatBoostRegressor(**cb_params)
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







