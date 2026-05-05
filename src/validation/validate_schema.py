"""
src/validation/validate_schema.py
PropML — MagicBricks data validation before cleaning pipeline.

Validation levels:
  CRITICAL → pipeline stops (sys.exit(1))
  WARNING  → logged but pipeline continues
  INFO     → statistics only

Run:
  python src/validation/validate_schema.py --city gurgaon
  python src/validation/validate_schema.py --city all

DVC stage (dvc.yaml):
  validate:
    cmd: python src/validation/validate_schema.py --city all --fail-on-critical
"""

import json
import logging
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


# ─── Required columns ────────────────────────────────────────────────────────
# Every row in every city parquet must have these.
# Missing = CRITICAL failure → pipeline stops.
REQUIRED_COLUMNS = [
    "city", "property_type", "price", "area", "bhk",
    "bathroom", "floor_pos", "total_floors", "age",
    "furnish", "locality", "sector", "amenities",
    "scraped_at", "source",
]

# Optional — present on most rows but allowed to be missing
OPTIONAL_COLUMNS = [
    "balcony", "parking", "facing", "rating",
    "nearbylocations", "property_id", "is_near_coaching",
]


# ─── Numeric column expectations ─────────────────────────────────────────────
NUMERIC_COLUMNS = [
    "price", "area", "bhk", "bathroom",
    "floor_pos", "total_floors", "rating",
    "balcony", "parking",
]


# ─── City-specific price/sqft bounds (₹/sqft) ────────────────────────────────
# Outside these bounds = almost certainly a data error.
# Threshold: at least 70% of rows must be within bounds.
# Source: domain knowledge + MagicBricks market reports 2024.
CITY_PPSF_BOUNDS = {
    "gurgaon":    (1_500,  45_000),
    "noida":      (2_000,  28_000),
    "chandigarh": (1_500,  22_000),
    "kota":       (800,    12_000),
}


# ─── Null rate thresholds ────────────────────────────────────────────────────
# Columns where null rate above threshold triggers WARNING or CRITICAL.
NULL_THRESHOLDS = {
    "price":       (0.01, "CRITICAL"),   # price is the target — near-zero nulls
    "area":        (0.02, "CRITICAL"),   # area is primary feature — near-zero nulls
    "bhk":         (0.05, "WARNING"),
    "bathroom":    (0.10, "WARNING"),
    "floor_pos":   (0.15, "WARNING"),    # floor info often missing online
    "furnish":     (0.20, "WARNING"),
    "locality":    (0.05, "WARNING"),
    "facing":      (0.40, "WARNING"),    # facing is frequently absent
    "rating":      (0.25, "WARNING"),
}


# ─── Valid property types ────────────────────────────────────────────────────
VALID_PROPERTY_TYPES = {"flat", "house", "independent_floor", "plot"}


# ─── Valid BHK range ────────────────────────────────────────────────────────
BHK_MIN, BHK_MAX = 0, 10














def load_latest_parquet(city: str, data_dir: str = "data/raw") -> Optional[pd.DataFrame]:
    """
    Load the most recently scraped parquet for this city.

    Sorts by filename (which contains YYYYMMDD) and loads the latest.
    Returns None if no parquet found.
    """
    city_dir = Path(data_dir) / city
    parquets = sorted(city_dir.glob("magicbricks_*.parquet"), reverse=True)

    if not parquets:
        logger.error(f"No parquet files found in {city_dir}")
        return None

    path = parquets[0]
    logger.info(f"Loading: {path} ({path.stat().st_size // 1024} KB)")

    try:
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df)} rows × {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return None
    












class SchemaValidator:
    """
    Validates one city's scraped MagicBricks data before it enters cleaning.

    Design:
    - Each check is a separate method → easy to add/remove/modify
    - check() helper logs result and tracks pass/warn/fail counts
    - run() calls all checks in order and returns a report dict
    - Report is saved to reports/validation/ as JSON
    """

    def __init__(self, df: pd.DataFrame, city: str):
        self.df = df.copy()
        self.city = city
        self.passed: list[str] = []
        self.warnings: list[str] = []
        self.failures: list[str] = []
        self.check_log: list[dict] = []

    def _check(
        self, name: str, condition: bool, message: str, level: str = "CRITICAL"
    ) -> bool:
        """Log a single check result."""
        status = "PASS" if condition else level
        self.check_log.append({"name": name, "status": status, "message": message})

        if condition:
            self.passed.append(name)
            logger.info(f"  ✅  {name}: {message}")
        elif level == "CRITICAL":
            self.failures.append(name)
            logger.error(f"  ❌  CRITICAL — {name}: {message}")
        else:
            self.warnings.append(name)
            logger.warning(f"  ⚠   WARNING — {name}: {message}")

        return condition

    def run(self) -> dict:
        """Run all checks. Returns structured report dict."""
        # Checks added Day 2 and Day 3
        raise NotImplementedError("Checks not yet implemented — add in Day 2/3")