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


def _load_params(params_path: str = "params.yaml") -> dict:
    """Load validation params from params.yaml. Returns defaults if file missing."""
    defaults = {
        "validation": {
            "min_rows_required": 10,
            "max_null_pct_price": 0.01,
            "max_null_pct_area": 0.02,
            "min_ppsf_valid_pct": 0.70,
        }
    }
    if not Path(params_path).exists():
        return defaults

    with open(params_path) as f:
        params = yaml.safe_load(f)

    # Merge with defaults so missing keys don't crash
    merged = defaults.copy()
    merged["validation"].update(params.get("validation", {}))
    return merged


PARAMS = _load_params()
VAL_PARAMS = PARAMS["validation"]


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

    # ── Checks ───────────────────────────────────────────────────────────────

    def check_not_empty(self):
        n = len(self.df)
        self._check(
            "row_count",
            n >= VAL_PARAMS["min_rows_required"],
            f"{n} rows loaded (minimum {VAL_PARAMS['min_rows_required']} required)",
        )

    def check_required_columns(self):
        missing = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        self._check(
            "required_columns",
            len(missing) == 0,
            f"Missing: {missing}" if missing else "All required columns present",
        )

    def check_numeric_parseable(self):
        """Verify numeric columns can be parsed as numbers."""
        for col in NUMERIC_COLUMNS:
            if col not in self.df.columns:
                continue
            parseable = pd.to_numeric(self.df[col], errors="coerce").notna()
            pct = parseable.mean()
            self._check(
                f"numeric_{col}",
                pct >= 0.80,
                f"'{col}': {pct:.1%} parseable as numeric (min 80%)",
                level="WARNING",
            )

    def check_null_rates(self):
        thresholds = {
            **NULL_THRESHOLDS,
            "price": (VAL_PARAMS["max_null_pct_price"], "CRITICAL"),
            "area":  (VAL_PARAMS["max_null_pct_area"],  "CRITICAL"),
        }
        for col, (threshold, level) in thresholds.items():
            if col not in self.df.columns:
                continue
            null_rate = self.df[col].isna().mean()
            self._check(
                f"null_rate_{col}",
                null_rate <= threshold,
                f"'{col}' null rate: {null_rate:.1%} (max: {threshold:.0%})",
                level=level,
            )

    def check_price_range(self):
        """Price column: values must be in plausible Crore range."""
        if "price" not in self.df.columns:
            return
        prices = pd.to_numeric(self.df["price"], errors="coerce").dropna()
        if len(prices) == 0:
            self._check("price_range", False, "No valid price values found")
            return
        pct_valid = prices.between(0.05, 100).mean()
        self._check(
            "price_range",
            pct_valid >= 0.80,
            f"{pct_valid:.1%} prices in range ₹5L–₹100Cr (min 80%)",
            level="WARNING",
        )

    def check_area_range(self):
        """Area column: values must be in plausible sqft range."""
        if "area" not in self.df.columns:
            return
        areas = pd.to_numeric(self.df["area"], errors="coerce").dropna()
        if len(areas) == 0:
            self._check("area_range", False, "No valid area values found")
            return
        pct_valid = areas.between(100, 50_000).mean()
        self._check(
            "area_range",
            pct_valid >= 0.80,
            f"{pct_valid:.1%} areas in range 100–50,000 sqft (min 80%)",
            level="WARNING",
        )

    def check_price_per_sqft(self):
        """
        Cross-validate price and area.

        price_per_sqft = price (Crores) * 1e7 / area (sqft)
        If ppsf is outside city-specific bounds for >30% of rows,
        either price or area column has a parsing problem.

        This is the most reliable data quality signal in real estate.
        A 500sqft flat at ₹10Cr = ₹200,000/sqft — doesn't exist in India.
        """
        if "price" not in self.df.columns or "area" not in self.df.columns:
            return

        prices = pd.to_numeric(self.df["price"], errors="coerce")
        areas  = pd.to_numeric(self.df["area"],  errors="coerce")
        valid  = prices.notna() & areas.notna() & (areas > 0)

        if valid.sum() < 5:
            return

        ppsf = (prices[valid] * 1e7) / areas[valid]
        lo, hi = CITY_PPSF_BOUNDS.get(self.city, (500, 80_000))
        in_range = ppsf.between(lo, hi).mean()

        self._check(
            "price_per_sqft",
            in_range >= 0.70,
            f"{in_range:.1%} records have ₹{lo:,}–₹{hi:,}/sqft (min 70%)",
            level="WARNING",
        )

    def check_bhk_distribution(self):
        """BHK values must be in 0–10 range."""
        if "bhk" not in self.df.columns:
            return
        bhks = pd.to_numeric(self.df["bhk"], errors="coerce").dropna()
        pct_valid = bhks.between(BHK_MIN, BHK_MAX).mean()
        self._check(
            "bhk_range",
            pct_valid >= 0.95,
            f"{pct_valid:.1%} BHK values in 0–10 (min 95%)",
            level="WARNING",
        )

    def check_city_column(self):
        """All rows must have the correct city value."""
        if "city" not in self.df.columns:
            return
        wrong = (self.df["city"] != self.city).sum()
        self._check(
            "city_consistency",
            wrong == 0,
            f"{wrong} rows have wrong city value (expected '{self.city}')",
            level="WARNING",
        )

    def check_source_column(self):
        """All rows must have source = magicbricks."""
        if "source" not in self.df.columns:
            return
        not_mb = (self.df["source"] != "magicbricks").sum()
        self._check(
            "source_magicbricks",
            not_mb == 0,
            f"{not_mb} rows have source != 'magicbricks'",
            level="WARNING",
        )

    def check_property_types(self):
        """property_type must be one of the 4 valid values."""
        if "property_type" not in self.df.columns:
            return
        unknown = set(self.df["property_type"].unique()) - VALID_PROPERTY_TYPES
        self._check(
            "property_type_values",
            len(unknown) == 0,
            f"Unknown types: {unknown}" if unknown else "All property_types valid",
            level="WARNING",
        )

    def check_duplicates(self):
        """Check for duplicate rows."""
        if "property_id" in self.df.columns:
            dups = self.df["property_id"].duplicated().sum()
        else:
            # Fall back to composite key deduplication
            dups = self.df.duplicated(
                subset=["price", "area", "bhk", "locality"]
            ).sum()
        self._check(
            "duplicates",
            dups == 0,
            f"{dups} duplicate rows found",
            level="WARNING",
        )

    def check_scraped_at(self):
        """scraped_at must be present and parseable."""
        if "scraped_at" not in self.df.columns:
            return
        parseable = pd.to_datetime(self.df["scraped_at"], errors="coerce").notna()
        pct = parseable.mean()
        self._check(
            "scraped_at_parseable",
            pct >= 0.95,
            f"{pct:.1%} scraped_at values are valid timestamps",
            level="WARNING",
        )

    # ── Statistics (INFO only — always pass) ─────────────────────────────────

    def _log_statistics(self):
        """Log summary statistics for the dataset."""
        logger.info(f"\n  📊 Statistics — {self.city.upper()}")
        logger.info(f"     Rows:            {len(self.df)}")

        if "price" in self.df.columns:
            p = pd.to_numeric(self.df["price"], errors="coerce")
            logger.info(
                f"     Price (Cr):      min={p.min():.2f}  median={p.median():.2f}  max={p.max():.2f}"
            )

        if "area" in self.df.columns:
            a = pd.to_numeric(self.df["area"], errors="coerce")
            logger.info(
                f"     Area (sqft):     min={a.min():.0f}  median={a.median():.0f}  max={a.max():.0f}"
            )

        if "property_type" in self.df.columns:
            logger.info(f"     Types:  {self.df['property_type'].value_counts().to_dict()}")

        if "bhk" in self.df.columns:
            logger.info(f"     BHK:    {pd.to_numeric(self.df['bhk'], errors='coerce').value_counts().sort_index().to_dict()}")

    # ── Main run ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Run all checks. Returns structured report dict."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Validation — {self.city.upper()} ({len(self.df)} rows)")
        logger.info("=" * 50)

        self.check_not_empty()
        self.check_required_columns()
        self.check_numeric_parseable()
        self.check_null_rates()
        self.check_price_range()
        self.check_area_range()
        self.check_price_per_sqft()
        self.check_bhk_distribution()
        self.check_city_column()
        self.check_source_column()
        self.check_property_types()
        self.check_duplicates()
        self.check_scraped_at()
        self._log_statistics()

        overall = "PASS" if len(self.failures) == 0 else "FAIL"

        logger.info(f"\n  Result: {overall}")
        logger.info(f"  ✅ Passed:   {len(self.passed)}")
        logger.info(f"  ⚠  Warnings: {len(self.warnings)}")
        logger.info(f"  ❌ Failed:   {len(self.failures)}")

        return {
            "city":         self.city,
            "n_rows":       len(self.df),
            "n_cols":       len(self.df.columns),
            "validated_at": datetime.now().isoformat(),
            "checks":       self.check_log,
            "summary": {
                "passed":   len(self.passed),
                "warnings": len(self.warnings),
                "failures": len(self.failures),
                "overall":  overall,
            },
        }
    

def save_report(report: dict, report_dir: str = "reports/validation") -> str:
    """
    Save validation report to timestamped JSON file.

    Path: reports/validation/validation_{city}_{YYYYMMDD}.json

    The report is:
    - Read by humans to diagnose data quality issues
    - Read by CI/CD to check if validation passed before training
    - Archived to track data quality over time
    """
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y%m%d")
    filename = f"validation_{report['city']}_{today}.json"
    path = report_path / filename

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Report saved → {path}")
    return str(path)



def main():
    import argparse

    parser = argparse.ArgumentParser(description="PropML Data Validator")
    parser.add_argument(
        "--city", default="all",
        help="City to validate, or 'all' for all 4 cities (default: all)",
    )
    parser.add_argument(
        "--data-dir", default="data/raw",
        help="Directory containing city parquet files (default: data/raw)",
    )
    parser.add_argument(
        "--report-dir", default="reports/validation",
        help="Directory to save validation reports (default: reports/validation)",
    )
    parser.add_argument(
        "--fail-on-critical", action="store_true", default=True,
        help="Exit with code 1 if any CRITICAL check fails (default: True)",
    )
    args = parser.parse_args()

    all_cities = ["gurgaon", "noida", "chandigarh", "kota"]
    targets = all_cities if args.city == "all" else [args.city]

    any_failed = False

    for city in targets:
        df = load_latest_parquet(city, args.data_dir)

        if df is None:
            logger.warning(f"Skipping {city} — no data found in {args.data_dir}/{city}/")
            continue

        validator = SchemaValidator(df, city)
        report    = validator.run()
        save_report(report, args.report_dir)

        if report["summary"]["overall"] == "FAIL":
            any_failed = True

    if any_failed and args.fail_on_critical:
        logger.error("One or more cities failed CRITICAL validation. Stopping pipeline.")
        sys.exit(1)

    logger.info("\n✅ Validation complete. Next: python src/cleaning/cleaning_pipeline.py")


if __name__ == "__main__":
    main()