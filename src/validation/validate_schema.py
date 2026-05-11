"""
src/validation/validate_schema.py
PropML — MagicBricks data validation before cleaning pipeline.

Validation levels:
  CRITICAL → pipeline stops (sys.exit(1))
  WARNING  → logged but pipeline continues
  INFO     → statistics only

Phase 3 updates:
  - 13 cities supported (was 4)
  - CITY_PPSF_BOUNDS updated for all 13 cities
  - New checks: lat/lng coords, rera_approved, amenity fill rate, nearby fill rate
  - Property type now validates 'plot' with bhk=0
  - Statistics: amenity score, nearby score, property type breakdown

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
            "min_rows_required":   10,
            "max_null_pct_price":  0.01,
            "max_null_pct_area":   0.02,
            "min_ppsf_valid_pct":  0.70,
        }
    }
    if not Path(params_path).exists():
        return defaults

    with open(params_path) as f:
        params = yaml.safe_load(f)

    merged = defaults.copy()
    merged["validation"].update(params.get("validation", {}))
    return merged


PARAMS     = _load_params()
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
    "lat", "lng", "rera_approved",
]


# ─── Numeric column expectations ─────────────────────────────────────────────
NUMERIC_COLUMNS = [
    "price", "area", "bhk", "bathroom",
    "floor_pos", "total_floors", "rating",
    "balcony", "parking",
]


# ─── All supported cities ─────────────────────────────────────────────────────
ALL_CITIES = [
    "gurgaon", "noida", "chandigarh", "kota",
    "delhi", "mumbai", "bangalore", "chennai",
    "pune", "hyderabad", "ahmedabad", "jaipur", "indore",
]


# ─── City-specific price/sqft bounds (Rs/sqft) ───────────────────────────────
# Outside these bounds = almost certainly a data error.
# Threshold: at least 70% of rows must be within bounds.
# Source: domain knowledge + MagicBricks market reports 2024-25.
CITY_PPSF_BOUNDS = {
    # NCR / North India
    "gurgaon":    (1_500,  55_000),
    "noida":      (2_000,  30_000),
    "chandigarh": (1_500,  22_000),
    "delhi":      (2_000,  60_000),
    "kota":       (800,    12_000),
    # West India
    "mumbai":     (5_000,  1_20_000),
    "pune":       (2_500,  35_000),
    "ahmedabad":  (1_500,  20_000),
    # South India
    "bangalore":  (3_000,  40_000),
    "chennai":    (2_500,  35_000),
    "hyderabad":  (2_500,  35_000),
    # Other
    "jaipur":     (1_200,  18_000),
    "indore":     (1_000,  15_000),
}


# ─── Null rate thresholds ────────────────────────────────────────────────────
NULL_THRESHOLDS = {
    "price":       (0.01, "CRITICAL"),
    "area":        (0.02, "CRITICAL"),
    "bhk":         (0.05, "WARNING"),
    "bathroom":    (0.10, "WARNING"),
    "floor_pos":   (0.15, "WARNING"),
    "furnish":     (0.20, "WARNING"),
    "locality":    (0.05, "WARNING"),
    "facing":      (0.40, "WARNING"),   # facing is frequently absent
    "rating":      (0.25, "WARNING"),
}


# ─── Valid property types ────────────────────────────────────────────────────
VALID_PROPERTY_TYPES = {"flat", "house", "independent_floor", "plot"}


# ─── Valid BHK range ─────────────────────────────────────────────────────────
BHK_MIN, BHK_MAX = 0, 10


# ─── City bounding boxes (lat/lon) ───────────────────────────────────────────
# Used to validate that scraped coordinates fall within city boundaries.
# Bounding box: (lat_min, lat_max, lon_min, lon_max)
CITY_BBOX = {
    "gurgaon":    (28.30, 28.56, 76.85, 77.15),
    "noida":      (28.45, 28.70, 77.25, 77.55),
    "chandigarh": (30.60, 30.85, 76.65, 76.90),
    "kota":       (25.05, 25.30, 75.75, 76.00),
    "delhi":      (28.40, 28.85, 76.85, 77.35),
    "mumbai":     (18.85, 19.30, 72.75, 73.05),
    "pune":       (18.40, 18.65, 73.70, 74.00),
    "ahmedabad":  (22.90, 23.20, 72.45, 72.75),
    "bangalore":  (12.80, 13.20, 77.40, 77.80),
    "chennai":    (12.85, 13.25, 80.10, 80.35),
    "hyderabad":  (17.25, 17.65, 78.25, 78.65),
    "jaipur":     (26.75, 27.05, 75.65, 75.95),
    "indore":     (22.60, 22.85, 75.75, 76.05),
}


# ─── Load data ───────────────────────────────────────────────────────────────

def load_latest_parquet(city: str, data_dir: str = "data/raw") -> Optional[pd.DataFrame]:
    """
    Load the most recently scraped parquet for this city.
    Sorts by filename (YYYYMMDD) and loads the latest.
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
        logger.info(f"Loaded {len(df)} rows x {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return None


# ─── Validator ───────────────────────────────────────────────────────────────

class SchemaValidator:
    """
    Validates one city's scraped MagicBricks data before it enters cleaning.

    Design:
    - Each check is a separate method — easy to add/remove/modify
    - _check() helper logs result and tracks pass/warn/fail counts
    - run() calls all checks in order and returns a report dict
    - Report is saved to reports/validation/ as JSON

    Phase 3 additions:
    - check_lat_lng_coords(): validates scraped coordinates
    - check_amenity_fill_rate(): warns if <10% listings have amenity data
    - check_nearby_fill_rate(): warns if <10% listings have nearby data
    - check_rera_column(): validates rera_approved is binary 0/1
    - check_plot_bhk_zero(): plots must have bhk=0
    """

    def __init__(self, df: pd.DataFrame, city: str):
        self.df       = df.copy()
        self.city     = city
        self.passed:  list[str] = []
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
            logger.info(f"  OK   {name}: {message}")
        elif level == "CRITICAL":
            self.failures.append(name)
            logger.error(f"  FAIL CRITICAL - {name}: {message}")
        else:
            self.warnings.append(name)
            logger.warning(f"  WARN WARNING  - {name}: {message}")

        return condition

    # ── Core checks ──────────────────────────────────────────────────────────

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
        prices    = pd.to_numeric(self.df["price"], errors="coerce").dropna()
        if len(prices) == 0:
            self._check("price_range", False, "No valid price values found")
            return
        pct_valid = prices.between(0.05, 100).mean()
        self._check(
            "price_range",
            pct_valid >= 0.80,
            f"{pct_valid:.1%} prices in range Rs5L-Rs100Cr (min 80%)",
            level="WARNING",
        )

    def check_area_range(self):
        """Area column: values must be in plausible sqft range."""
        if "area" not in self.df.columns:
            return
        areas     = pd.to_numeric(self.df["area"], errors="coerce").dropna()
        if len(areas) == 0:
            self._check("area_range", False, "No valid area values found")
            return
        pct_valid = areas.between(100, 50_000).mean()
        self._check(
            "area_range",
            pct_valid >= 0.80,
            f"{pct_valid:.1%} areas in range 100-50,000 sqft (min 80%)",
            level="WARNING",
        )

    def check_price_per_sqft(self):
        """
        Cross-validate price and area.

        price_per_sqft = price (Crores) * 1e7 / area (sqft)
        If ppsf is outside city-specific bounds for >30% of rows,
        either price or area column has a parsing problem.

        This is the most reliable data quality signal in real estate.
        A 500sqft flat at Rs10Cr = Rs200,000/sqft — doesn't exist in India.
        """
        if "price" not in self.df.columns or "area" not in self.df.columns:
            return

        prices = pd.to_numeric(self.df["price"], errors="coerce")
        areas  = pd.to_numeric(self.df["area"],  errors="coerce")
        valid  = prices.notna() & areas.notna() & (areas > 0)

        if valid.sum() < 5:
            return

        ppsf     = (prices[valid] * 1e7) / areas[valid]
        lo, hi   = CITY_PPSF_BOUNDS.get(self.city, (500, 1_50_000))
        in_range = ppsf.between(lo, hi).mean()

        self._check(
            "price_per_sqft",
            in_range >= 0.70,
            f"{in_range:.1%} records have Rs{lo:,}-Rs{hi:,}/sqft (min 70%)",
            level="WARNING",
        )

    def check_bhk_distribution(self):
        """BHK values must be in 0-10 range. Plots have bhk=0 which is valid."""
        if "bhk" not in self.df.columns:
            return
        bhks      = pd.to_numeric(self.df["bhk"], errors="coerce").dropna()
        pct_valid = bhks.between(BHK_MIN, BHK_MAX).mean()
        self._check(
            "bhk_range",
            pct_valid >= 0.95,
            f"{pct_valid:.1%} BHK values in 0-10 (min 95%, 0 is valid for plots)",
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
        pct       = parseable.mean()
        self._check(
            "scraped_at_parseable",
            pct >= 0.95,
            f"{pct:.1%} scraped_at values are valid timestamps",
            level="WARNING",
        )

    # ── Phase 3 checks ───────────────────────────────────────────────────────

    def check_lat_lng_coords(self):
        """
        Validate lat/lng columns (Phase 3 — added by SERVER_PRELOADED_STATE_ scraper).

        Checks:
        1. Columns exist (WARNING only — lat/lng may not be available for all runs)
        2. At least 50% of rows have non-zero coordinates
        3. Non-zero coords fall within city bounding box
        """
        if "lat" not in self.df.columns or "lng" not in self.df.columns:
            self._check(
                "lat_lng_present",
                False,
                "lat/lng columns missing — OSM enrichment unavailable",
                level="WARNING",
            )
            return

        lats = pd.to_numeric(self.df["lat"], errors="coerce").fillna(0)
        lngs = pd.to_numeric(self.df["lng"], errors="coerce").fillna(0)

        has_coords   = ((lats != 0) & (lngs != 0))
        pct_with_coords = has_coords.mean()

        self._check(
            "lat_lng_fill_rate",
            pct_with_coords >= 0.50,
            f"{pct_with_coords:.1%} rows have non-zero coordinates (min 50%)",
            level="WARNING",
        )

        if has_coords.sum() < 5:
            return

        # Bounding box check for non-zero coords
        bbox = CITY_BBOX.get(self.city)
        if bbox:
            lat_min, lat_max, lon_min, lon_max = bbox
            valid_lats = lats[has_coords].between(lat_min, lat_max)
            valid_lons = lngs[has_coords].between(lon_min, lon_max)
            in_bbox    = (valid_lats & valid_lons).mean()

            self._check(
                "lat_lng_in_city_bbox",
                in_bbox >= 0.80,
                f"{in_bbox:.1%} coordinates fall within {self.city} bounding box (min 80%)",
                level="WARNING",
            )

    def check_amenity_fill_rate(self):
        """
        Warn if very few listings have amenity data.

        Phase 3 scraper fills amenities from psmAmenDesc in SERVER_PRELOADED_STATE_.
        Low fill rate means scraper is still using DOM fallback (no amenity data).
        Target: >10% fill rate (most luxury listings have amenities).
        """
        if "amenities" not in self.df.columns:
            return

        filled   = (
            self.df["amenities"].fillna("").astype(str).str.strip() != ""
        ).mean()

        self._check(
            "amenity_fill_rate",
            filled >= 0.10,
            f"{filled:.1%} listings have amenity data (min 10%)",
            level="WARNING",
        )

    def check_nearby_fill_rate(self):
        """
        Warn if very few listings have nearby locations data.

        Phase 3 scraper fills nearbylocations from landmarkDetails.
        Low fill rate means DOM fallback is being used.
        Target: >20% fill rate (landmark data available for most project listings).
        """
        if "nearbylocations" not in self.df.columns:
            return

        filled = (
            self.df["nearbylocations"].fillna("").astype(str).str.strip() != ""
        ).mean()

        self._check(
            "nearby_fill_rate",
            filled >= 0.20,
            f"{filled:.1%} listings have nearby location data (min 20%)",
            level="WARNING",
        )

    def check_rera_column(self):
        """
        rera_approved must be binary 0/1.
        Phase 3 — scraper adds this from reraApproved field.
        """
        if "rera_approved" not in self.df.columns:
            return

        unique_vals = set(
            pd.to_numeric(self.df["rera_approved"], errors="coerce").dropna().unique()
        )
        valid_vals  = {0, 1, 0.0, 1.0}
        is_binary   = unique_vals.issubset(valid_vals)

        self._check(
            "rera_approved_binary",
            is_binary,
            f"rera_approved values: {unique_vals} (must be binary 0/1)",
            level="WARNING",
        )

    def check_plot_bhk_zero(self):
        """
        Plots must have bhk = 0.
        Phase 3 — scraper sets bhk=0 for Residential Plot listings.
        """
        if "property_type" not in self.df.columns or "bhk" not in self.df.columns:
            return

        plots        = self.df[self.df["property_type"] == "plot"]
        if len(plots) == 0:
            return

        bhk_vals     = pd.to_numeric(plots["bhk"], errors="coerce").fillna(-1)
        pct_zero_bhk = (bhk_vals == 0).mean()

        self._check(
            "plot_bhk_zero",
            pct_zero_bhk >= 0.90,
            f"{pct_zero_bhk:.1%} plots have bhk=0 ({len(plots)} plots total, min 90%)",
            level="WARNING",
        )

    # ── Statistics (INFO only — always pass) ─────────────────────────────────

    def _log_statistics(self):
        """Log summary statistics for the dataset."""
        logger.info(f"\n  Stats - {self.city.upper()}")
        logger.info(f"     Rows:  {len(self.df)}")

        if "price" in self.df.columns:
            p = pd.to_numeric(self.df["price"], errors="coerce")
            logger.info(
                f"     Price (Cr):  min={p.min():.2f}  median={p.median():.2f}  max={p.max():.2f}"
            )

        if "area" in self.df.columns:
            a = pd.to_numeric(self.df["area"], errors="coerce")
            logger.info(
                f"     Area (sqft): min={a.min():.0f}  median={a.median():.0f}  max={a.max():.0f}"
            )

        if "property_type" in self.df.columns:
            logger.info(f"     Types:  {self.df['property_type'].value_counts().to_dict()}")

        if "bhk" in self.df.columns:
            logger.info(
                f"     BHK:    {pd.to_numeric(self.df['bhk'], errors='coerce').value_counts().sort_index().to_dict()}"
            )

        # Phase 3 stats
        if "amenities" in self.df.columns:
            filled = (self.df["amenities"].fillna("").astype(str).str.strip() != "").sum()
            logger.info(f"     Amenities filled: {filled}/{len(self.df)} ({filled/len(self.df):.1%})")

        if "nearbylocations" in self.df.columns:
            filled = (self.df["nearbylocations"].fillna("").astype(str).str.strip() != "").sum()
            logger.info(f"     Nearby filled:    {filled}/{len(self.df)} ({filled/len(self.df):.1%})")

        if "lat" in self.df.columns:
            has_coords = ((pd.to_numeric(self.df["lat"], errors="coerce").fillna(0) != 0)).sum()
            logger.info(f"     Coords (lat/lng): {has_coords}/{len(self.df)} ({has_coords/len(self.df):.1%})")

        if "rera_approved" in self.df.columns:
            rera_pct = pd.to_numeric(self.df["rera_approved"], errors="coerce").fillna(0).mean()
            logger.info(f"     RERA approved:    {rera_pct:.1%}")

    # ── Main run ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Run all checks. Returns structured report dict."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Validation - {self.city.upper()} ({len(self.df)} rows)")
        logger.info("=" * 50)

        # Core checks (Phase 1)
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

        # Phase 3 checks
        self.check_lat_lng_coords()
        self.check_amenity_fill_rate()
        self.check_nearby_fill_rate()
        self.check_rera_column()
        self.check_plot_bhk_zero()

        self._log_statistics()

        overall = "PASS" if len(self.failures) == 0 else "FAIL"

        logger.info(f"\n  Result:   {overall}")
        logger.info(f"  Passed:   {len(self.passed)}")
        logger.info(f"  Warnings: {len(self.warnings)}")
        logger.info(f"  Failed:   {len(self.failures)}")

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


# ─── Report saver ────────────────────────────────────────────────────────────

def save_report(report: dict, report_dir: str = "reports/validation") -> str:
    """
    Save validation report to timestamped JSON file.

    Path: reports/validation/validation_{city}_{YYYYMMDD}.json

    Read by humans to diagnose data quality issues.
    Read by CI/CD to check if validation passed before training.
    Archived to track data quality over time.
    """
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    today    = datetime.now().strftime("%Y%m%d")
    filename = f"validation_{report['city']}_{today}.json"
    path     = report_path / filename

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Report saved -> {path}")
    return str(path)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="PropML Data Validator")
    parser.add_argument(
        "--city", default="all",
        help=(
            "City to validate, or 'all' for all 13 cities (default: all). "
            f"Available: {', '.join(ALL_CITIES)}"
        ),
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

    targets    = ALL_CITIES if args.city == "all" else [args.city]
    any_failed = False
    summary    = {}

    for city in targets:
        df = load_latest_parquet(city, args.data_dir)

        if df is None:
            logger.warning(f"Skipping {city} — no data found in {args.data_dir}/{city}/")
            summary[city] = "SKIP"
            continue

        validator = SchemaValidator(df, city)
        report    = validator.run()
        save_report(report, args.report_dir)
        summary[city] = report["summary"]["overall"]

        if report["summary"]["overall"] == "FAIL":
            any_failed = True

    # Print final summary table
    logger.info("\n" + "=" * 50)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 50)
    for city, result in summary.items():
        icon = "OK  " if result == "PASS" else ("SKIP" if result == "SKIP" else "FAIL")
        logger.info(f"  {icon}  {city}")

    total_pass = sum(1 for v in summary.values() if v == "PASS")
    total_fail = sum(1 for v in summary.values() if v == "FAIL")
    total_skip = sum(1 for v in summary.values() if v == "SKIP")
    logger.info(f"\n  PASS: {total_pass}  FAIL: {total_fail}  SKIP: {total_skip}")

    if any_failed and args.fail_on_critical:
        logger.error("One or more cities failed CRITICAL validation. Stopping pipeline.")
        sys.exit(1)

    logger.info("\nValidation complete. Next: python src/cleaning/cleaning_pipeline.py")


if __name__ == "__main__":
    main()