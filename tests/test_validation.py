"""
tests/test_validation.py
Unit tests for the SchemaValidator.

Tests verify:
1. A clean DataFrame passes all checks
2. Specific broken DataFrames trigger the correct check failures
3. CRITICAL failures are distinguished from WARNINGS

Run: pytest tests/test_validation.py -v
"""

import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, ".")
from src.validation.validate_schema import SchemaValidator


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_clean_df(n: int = 50, city: str = "gurgaon") -> pd.DataFrame:
    """
    Make a minimal valid DataFrame that should pass all checks.
    Adjust values to match Gurgaon price/sqft bounds.
    """
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "city":          [city] * n,
        "property_type": ["flat"] * n,
        "price":         rng.uniform(0.5, 3.0, n).round(3),      # Crores
        "area":          rng.uniform(800, 2500, n).round(1),      # sqft
        "bhk":           rng.integers(1, 5, n),
        "bathroom":      rng.integers(1, 4, n),
        "balcony":       rng.integers(0, 3, n),
        "floor_pos":     rng.integers(0, 15, n),
        "total_floors":  rng.integers(5, 20, n),
        "age":           ["1-5 years"] * n,
        "furnish":       ["semi-furnished"] * n,
        "locality":      ["Sector 62"] * n,
        "sector":        ["DLF Phase 1"] * n,
        "amenities":     ["gym,lift"] * n,
        "nearbylocations": ["metro,hospital"] * n,
        "parking":       rng.integers(0, 3, n),
        "facing":        ["North"] * n,
        "rating":        rng.uniform(3.5, 4.9, n).round(1),
        "is_near_coaching": [0] * n,
        "property_id":   [f"prop_{i}" for i in range(n)],
        "source":        ["magicbricks"] * n,
        "scraped_at":    ["2025-05-04T10:00:00"] * n,
    })


# ─── Tests: clean data passes ────────────────────────────────────────────────

class TestCleanDataPasses:
    def test_all_checks_pass_on_clean_data(self):
        df = make_clean_df()
        v = SchemaValidator(df, "gurgaon")
        r = v.run()
        assert r["summary"]["overall"] == "PASS"
        assert r["summary"]["failures"] == 0

    def test_noida_passes_with_correct_city(self):
        df = make_clean_df(city="noida")
        # Adjust price for Noida bounds
        df["price"] = 0.8
        v = SchemaValidator(df, "noida")
        r = v.run()
        assert r["summary"]["failures"] == 0


# ─── Tests: CRITICAL failures ────────────────────────────────────────────────

class TestCriticalFailures:
    def test_empty_dataframe_fails(self):
        df = make_clean_df(n=5)  # Below min_rows_required=10
        v = SchemaValidator(df, "gurgaon")
        r = v.run()
        assert r["summary"]["overall"] == "FAIL"
        failed_names = [c["name"] for c in r["checks"] if c["status"] == "CRITICAL"]
        assert "row_count" in failed_names

    def test_missing_price_column_fails(self):
        df = make_clean_df().drop(columns=["price"])
        v = SchemaValidator(df, "gurgaon")
        r = v.run()
        assert r["summary"]["overall"] == "FAIL"
        failed_names = [c["name"] for c in r["checks"] if c["status"] == "CRITICAL"]
        assert "required_columns" in failed_names

    def test_missing_area_column_fails(self):
        df = make_clean_df().drop(columns=["area"])
        v = SchemaValidator(df, "gurgaon")
        r = v.run()
        assert r["summary"]["overall"] == "FAIL"

    def test_all_prices_null_fails(self):
        df = make_clean_df()
        df["price"] = np.nan
        v = SchemaValidator(df, "gurgaon")
        r = v.run()
        assert r["summary"]["overall"] == "FAIL"

    def test_missing_required_columns_fails(self):
        df = make_clean_df().drop(columns=["city", "bhk"])
        v = SchemaValidator(df, "gurgaon")
        r = v.run()
        assert r["summary"]["overall"] == "FAIL"


# ─── Tests: WARNING conditions ───────────────────────────────────────────────

class TestWarnings:
    def test_wrong_city_in_column_is_warning(self):
        df = make_clean_df()
        df["city"] = "noida"  # Wrong city value in column
        v = SchemaValidator(df, "gurgaon")
        r = v.run()
        warn_names = [c["name"] for c in r["checks"] if c["status"] == "WARNING"]
        assert "city_consistency" in warn_names

    def test_high_facing_nulls_is_warning_not_critical(self):
        df = make_clean_df()
        df.loc[:40, "facing"] = np.nan  # 80% null — above 40% threshold
        v = SchemaValidator(df, "gurgaon")
        r = v.run()
        warn_names = [c["name"] for c in r["checks"] if c["status"] == "WARNING"]
        crit_names = [c["name"] for c in r["checks"] if c["status"] == "CRITICAL"]
        assert "null_rate_facing" in warn_names
        assert "null_rate_facing" not in crit_names

    def test_unknown_property_type_is_warning(self):
        df = make_clean_df()
        df.loc[0, "property_type"] = "villa"  # Not in VALID_PROPERTY_TYPES
        v = SchemaValidator(df, "gurgaon")
        r = v.run()
        warn_names = [c["name"] for c in r["checks"] if c["status"] == "WARNING"]
        assert "property_type_values" in warn_names

    def test_out_of_range_bhk_is_warning(self):
        df = make_clean_df()
        df.loc[:5, "bhk"] = 25  # Clearly invalid
        v = SchemaValidator(df, "gurgaon")
        r = v.run()
        warn_names = [c["name"] for c in r["checks"] if c["status"] == "WARNING"]
        assert "bhk_range" in warn_names

    def test_duplicate_property_ids_is_warning(self):
        df = make_clean_df()
        df.loc[1, "property_id"] = df.loc[0, "property_id"]  # Create duplicate
        v = SchemaValidator(df, "gurgaon")
        r = v.run()
        warn_names = [c["name"] for c in r["checks"] if c["status"] == "WARNING"]
        assert "duplicates" in warn_names


# ─── Tests: price/sqft cross-validation ─────────────────────────────────────

class TestPricePerSqft:
    def test_valid_ppsf_passes(self):
        df = make_clean_df()
        # price=1.5Cr, area=1500sqft → ppsf=10,000 — within Gurgaon bounds
        df["price"] = 1.5
        df["area"]  = 1500.0
        v = SchemaValidator(df, "gurgaon")
        r = v.run()
        ppsf_check = next((c for c in r["checks"] if c["name"] == "price_per_sqft"), None)
        assert ppsf_check is not None
        assert ppsf_check["status"] == "PASS"

    def test_impossible_ppsf_triggers_warning(self):
        df = make_clean_df()
        # price=0.01Cr, area=5000sqft → ppsf=20 — impossibly cheap
        df["price"] = 0.01
        df["area"]  = 5000.0
        v = SchemaValidator(df, "gurgaon")
        r = v.run()
        ppsf_check = next((c for c in r["checks"] if c["name"] == "price_per_sqft"), None)
        assert ppsf_check["status"] in ("WARNING", "CRITICAL") # type: ignore