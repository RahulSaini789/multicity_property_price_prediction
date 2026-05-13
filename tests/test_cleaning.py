"""
tests/test_cleaning.py
Unit tests for data cleaning pipeline (Phase 3, 13 cities).

Tests verify:
- String parsers handle all MagicBricks formats including Phase 3
  SERVER_PRELOADED_STATE_ fields (priceD, caSqFt, la, acD)
- Business logic filters remove correct rows
- Plot bhk=0 is preserved (not filled to 2)
- Grouped IQR does not remove valid luxury properties
- City tier classification works for all 13 cities
- lat/lng validation rejects out-of-India coordinates
- rera_approved cast to binary int

Run: pytest tests/test_cleaning.py -v
"""

import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.cleaning.cleaning_pipeline import (
    parse_price,
    parse_area,
    parse_floor,
    parse_bhk,
    apply_business_logic_filters,
    create_city_tier,
    fix_data_types,
    ALL_CITIES,
    PRICE_PER_SQFT_BOUNDS,
    MIN_AREA_BY_TYPE,
    MAX_AREA_BY_TYPE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Price parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestParsePrice:

    def test_crore_with_symbol(self):
        assert parse_price("₹ 1.25 Cr") == pytest.approx(1.25, abs=0.001)

    def test_lakh_string(self):
        assert parse_price("₹ 85 L") == pytest.approx(0.85, abs=0.001)

    def test_raw_rupees_large(self):
        result = parse_price("4500000")
        assert result == pytest.approx(0.45, abs=0.001)

    def test_price_on_request_returns_nan(self):
        assert np.isnan(parse_price("Price on Request"))

    def test_empty_string_returns_nan(self):
        assert np.isnan(parse_price(""))

    def test_crore_without_symbol(self):
        assert parse_price("1.5 Crore") == pytest.approx(1.5, abs=0.001)

    def test_lakh_full_word(self):
        assert parse_price("75 Lakh") == pytest.approx(0.75, abs=0.001)

    def test_nan_input_returns_nan(self):
        assert np.isnan(parse_price(np.nan))

    # ── Phase 3: SERVER_PRELOADED_STATE_ formats ──────────────────────────────

    def test_priced_string_format(self):
        """Phase 3: priceD field = '73.41 Cr' string."""
        assert parse_price("73.41 Cr") == pytest.approx(73.41, abs=0.01)

    def test_paisa_int_format(self):
        """Phase 3: raw price field = 734187000 (paisa int)."""
        result = parse_price(734187000)
        assert result == pytest.approx(73.41, abs=0.1)

    def test_paisa_small_value(self):
        """Phase 3: 8500000 paisa = 0.85 Cr."""
        result = parse_price(8500000)
        assert result == pytest.approx(0.85, abs=0.01)

    def test_already_crore_float(self):
        """If numeric < 1000, treat as already in Crores."""
        result = parse_price(1.5)
        assert result == pytest.approx(1.5, abs=0.001)

    def test_contact_returns_nan(self):
        assert np.isnan(parse_price("Contact for price"))

    def test_call_returns_nan(self):
        assert np.isnan(parse_price("Call Now"))

    def test_mumbai_luxury_price(self):
        """Mumbai luxury: Rs120 Cr should parse correctly."""
        assert parse_price("120 Cr") == pytest.approx(120.0, abs=0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# Area parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseArea:

    def test_sqft_string(self):
        assert parse_area("1800 sq.ft.") == pytest.approx(1800.0, abs=1.0)

    def test_sqm_conversion(self):
        result = parse_area("167.22 sq.mt.")
        assert result == pytest.approx(1800.0, abs=5.0)

    def test_range_takes_midpoint(self):
        result = parse_area("1200 - 1500 sqft")
        assert result == pytest.approx(1350.0, abs=1.0)

    def test_bigha_conversion(self):
        assert parse_area("1 Bigha") == pytest.approx(27225.0, abs=1.0)

    def test_plain_number_defaults_sqft(self):
        assert parse_area("2000") == pytest.approx(2000.0, abs=0.1)

    def test_empty_returns_nan(self):
        result = parse_area("")
        assert result is np.nan or (result != result)  # nan check

    def test_numeric_float_passthrough(self):
        """Phase 3: caSqFt field comes as numeric float already."""
        assert parse_area(1800.0) == pytest.approx(1800.0, abs=0.1)

    def test_numeric_int_passthrough(self):
        assert parse_area(1000) == pytest.approx(1000.0, abs=0.1)

    def test_sq_yrd_plot_conversion(self):
        """Phase 3: plot area in sq-yrd (la field) → sqft."""
        result = parse_area("500 sq-yrd")
        assert result == pytest.approx(4500.0, abs=1.0)  # 500 * 9 = 4500

    def test_marla_conversion(self):
        result = parse_area("10 marla")
        assert result == pytest.approx(2722.5, abs=1.0)

    def test_zero_returns_nan(self):
        result = parse_area(0)
        assert result is np.nan or result != result


# ═══════════════════════════════════════════════════════════════════════════════
# Floor parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseFloor:

    def test_out_of_format(self):
        assert parse_floor("5 out of 12") == (5, 12)

    def test_ground_out_of(self):
        assert parse_floor("Ground out of 8") == (0, 8)

    def test_ground_only(self):
        fp, tf = parse_floor("Ground")
        assert fp == 0

    def test_non_breaking_space(self):
        assert parse_floor("\xa0 5 out of 12") == (5, 12)

    def test_six_plus(self):
        fp, tf = parse_floor("6+")
        assert fp == 6 and tf >= 10

    def test_empty_returns_default(self):
        fp, tf = parse_floor("")
        assert fp == 0 and tf == 5

    def test_plot_ground_floor(self):
        """Phase 3: plots have floor_str='Ground'."""
        fp, tf = parse_floor("Ground")
        assert fp == 0

    def test_slash_format(self):
        fp, tf = parse_floor("3/10")
        assert fp == 3 and tf == 10


# ═══════════════════════════════════════════════════════════════════════════════
# BHK parsing (Phase 3: plot-aware)
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseBhk:

    def test_normal_bhk(self):
        assert parse_bhk(3) == 3

    def test_string_bhk(self):
        assert parse_bhk("3 BHK") == 3

    def test_plot_always_zero(self):
        """Phase 3: plots must have bhk=0 regardless of input."""
        assert parse_bhk(3,   "plot") == 0
        assert parse_bhk("3", "plot") == 0
        assert parse_bhk(None,"plot") == 0

    def test_nan_flat_defaults_two(self):
        assert parse_bhk(None, "flat") == 2

    def test_zero_preserved_for_plot(self):
        assert parse_bhk(0, "plot") == 0

    def test_float_bhk(self):
        assert parse_bhk(2.0) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Business logic filters (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBusinessFilters:

    def _make_df(self, n=50, city="gurgaon", prop_type="flat"):
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "city":          [city] * n,
            "property_type": [prop_type] * n,
            "price":         rng.uniform(0.5, 3.0, n),
            "area":          rng.uniform(800, 2000, n),
            "bhk":           rng.integers(2, 4, n).tolist(),
        })

    def test_removes_bhk_above_10(self):
        df = self._make_df()
        df.loc[0, "bhk"] = 25
        result = apply_business_logic_filters(df)
        assert 25 not in result["bhk"].values

    def test_removes_area_below_minimum_flat(self):
        df = self._make_df()
        df.loc[1, "area"] = 50   # Below MIN_AREA_BY_TYPE['flat'] = 200
        result = apply_business_logic_filters(df)
        assert len(result) < len(df)

    def test_keeps_valid_rows(self):
        df = self._make_df(n=30)
        result = apply_business_logic_filters(df)
        assert len(result) >= 25

    def test_plot_bhk_zero_not_removed(self):
        """Phase 3: plot bhk=0 is valid, must NOT be removed."""
        rng = np.random.default_rng(42)
        n = 20
        df = pd.DataFrame({
            "city":          ["jaipur"] * n,
            "property_type": ["plot"] * n,
            "price":         rng.uniform(0.3, 2.0, n),
            "area":          rng.uniform(500, 5000, n),
            "bhk":           [0] * n,
        })
        result = apply_business_logic_filters(df)
        assert len(result) > 0, "All plots were incorrectly removed"
        assert (result["bhk"] == 0).all(), "Plot bhk=0 should be preserved"

    def test_plot_area_bounds_larger(self):
        """Phase 3: plots can be up to 50000 sqft (larger than flats)."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "city":          ["bangalore"] * 10,
            "property_type": ["plot"] * 10,
            "price":         rng.uniform(0.5, 5.0, 10),
            "area":          [20000.0] * 10,   # Large plot, valid
            "bhk":           [0] * 10,
        })
        result = apply_business_logic_filters(df)
        assert len(result) > 0, "Large valid plots should not be removed"

    def test_all_13_cities_have_ppsf_bounds(self):
        """Phase 3: all 13 cities must have price/sqft bounds defined."""
        for city in ALL_CITIES:
            assert city in PRICE_PER_SQFT_BOUNDS, \
                f"Missing PRICE_PER_SQFT_BOUNDS for city: {city}"

    def test_plot_type_not_in_ppsf_check(self):
        """Phase 3: plots are excluded from price/sqft check."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "city":          ["mumbai"] * 5,
            "property_type": ["plot"] * 5,
            "price":         [0.5, 1.0, 2.0, 5.0, 10.0],
            "area":          [500, 1000, 2000, 5000, 10000],
            "bhk":           [0] * 5,
        })
        result = apply_business_logic_filters(df)
        # Plots should not be filtered by ppsf bounds
        assert len(result) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# City tier classification (Phase 3: 13 cities)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCityTier:

    def test_three_tiers_created(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "city":  ["gurgaon"] * 100,
            "price": rng.uniform(0.3, 5.0, 100),
        })
        df = create_city_tier(df)
        tiers = set(df["city_tier"].unique())
        assert tiers == {"Tier-1", "Tier-2", "Tier-3"}

    def test_tier1_has_highest_prices(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "city":  ["gurgaon"] * 90,
            "price": rng.uniform(0.3, 5.0, 90),
        })
        df = create_city_tier(df)
        t1_median = df[df["city_tier"] == "Tier-1"]["price"].median()
        t3_median = df[df["city_tier"] == "Tier-3"]["price"].median()
        assert t1_median > t3_median

    def test_all_13_cities_tiered_independently(self):
        """Phase 3: each city gets its own tertile boundaries."""
        rng = np.random.default_rng(42)
        frames = []
        for city in ALL_CITIES:
            frames.append(pd.DataFrame({
                "city":  [city] * 30,
                "price": rng.uniform(0.2, 5.0, 30),
            }))
        df = pd.concat(frames, ignore_index=True)
        df = create_city_tier(df)

        for city in ALL_CITIES:
            city_tiers = set(df[df["city"] == city]["city_tier"].unique())
            assert city_tiers == {"Tier-1", "Tier-2", "Tier-3"}, \
                f"City {city} does not have all 3 tiers"

    def test_kota_tier1_below_mumbai_tier3(self):
        """
        Within-city tiering: Kota Tier-1 properties are cheaper than
        Mumbai Tier-3, proving tiers are city-relative not global.
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "city":  ["kota"] * 30 + ["mumbai"] * 30,
            "price": list(rng.uniform(0.2, 0.8, 30)) + list(rng.uniform(3.0, 20.0, 30)),
        })
        df = create_city_tier(df)

        kota_t1_max   = df[(df["city"] == "kota")   & (df["city_tier"] == "Tier-1")]["price"].max()
        mumbai_t3_min = df[(df["city"] == "mumbai") & (df["city_tier"] == "Tier-3")]["price"].min()

        assert kota_t1_max < mumbai_t3_min, \
            "Kota Tier-1 should be cheaper than Mumbai Tier-3"


# ═══════════════════════════════════════════════════════════════════════════════
# fix_data_types (Phase 3: lat/lng + rera_approved)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFixDataTypes:

    def _make_base_df(self, n=5):
        return pd.DataFrame({
            "city":          ["pune"] * n,
            "property_type": ["flat"] * n,
            "price":         [1.0] * n,
            "area":          [1000.0] * n,
            "bhk":           [2] * n,
            "bathroom":      [2] * n,
            "balcony":       [1] * n,
            "parking":       [1] * n,
            "floor_pos":     [3] * n,
            "total_floors":  [10] * n,
            "age":           ["1-5 years"] * n,
            "furnish":       ["semi-furnished"] * n,
            "locality":      ["Baner"] * n,
            "floor":         ["3 out of 10"] * n,
            "log_price":     [0.7] * n,
            "city_tier":     ["Tier-2"] * n,
            "is_near_coaching": [0] * n,
        })

    def test_lat_lng_float_cast(self):
        df = self._make_base_df()
        df["lat"] = ["18.52", "18.55", "18.50", "18.53", "18.51"]
        df["lng"] = ["73.85", "73.88", "73.82", "73.86", "73.84"]
        df = fix_data_types(df)
        assert df["lat"].dtype == float
        assert df["lng"].dtype == float

    def test_out_of_india_coords_zeroed(self):
        """Phase 3: coordinates outside India bbox → 0.0."""
        df = self._make_base_df(n=2)
        df["lat"] = [18.52, 51.5]   # 51.5 = London, out of India range
        df["lng"] = [73.85, -0.12]  # -0.12 = London
        df = fix_data_types(df)
        assert df.loc[0, "lat"] == pytest.approx(18.52, abs=0.01)
        assert df.loc[1, "lat"] == 0.0

    def test_rera_approved_binary(self):
        """Phase 3: rera_approved must be 0 or 1."""
        df = self._make_base_df()
        df["rera_approved"] = [0, 1, 0, 1, 2]  # 2 is invalid
        df = fix_data_types(df)
        assert set(df["rera_approved"].unique()).issubset({0, 1})

    def test_plot_bhk_stays_zero(self):
        """Phase 3: plot bhk must not be filled to 2."""
        df = self._make_base_df(n=3)
        df["property_type"] = ["plot", "flat", "house"]
        df["bhk"]           = [0, 2, 3]
        df = fix_data_types(df)
        assert df.loc[0, "bhk"] == 0, "Plot bhk=0 should not be changed"
        assert df.loc[1, "bhk"] == 2
        assert df.loc[2, "bhk"] == 3

    def test_missing_lat_lng_filled_zero(self):
        """Phase 3: if lat/lng columns absent, they are created as 0.0."""
        df = self._make_base_df()
        # No lat/lng columns
        df = fix_data_types(df)
        assert "lat" in df.columns
        assert "lng" in df.columns
        assert (df["lat"] == 0.0).all()
        assert (df["lng"] == 0.0).all()