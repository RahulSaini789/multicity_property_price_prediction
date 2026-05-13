"""
tests/test_features.py
Unit tests for feature engineering functions (Phase 3, 13 cities).

Run: pytest tests/test_features.py -v
"""

import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.features.feature_engineering import (
    parse_amenities_full,
    parse_avg_rating,
    parse_furnish_score,
    build_property_type_flags,
    build_ratio_features,
    build_floor_features,
    build_age_bucket,
    build_nearby_flags,
    build_osm_distance_columns,
    validate_features,
    FINAL_FEATURES,
    AMENITY_CATALOG,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Amenity parsing (Phase 3: parse_amenities_full replaces parse_amenity_score)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAmenitiesFull:

    def test_pool_gym_lift_score(self):
        result = parse_amenities_full("gym,pool,lift")
        # pool=3 + gym=2 + lift=1 = 6.0
        assert result["amenity_score"] == pytest.approx(6.0, abs=0.1)

    def test_empty_string_returns_zero(self):
        result = parse_amenities_full("")
        assert result["amenity_score"] == 0.0
        assert result["amenity_count"] == 0

    def test_nan_returns_zero(self):
        result = parse_amenities_full(np.nan)
        assert result["amenity_score"] == 0.0
        assert result["amenity_count"] == 0

    def test_pool_flag_set(self):
        result = parse_amenities_full("swimming pool, gym")
        assert result["has_pool"] == 1
        assert result["has_gym"] == 1

    def test_security_flag_set(self):
        result = parse_amenities_full("24x7 security")
        assert result["has_security"] == 1
        assert result["amenity_score"] == pytest.approx(0.5, abs=0.1)

    def test_no_match_flags_zero(self):
        result = parse_amenities_full("some random text")
        assert result["has_pool"] == 0
        assert result["has_gym"] == 0
        assert result["has_lift"] == 0

    def test_amenity_count_correct(self):
        result = parse_amenities_full("pool,gym,lift,security")
        assert result["amenity_count"] == 4

    def test_tier_luxury(self):
        result = parse_amenities_full("pool,gym,lift,security,power backup,club,tennis,badminton,spa")
        assert result["amenity_tier"] == "luxury"

    def test_tier_none(self):
        result = parse_amenities_full("")
        assert result["amenity_tier"] == "none"

    def test_tier_basic(self):
        result = parse_amenities_full("lift")
        assert result["amenity_tier"] == "basic"

    def test_all_flags_present_in_output(self):
        result = parse_amenities_full("pool")
        for _, _, flag in AMENITY_CATALOG:
            assert flag in result, f"Missing flag: {flag}"


# ═══════════════════════════════════════════════════════════════════════════════
# Rating parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestAvgRating:

    def test_direct_float(self):
        assert parse_avg_rating(4.2) == pytest.approx(4.2, abs=0.01)

    def test_string_format(self):
        assert parse_avg_rating("4.5 out of 5") == pytest.approx(4.5, abs=0.01)

    def test_nan_returns_nan(self):
        assert np.isnan(parse_avg_rating(np.nan))

    def test_over_five_scale_converts(self):
        # 42 on a 50-point scale → 4.2
        result = parse_avg_rating(42)
        assert result == pytest.approx(4.2, abs=0.01)

    def test_valid_range_preserved(self):
        for val in [1.0, 2.5, 3.0, 4.8, 5.0]:
            result = parse_avg_rating(val)
            assert 1.0 <= result <= 5.0


# ═══════════════════════════════════════════════════════════════════════════════
# Furnish score
# ═══════════════════════════════════════════════════════════════════════════════

class TestFurnishScore:

    def test_unfurnished_zero(self):
        assert parse_furnish_score("unfurnished") == 0.0

    def test_semi_furnished_half(self):
        assert parse_furnish_score("semi-furnished") == 0.5

    def test_furnished_one(self):
        assert parse_furnish_score("furnished") == 1.0

    def test_nan_returns_zero(self):
        assert parse_furnish_score(np.nan) == 0.0

    def test_case_insensitive(self):
        assert parse_furnish_score("FURNISHED") == 1.0
        assert parse_furnish_score("Semi-Furnished") == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# Property type flags (Phase 3: is_plot added)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropertyTypeFlags:

    def _make_df(self, types):
        return pd.DataFrame({"property_type": types})

    def test_flat_flag(self):
        df = self._make_df(["flat", "house"])
        df = build_property_type_flags(df)
        assert df.loc[0, "is_flat"] == 1
        assert df.loc[0, "is_house"] == 0

    def test_house_flag(self):
        df = self._make_df(["house"])
        df = build_property_type_flags(df)
        assert df.loc[0, "is_house"] == 1
        assert df.loc[0, "is_flat"] == 0

    def test_plot_flag(self):
        df = self._make_df(["plot"])
        df = build_property_type_flags(df)
        assert df.loc[0, "is_plot"] == 1
        assert df.loc[0, "is_flat"] == 0

    def test_independent_floor_flag(self):
        df = self._make_df(["independent_floor"])
        df = build_property_type_flags(df)
        assert df.loc[0, "is_independent_floor"] == 1

    def test_flags_mutually_exclusive(self):
        df = self._make_df(["flat", "house", "independent_floor", "plot"])
        df = build_property_type_flags(df)
        row_sums = df[["is_flat", "is_house", "is_independent_floor", "is_plot"]].sum(axis=1)
        assert (row_sums == 1).all(), "Each row should have exactly one type flag set"

    def test_all_13_city_data_has_valid_types(self):
        """Simulate data from all 13 cities — all types should be handled."""
        all_types = ["flat", "house", "independent_floor", "plot"] * 13
        df = self._make_df(all_types)
        df = build_property_type_flags(df)
        assert df[["is_flat", "is_house", "is_independent_floor", "is_plot"]].isnull().sum().sum() == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Ratio features
# ═══════════════════════════════════════════════════════════════════════════════

class TestRatioFeatures:

    def _make_df(self):
        return pd.DataFrame({
            "bathroom": [2, 3, 4],
            "bhk":      [2, 3, 2],
            "area":     [1000, 1500, 2000],
        })

    def test_bath_per_bed_computed(self):
        df = build_ratio_features(self._make_df())
        assert df.loc[0, "bath_per_bed"] == pytest.approx(1.0, abs=0.01)

    def test_log_area_positive(self):
        df = build_ratio_features(self._make_df())
        assert (df["log_area"] > 0).all()

    def test_area_per_bhk_computed(self):
        df = build_ratio_features(self._make_df())
        assert df.loc[0, "area_per_bhk"] == pytest.approx(500.0, abs=1.0)

    def test_plot_bhk_zero_no_division_error(self):
        """Plots have bhk=0 -- area_per_bhk should not divide by zero."""
        df = pd.DataFrame({
            "bathroom": [0],
            "bhk":      [0],
            "area":     [2000],
        })
        df = build_ratio_features(df)
        assert not np.isnan(df.loc[0, "area_per_bhk"])
        assert not np.isinf(df.loc[0, "area_per_bhk"])


# ═══════════════════════════════════════════════════════════════════════════════
# Floor features
# ═══════════════════════════════════════════════════════════════════════════════

class TestFloorFeatures:

    def _make_df(self):
        return pd.DataFrame({
            "floor_pos":    [0, 5, 10],
            "total_floors": [1, 10, 10],
        })

    def test_relative_floor_range(self):
        df = build_floor_features(self._make_df())
        assert (df["relative_floor"] >= 0).all()
        assert (df["relative_floor"] <= 1).all()

    def test_top_floor_is_high(self):
        df = build_floor_features(self._make_df())
        assert df.loc[2, "is_high_floor"] == 1  # 10/10 = 1.0 >= 0.75

    def test_ground_floor_not_high(self):
        df = build_floor_features(self._make_df())
        assert df.loc[0, "is_high_floor"] == 0

    def test_plot_floor_handling(self):
        """Plots have floor_pos=0, total_floors=1 -- should not crash."""
        df = pd.DataFrame({"floor_pos": [0], "total_floors": [1]})
        df = build_floor_features(df)
        assert df.loc[0, "relative_floor"] == pytest.approx(0.0, abs=0.01)
        assert df.loc[0, "is_high_floor"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Age bucket
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgeBucket:

    def _make_df(self, ages):
        return pd.DataFrame({"age": ages})

    def test_under_construction_is_zero(self):
        df = build_age_bucket(self._make_df(["under construction"]))
        assert df.loc[0, "age_bucket"] == 0

    def test_zero_to_one_is_one(self):
        df = build_age_bucket(self._make_df(["0-1 years"]))
        assert df.loc[0, "age_bucket"] == 1

    def test_one_to_five_is_two(self):
        df = build_age_bucket(self._make_df(["1-5 years"]))
        assert df.loc[0, "age_bucket"] == 2

    def test_less_than_five_is_two(self):
        df = build_age_bucket(self._make_df(["less than 5 years"]))
        assert df.loc[0, "age_bucket"] == 2

    def test_five_to_ten_is_three(self):
        df = build_age_bucket(self._make_df(["5-10 years"]))
        assert df.loc[0, "age_bucket"] == 3

    def test_ten_plus_is_four(self):
        df = build_age_bucket(self._make_df(["10+ years"]))
        assert df.loc[0, "age_bucket"] == 4

    def test_unknown_defaults_to_two(self):
        df = build_age_bucket(self._make_df(["some unknown value"]))
        assert df.loc[0, "age_bucket"] == 2

    def test_plot_age_land_is_zero(self):
        df = build_age_bucket(self._make_df(["land"]))
        assert df.loc[0, "age_bucket"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Nearby flags (Phase 3: 6 flags + nearby_score + is_well_served)
# ═══════════════════════════════════════════════════════════════════════════════

class TestNearbyFlags:

    def _make_df(self, locations):
        return pd.DataFrame({"nearbylocations": locations})

    def test_metro_flag(self):
        df = build_nearby_flags(self._make_df(["metro station nearby"]))
        assert df.loc[0, "has_metro_nearby"] == 1

    def test_hospital_flag(self):
        df = build_nearby_flags(self._make_df(["hospital and clinic"]))
        assert df.loc[0, "has_hospital_nearby"] == 1

    def test_school_flag(self):
        df = build_nearby_flags(self._make_df(["school nearby"]))
        assert df.loc[0, "has_school_nearby"] == 1

    def test_market_flag(self):
        df = build_nearby_flags(self._make_df(["supermarket and mall"]))
        assert df.loc[0, "has_market_nearby"] == 1

    def test_park_flag(self):
        df = build_nearby_flags(self._make_df(["park and garden"]))
        assert df.loc[0, "has_park_nearby"] == 1

    def test_police_flag(self):
        df = build_nearby_flags(self._make_df(["police station"]))
        assert df.loc[0, "has_police_nearby"] == 1

    def test_no_match_returns_zero(self):
        df = build_nearby_flags(self._make_df(["some random text"]))
        assert df.loc[0, "has_metro_nearby"] == 0
        assert df.loc[0, "has_hospital_nearby"] == 0

    def test_empty_returns_zeros(self):
        df = build_nearby_flags(self._make_df([""]))
        for col in ["has_metro_nearby", "has_hospital_nearby", "has_school_nearby",
                    "has_mall_nearby", "has_market_nearby", "has_park_nearby",
                    "has_police_nearby"]:
            assert df.loc[0, col] == 0

    def test_nearby_score_computed(self):
        """Metro(3) + Hospital(2) = 5.0 nearby_score."""
        df = build_nearby_flags(self._make_df(["metro and hospital nearby"]))
        assert df.loc[0, "nearby_score"] == pytest.approx(5.0, abs=0.1)

    def test_is_well_served_threshold(self):
        """nearby_score >= 5 → is_well_served = 1."""
        df = build_nearby_flags(self._make_df(["metro and hospital nearby"]))
        assert df.loc[0, "is_well_served"] == 1

    def test_is_not_well_served_below_threshold(self):
        """Only park (1.0) → is_well_served = 0."""
        df = build_nearby_flags(self._make_df(["park nearby"]))
        assert df.loc[0, "is_well_served"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# OSM distance columns (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOSMDistanceColumns:

    def test_missing_columns_filled_with_minus_one(self):
        df = pd.DataFrame({"area": [1000, 2000]})
        df = build_osm_distance_columns(df)
        for col in ["dist_hospital_km", "dist_school_km", "dist_metro_km",
                    "dist_market_km", "dist_park_km"]:
            assert col in df.columns
            assert (df[col] == -1.0).all()

    def test_existing_values_preserved(self):
        df = pd.DataFrame({
            "area":             [1000],
            "dist_hospital_km": [1.5],
            "dist_school_km":   [0.8],
        })
        df = build_osm_distance_columns(df)
        assert df.loc[0, "dist_hospital_km"] == pytest.approx(1.5, abs=0.01)
        assert df.loc[0, "dist_school_km"] == pytest.approx(0.8, abs=0.01)

    def test_invalid_values_replaced_with_minus_one(self):
        df = pd.DataFrame({
            "area":             [1000],
            "dist_hospital_km": [float("nan")],
        })
        df = build_osm_distance_columns(df)
        assert df.loc[0, "dist_hospital_km"] == -1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Feature validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateFeatures:

    def _make_valid_df(self, features):
        return pd.DataFrame({f: [1.0, 2.0, 3.0] for f in features})

    def test_clean_df_passes(self):
        feats = ["area", "bhk", "city_tier_num"]
        df = self._make_valid_df(feats)
        assert validate_features(df, feats) is True

    def test_missing_feature_fails(self):
        feats = ["area", "bhk", "city_tier_num"]
        df = self._make_valid_df(feats[:2])
        assert validate_features(df, feats) is False

    def test_nan_column_fails(self):
        feats = ["area", "bhk"]
        df = self._make_valid_df(feats)
        df.loc[0, "area"] = np.nan
        assert validate_features(df, feats) is False

    def test_inf_column_fails(self):
        feats = ["area", "bhk"]
        df = self._make_valid_df(feats)
        df.loc[0, "area"] = np.inf
        assert validate_features(df, feats) is False

    def test_phase3_features_in_final_list(self):
        """Ensure Phase 3 features are declared in FINAL_FEATURES."""
        phase3_features = [
            "has_pool", "has_gym", "has_lift",
            "has_hospital_nearby", "has_market_nearby",
            "nearby_score", "is_well_served",
            "dist_hospital_km", "dist_school_km",
            "rera_approved",
        ]
        for feat in phase3_features:
            assert feat in FINAL_FEATURES, f"Phase 3 feature '{feat}' missing from FINAL_FEATURES"