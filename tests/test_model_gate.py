"""
tests/test_model_gate.py
Production gate tests (Phase 3, 13 cities).

Tests gate logic with synthetic values and validates
actual reports/metrics.json if it exists.

Run: pytest tests/test_model_gate.py -v
"""

import sys
import json
import pytest
from pathlib import Path

sys.path.insert(0, ".")

METRICS_PATH  = "reports/metrics.json"
GATE_MAPE_MAX = 22.0
GATE_R2_MIN   = 0.82

ALL_CITIES = [
    "gurgaon", "noida", "chandigarh", "kota",
    "delhi", "mumbai", "bangalore", "chennai",
    "pune", "hyderabad", "ahmedabad", "jaipur", "indore",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Gate logic (synthetic values)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGateThresholds:

    def _gate_passes(self, mape: float, r2: float) -> bool:
        return mape <= GATE_MAPE_MAX and r2 >= GATE_R2_MIN

    def test_passing_metrics_returns_true(self):
        assert self._gate_passes(18.0, 0.87) is True

    def test_mape_above_threshold_fails(self):
        assert self._gate_passes(25.0, 0.87) is False

    def test_r2_below_threshold_fails(self):
        assert self._gate_passes(18.0, 0.75) is False

    def test_both_at_boundary_passes(self):
        assert self._gate_passes(22.0, 0.82) is True

    def test_both_above_threshold_fails(self):
        assert self._gate_passes(25.0, 0.70) is False

    def test_mape_just_below_gate_passes(self):
        assert self._gate_passes(21.99, 0.90) is True

    def test_mape_just_above_gate_fails(self):
        assert self._gate_passes(22.01, 0.90) is False

    def test_r2_just_above_gate_passes(self):
        assert self._gate_passes(20.0, 0.821) is True

    def test_r2_just_below_gate_fails(self):
        assert self._gate_passes(20.0, 0.819) is False


# ═══════════════════════════════════════════════════════════════════════════════
# metrics.json structure
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetricsJson:

    def _load(self):
        if not Path(METRICS_PATH).exists():
            pytest.skip("reports/metrics.json not found — run training first")
        with open(METRICS_PATH) as f:
            return json.load(f)

    def test_metrics_json_exists(self):
        if not Path(METRICS_PATH).exists():
            pytest.skip("reports/metrics.json not found")
        assert Path(METRICS_PATH).exists()

    def test_required_keys_present(self):
        metrics = self._load()
        required = [
            "cv_mape", "cv_r2", "production_gate",
            "baseline_ridge_mape", "improvement_vs_baseline",
            "ensemble_weights", "mlflow_run_id",
        ]
        for key in required:
            assert key in metrics, f"Missing key: {key}"

    # ── Phase 3 keys ──────────────────────────────────────────────────────────

    def test_phase3_n_rows_present(self):
        metrics = self._load()
        assert "n_rows" in metrics
        assert metrics["n_rows"] > 0

    def test_phase3_n_cities_is_13(self):
        metrics = self._load()
        assert "n_cities" in metrics
        assert metrics["n_cities"] == 13, \
            f"Expected 13 cities, got {metrics['n_cities']}"

    def test_phase3_feature_count_present(self):
        metrics = self._load()
        assert "feature_count" in metrics
        assert metrics["feature_count"] >= 50, \
            f"Expected >=50 features (Phase 3), got {metrics['feature_count']}"

    def test_phase3_city_mapes_present(self):
        metrics = self._load()
        assert "city_mapes" in metrics
        city_mapes = metrics["city_mapes"]
        for city in ALL_CITIES:
            assert city in city_mapes, f"Missing city MAPE: {city}"

    def test_phase3_tier_weights_present(self):
        metrics = self._load()
        assert "tier_weights" in metrics
        for tier in ["Tier-1", "Tier-2", "Tier-3"]:
            assert tier in metrics["tier_weights"], f"Missing tier weight: {tier}"

    def test_ensemble_weights_sum_to_one(self):
        metrics = self._load()
        weights = metrics["ensemble_weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, \
            f"Ensemble weights sum to {total}, expected ~1.0"

    # ── Gate thresholds ───────────────────────────────────────────────────────

    def test_mape_below_threshold(self):
        metrics = self._load()
        assert metrics["cv_mape"] <= GATE_MAPE_MAX, \
            f"MAPE {metrics['cv_mape']:.2f}% exceeds {GATE_MAPE_MAX}% gate"

    def test_r2_above_threshold(self):
        metrics = self._load()
        assert metrics["cv_r2"] >= GATE_R2_MIN, \
            f"R² {metrics['cv_r2']:.4f} below {GATE_R2_MIN} gate"

    def test_production_gate_is_pass(self):
        metrics = self._load()
        assert metrics["production_gate"] == "PASS", \
            f"production_gate is '{metrics['production_gate']}', expected 'PASS'"

    def test_improvement_over_baseline_positive(self):
        metrics = self._load()
        assert metrics["improvement_vs_baseline"] > 0, \
            "Ensemble should beat Ridge baseline"

    # ── Per-city MAPE sanity ──────────────────────────────────────────────────

    def test_no_city_mape_above_40(self):
        """No single city should be above 40% -- that would be unacceptable."""
        metrics = self._load()
        for city, mape in metrics["city_mapes"].items():
            assert mape <= 40.0, \
                f"City {city} MAPE {mape:.1f}% is unacceptably high (>40%)"

    def test_best_city_mape_below_20(self):
        """At least one city should be below 20% with 29k rows of data."""
        metrics = self._load()
        best_mape = min(metrics["city_mapes"].values())
        assert best_mape < 20.0, \
            f"Best city MAPE {best_mape:.2f}% should be < 20%"

    def test_n_rows_above_20k(self):
        """Phase 3 should have >20k rows with 13 cities."""
        metrics = self._load()
        assert metrics["n_rows"] >= 20_000, \
            f"Expected >=20k rows (Phase 3), got {metrics['n_rows']}"