"""
tests/test_api.py
Integration tests for FastAPI endpoints (Phase 3, 13 cities).

Uses TestClient — no real server needed.
Models are NOT loaded in tests — endpoints requiring models
return 503, which is expected and tested for.

Run: pytest tests/test_api.py -v
"""

import sys
import pytest

sys.path.insert(0, ".")

from fastapi.testclient import TestClient


# ── All 13 supported cities ───────────────────────────────────────────────────
ALL_CITIES = [
    "gurgaon", "noida", "chandigarh", "kota",
    "delhi", "mumbai", "bangalore", "chennai",
    "pune", "hyderabad", "ahmedabad", "jaipur", "indore",
]

INVALID_CITIES = ["london", "newyork", "tokyo", "sydney", "lahore"]


@pytest.fixture
def client():
    """
    TestClient with lifespan disabled for unit tests.
    Tests HTTP layer and Pydantic validation only.
    Model-dependent endpoints return 503 (no models loaded in test env).
    """
    from src.serving.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ═══════════════════════════════════════════════════════════════════════════════
# Root endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestRootEndpoint:

    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_contains_cities(self, client):
        data = client.get("/").json()
        assert "cities" in data
        assert len(data["cities"]) == 13

    def test_root_all_13_cities_present(self, client):
        data = client.get("/").json()
        for city in ALL_CITIES:
            assert city in data["cities"], f"Missing city: {city}"

    def test_root_contains_endpoints(self, client):
        data = client.get("/").json()
        assert "endpoints" in data
        for ep in ["/predict", "/recommend", "/health", "/model-info"]:
            assert ep in data["endpoints"]

    def test_root_version_is_3(self, client):
        data = client.get("/").json()
        assert "3" in str(data.get("version", ""))


# ═══════════════════════════════════════════════════════════════════════════════
# Health endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_status_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_has_models_loaded(self, client):
        data = client.get("/health").json()
        assert "models_loaded" in data
        for key in ["xgb", "lgb", "cat", "shap"]:
            assert key in data["models_loaded"]

    def test_health_has_uptime(self, client):
        data = client.get("/health").json()
        assert data["uptime_seconds"] >= 0

    def test_health_n_cities_is_13(self, client):
        data = client.get("/health").json()
        assert data.get("n_cities") == 13

    def test_health_has_shap_mode(self, client):
        data = client.get("/health").json()
        assert "shap_mode" in data


# ═══════════════════════════════════════════════════════════════════════════════
# Model info endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelInfoEndpoint:

    def test_model_info_returns_200(self, client):
        assert client.get("/model-info").status_code == 200

    def test_model_info_has_cities(self, client):
        data = client.get("/model-info").json()
        assert "cities" in data
        assert len(data["cities"]) == 13

    def test_model_info_has_city_mapes(self, client):
        data = client.get("/model-info").json()
        assert "city_mapes" in data
        for city in ALL_CITIES:
            assert city in data["city_mapes"], f"Missing city MAPE: {city}"

    def test_model_info_has_weights(self, client):
        data = client.get("/model-info").json()
        assert "weights" in data
        assert abs(sum(data["weights"].values()) - 1.0) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# Predict validation (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredictValidation:

    # Base valid payload using Mumbai (Phase 3 city)
    VALID_PAYLOAD = {
        "city":          "mumbai",
        "property_type": "flat",
        "area_sqft":     950,
        "bhk":           2,
        "bathroom":      2,
        "floor_pos":     8,
        "total_floors":  20,
        "amenity_score": 3.5,
        "avg_rating":    4.2,
    }

    # ── City validation ───────────────────────────────────────────────────────

    def test_invalid_city_returns_error(self, client):
        payload = {**self.VALID_PAYLOAD, "city": "london"}
        resp = client.post("/predict", json=payload)
        # 422 (pydantic) or 503 (no models but city rejected) or 422 from endpoint
        assert resp.status_code in (422, 503)

    def test_all_13_cities_pass_validation(self, client):
        for city in ALL_CITIES:
            payload = {**self.VALID_PAYLOAD, "city": city}
            resp = client.post("/predict", json=payload)
            # 503 = models not loaded (expected), 200 = success
            # 422 = validation failure (NOT expected for valid city)
            assert resp.status_code in (200, 503), \
                f"Unexpected {resp.status_code} for city={city}"

    def test_invalid_cities_rejected(self, client):
        for city in INVALID_CITIES:
            payload = {**self.VALID_PAYLOAD, "city": city}
            resp = client.post("/predict", json=payload)
            assert resp.status_code in (422, 503), \
                f"City '{city}' should be rejected but got {resp.status_code}"

    # ── Property type validation ──────────────────────────────────────────────

    def test_all_valid_property_types_pass(self, client):
        for pt in ["flat", "house", "independent_floor", "plot"]:
            payload = {**self.VALID_PAYLOAD, "property_type": pt}
            resp = client.post("/predict", json=payload)
            assert resp.status_code in (200, 503), \
                f"Unexpected {resp.status_code} for type={pt}"

    def test_invalid_property_type_rejected(self, client):
        payload = {**self.VALID_PAYLOAD, "property_type": "penthouse"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code in (422, 503)

    # ── Area validation ───────────────────────────────────────────────────────

    def test_area_below_minimum_returns_422(self, client):
        payload = {**self.VALID_PAYLOAD, "area_sqft": 50}
        assert client.post("/predict", json=payload).status_code == 422

    def test_area_above_maximum_returns_422(self, client):
        payload = {**self.VALID_PAYLOAD, "area_sqft": 99999}
        assert client.post("/predict", json=payload).status_code == 422

    def test_valid_plot_area_accepted(self, client):
        """Plots can be large -- 10000 sqft should be valid."""
        payload = {**self.VALID_PAYLOAD, "property_type": "plot", "area_sqft": 10000, "bhk": 0}
        resp = client.post("/predict", json=payload)
        assert resp.status_code in (200, 503)

    # ── BHK validation ────────────────────────────────────────────────────────

    def test_bhk_above_max_returns_422(self, client):
        payload = {**self.VALID_PAYLOAD, "bhk": 15}
        assert client.post("/predict", json=payload).status_code == 422

    def test_plot_bhk_zero_accepted(self, client):
        """Phase 3: plots have bhk=0 which must be valid."""
        payload = {**self.VALID_PAYLOAD, "property_type": "plot", "bhk": 0}
        resp = client.post("/predict", json=payload)
        assert resp.status_code in (200, 503)

    # ── Required fields ───────────────────────────────────────────────────────

    def test_missing_city_returns_422(self, client):
        payload = {k: v for k, v in self.VALID_PAYLOAD.items() if k != "city"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_missing_area_returns_422(self, client):
        payload = {k: v for k, v in self.VALID_PAYLOAD.items() if k != "area_sqft"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_missing_bhk_returns_422(self, client):
        payload = {k: v for k, v in self.VALID_PAYLOAD.items() if k != "bhk"}
        assert client.post("/predict", json=payload).status_code == 422

    # ── Phase 3: nearby + amenity flags ──────────────────────────────────────

    def test_nearby_flags_accepted(self, client):
        payload = {
            **self.VALID_PAYLOAD,
            "has_metro_nearby":    1,
            "has_hospital_nearby": 1,
            "has_school_nearby":   0,
            "has_market_nearby":   1,
            "has_park_nearby":     0,
            "has_police_nearby":   0,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code in (200, 503)

    def test_amenity_flags_accepted(self, client):
        payload = {
            **self.VALID_PAYLOAD,
            "has_pool":         1,
            "has_gym":          1,
            "has_lift":         1,
            "has_security":     1,
            "has_power_backup": 0,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code in (200, 503)

    def test_osm_distances_accepted(self, client):
        payload = {
            **self.VALID_PAYLOAD,
            "dist_hospital_km": 1.2,
            "dist_school_km":   0.5,
            "dist_metro_km":    2.3,
            "dist_market_km":   0.8,
            "dist_park_km":     1.5,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code in (200, 503)

    def test_unknown_osm_minus_one_accepted(self, client):
        """Phase 3: dist=-1 means unknown, must be valid."""
        payload = {
            **self.VALID_PAYLOAD,
            "dist_hospital_km": -1.0,
            "dist_school_km":   -1.0,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code in (200, 503)


# ═══════════════════════════════════════════════════════════════════════════════
# Recommend endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecommendEndpoint:

    VALID_PAYLOAD = {
        "city":       "bangalore",
        "budget_cr":  1.2,
        "bhk":        2,
        "property_type": "flat",
        "area_min":   900.0,
        "area_max":   1400.0,
    }

    def test_valid_request_returns_200_or_503(self, client):
        resp = client.post("/recommend", json=self.VALID_PAYLOAD)
        assert resp.status_code in (200, 503)

    def test_all_13_cities_accepted(self, client):
        for city in ALL_CITIES:
            payload = {**self.VALID_PAYLOAD, "city": city}
            resp = client.post("/recommend", json=payload)
            assert resp.status_code in (200, 503), \
                f"Unexpected {resp.status_code} for city={city}"

    def test_invalid_city_rejected(self, client):
        payload = {**self.VALID_PAYLOAD, "city": "london"}
        resp = client.post("/recommend", json=payload)
        assert resp.status_code in (422, 503)

    def test_missing_budget_returns_422(self, client):
        payload = {k: v for k, v in self.VALID_PAYLOAD.items() if k != "budget_cr"}
        assert client.post("/recommend", json=payload).status_code == 422

    def test_top_n_max_respected(self, client):
        payload = {**self.VALID_PAYLOAD, "top_n": 25}
        assert client.post("/recommend", json=payload).status_code == 422

    def test_top_n_min_respected(self, client):
        payload = {**self.VALID_PAYLOAD, "top_n": 0}
        assert client.post("/recommend", json=payload).status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
# Analytics endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalyticsEndpoint:

    def test_analytics_list_returns_200(self, client):
        assert client.get("/analytics").status_code == 200

    def test_analytics_missing_report_returns_404(self, client):
        assert client.get("/analytics/nonexistent_report_xyz").status_code == 404

    def test_analytics_path_traversal_blocked(self, client):
        resp = client.get("/analytics/../../../etc/passwd")
        assert resp.status_code in (404, 422, 400)

    def test_analytics_json_extension_added_auto(self, client):
        """Requesting without .json should not crash."""
        resp = client.get("/analytics/market_summary")
        assert resp.status_code in (200, 404)