"""
tests/test_api.py
-----------------
Integration tests for the FastAPI prediction endpoint.
Run with: pytest tests/test_api.py -v
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Skip API tests if model not available
if not os.path.exists("outputs/edt_model.pkl"):
    pytest.skip("Model not trained yet – run src/train.py first", allow_module_level=True)

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

VALID_ORDER = {
    "store_id":          "Tokyo-Shibuya",
    "order_hour":        19,
    "day_of_week":       "Friday",
    "is_weekend":        0,
    "is_peak_hour":      1,
    "weather_condition": "rain",
    "traffic_level":     "high",
    "vehicle_type":      "scooter",
    "distance_km":       4.2,
    "num_items":         3,
    "prep_time_seconds": 720,
}


class TestHealthEndpoint:

    def test_root_returns_healthy(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_health_returns_healthy(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestPredictEndpoint:

    def test_valid_order_returns_200(self):
        resp = client.post("/predict", json=VALID_ORDER)
        assert resp.status_code == 200

    def test_response_has_required_fields(self):
        resp = client.post("/predict", json=VALID_ORDER)
        data = resp.json()
        assert "edt_seconds" in data
        assert "edt_minutes" in data
        assert "model_version" in data

    def test_edt_seconds_is_positive(self):
        resp = client.post("/predict", json=VALID_ORDER)
        assert resp.json()["edt_seconds"] > 0

    def test_edt_minutes_matches_seconds(self):
        resp = client.post("/predict", json=VALID_ORDER)
        data = resp.json()
        expected_minutes = round(data["edt_seconds"] / 60, 1)
        assert abs(data["edt_minutes"] - expected_minutes) < 0.2

    def test_invalid_day_returns_422(self):
        bad_order = {**VALID_ORDER, "day_of_week": "Funday"}
        resp = client.post("/predict", json=bad_order)
        assert resp.status_code == 422

    def test_invalid_weather_returns_422(self):
        bad_order = {**VALID_ORDER, "weather_condition": "hurricane"}
        resp = client.post("/predict", json=bad_order)
        assert resp.status_code == 422

    def test_negative_distance_returns_422(self):
        bad_order = {**VALID_ORDER, "distance_km": -1.0}
        resp = client.post("/predict", json=bad_order)
        assert resp.status_code == 422

    def test_missing_field_returns_422(self):
        incomplete = {k: v for k, v in VALID_ORDER.items() if k != "store_id"}
        resp = client.post("/predict", json=incomplete)
        assert resp.status_code == 422


class TestBatchEndpoint:

    def test_batch_three_orders(self):
        payload = {"orders": [VALID_ORDER] * 3}
        resp    = client.post("/predict/batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3
        assert len(data["predictions"]) == 3

    def test_batch_exceeds_limit(self):
        payload = {"orders": [VALID_ORDER] * 101}
        resp    = client.post("/predict/batch", json=payload)
        assert resp.status_code == 400

    def test_batch_all_predictions_positive(self):
        payload = {"orders": [VALID_ORDER] * 5}
        resp    = client.post("/predict/batch", json=payload)
        for pred in resp.json()["predictions"]:
            assert pred["edt_seconds"] > 0
