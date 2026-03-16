"""
sample_request.py
-----------------
Demo script showing how to make prediction requests to the EDT API.

Usage:
    # Make sure the API is running first:
    #   uvicorn api.main:app --port 8000
    python api/sample_request.py
"""

import json
import requests

BASE_URL = "http://localhost:8000"


def print_section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


# ── 1. Health check ───────────────────────────────────────────────────────────
print_section("1. Health Check")
resp = requests.get(f"{BASE_URL}/health")
print(f"Status: {resp.status_code}")
print(json.dumps(resp.json(), indent=2))


# ── 2. Single prediction ──────────────────────────────────────────────────────
print_section("2. Single Order Prediction")

order = {
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

print("Request payload:")
print(json.dumps(order, indent=2))

resp = requests.post(f"{BASE_URL}/predict", json=order)
print(f"\nResponse ({resp.status_code}):")
result = resp.json()
print(json.dumps(result, indent=2))
print(f"\n→ Estimated delivery: {result['edt_minutes']} minutes ({result['edt_seconds']} seconds)")


# ── 3. Batch prediction ────────────────────────────────────────────────────────
print_section("3. Batch Prediction (3 orders)")

batch_orders = [
    {
        "store_id": "Osaka-Namba", "order_hour": 12, "day_of_week": "Saturday",
        "is_weekend": 1, "is_peak_hour": 1, "weather_condition": "clear",
        "traffic_level": "medium", "vehicle_type": "bicycle",
        "distance_km": 1.5, "num_items": 1, "prep_time_seconds": 400,
    },
    {
        "store_id": "Tokyo-Akihabara", "order_hour": 20, "day_of_week": "Sunday",
        "is_weekend": 1, "is_peak_hour": 1, "weather_condition": "snow",
        "traffic_level": "very_high", "vehicle_type": "car",
        "distance_km": 8.0, "num_items": 6, "prep_time_seconds": 800,
    },
    {
        "store_id": "Kyoto-Gion", "order_hour": 14, "day_of_week": "Wednesday",
        "is_weekend": 0, "is_peak_hour": 0, "weather_condition": "cloudy",
        "traffic_level": "low", "vehicle_type": "scooter",
        "distance_km": 2.8, "num_items": 2, "prep_time_seconds": 500,
    },
]

resp = requests.post(f"{BASE_URL}/predict/batch", json={"orders": batch_orders})
print(f"Response ({resp.status_code}):")
data = resp.json()
for i, pred in enumerate(data["predictions"]):
    store = batch_orders[i]["store_id"]
    print(f"  Order {i+1} [{store}]: {pred['edt_minutes']} min ({pred['edt_seconds']}s)")
