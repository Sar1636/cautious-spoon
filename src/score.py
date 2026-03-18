# """
# score.py
# --------
# Inference logic for the EDT model.
# Used by both the FastAPI server and Azure ML online endpoint.
# """

# import os
# import joblib
# import numpy as np
# import pandas as pd
# from typing import List, Dict

# MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/edt_model.pkl")
# _model = None


# def load_model(path: str = MODEL_PATH):
#     """Load model from disk (cached)."""
#     global _model
#     if _model is None:
#         _model = joblib.load(path)
#         print(f"Model loaded from {path}")
#     return _model


# def predict(features: dict) -> dict:
#     """
#     Run inference on a single order dict.

#     Expected keys:
#         store_id, order_hour, day_of_week, is_weekend, is_peak_hour,
#         weather_condition, traffic_level, vehicle_type,
#         distance_km, num_items, prep_time_seconds

#     Returns:
#         dict with edt_seconds and edt_minutes
#     """
#     model = load_model()
#     df    = pd.DataFrame([features])
#     edt   = float(model.predict(df)[0])
#     edt   = max(0, round(edt))
#     return {
#         "edt_seconds": edt,
#         "edt_minutes": round(edt / 60, 1),
#     }


# def predict_batch(records: List[Dict]) -> List[Dict]:
#     """Run inference on a list of order dicts."""
#     model  = load_model()
#     df     = pd.DataFrame(records)
#     preds  = model.predict(df)
#     return [
#         {"edt_seconds": max(0, round(float(p))), "edt_minutes": round(max(0, float(p)) / 60, 1)}
#         for p in preds
#     ]


# # ── Azure ML entry-point ──────────────────────────────────────────────────────
# def init():
#     """Called once when the Azure ML endpoint container starts."""
#     model_dir = os.environ.get("AZUREML_MODEL_DIR", "outputs")
#     path      = os.path.join(model_dir, "edt_model.pkl")
#     load_model(path)


# def run(raw_data: str) -> str:
#     """Azure ML scoring entry point."""
#     import json
#     data = json.loads(raw_data)
#     if isinstance(data, list):
#         results = predict_batch(data)
#     else:
#         results = predict(data)
#     return json.dumps(results)


"""
score.py
--------
Inference logic for the EDT model.
Used by both the FastAPI server and Azure ML online endpoint.
"""

import os
import json
import joblib
import pandas as pd
from typing import List, Dict

MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/edt_model.pkl")
_model = None


def load_model(path: str = MODEL_PATH):
    """Load model from disk (cached)."""
    global _model
    if _model is None:
        _model = joblib.load(path)
        print(f"Model loaded from {path}")
    return _model


def predict(features: dict) -> dict:
    """
    Run inference on a single order dict.

    Expected keys:
        store_id, order_hour, day_of_week, is_weekend, is_peak_hour,
        weather_condition, traffic_level, vehicle_type,
        distance_km, num_items, prep_time_seconds

    Returns:
        dict with edt_seconds and edt_minutes
    """
    model = load_model()
    df = pd.DataFrame([features])
    edt = float(model.predict(df)[0])
    edt = max(0, round(edt))
    return {
        "edt_seconds": edt,
        "edt_minutes": round(edt / 60, 1),
    }


def predict_batch(records: List[Dict]) -> List[Dict]:
    """Run inference on a list of order dicts."""
    model = load_model()
    df = pd.DataFrame(records)
    preds = model.predict(df)
    return [
        {"edt_seconds": max(0, round(float(p))), "edt_minutes": round(max(0, float(p)) / 60, 1)}
        for p in preds
    ]


# ── Azure ML entry-point ──────────────────────────────────────────────────────
def init():
    """Called once when the Azure ML endpoint container starts."""
    model_dir = os.environ.get("AZUREML_MODEL_DIR", "outputs")
    path = os.path.join(model_dir, "edt_model.pkl")
    load_model(path)


def run(raw_data: str) -> str:
    """Azure ML scoring entry point."""
    data = json.loads(raw_data)
    if isinstance(data, list):
        results = predict_batch(data)
    else:
        results = predict(data)
    return json.dumps(results)


