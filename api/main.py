# """
# main.py
# -------
# FastAPI prediction API for the EDT model.

# Run locally:
#     uvicorn api.main:app --reload --port 8000

# Endpoints:
#     GET  /          -> health check
#     GET  /health    -> health check
#     POST /predict   -> single prediction
#     POST /predict/batch -> batch predictions
# """

# import os
# import sys
# import time
# from typing import List, Optional, Dict
# from contextlib import asynccontextmanager

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field, validator

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# from src.score import load_model, predict, predict_batch


# # ── Pydantic schemas ──────────────────────────────────────────────────────────
# class OrderFeatures(BaseModel):
#     store_id:          str   = Field(..., example="Tokyo-Shibuya")
#     order_hour:        int   = Field(..., ge=0, le=23, example=19)
#     day_of_week:       str   = Field(..., example="Friday")
#     is_weekend:        int   = Field(..., ge=0, le=1, example=0)
#     is_peak_hour:      int   = Field(..., ge=0, le=1, example=1)
#     weather_condition: str   = Field(..., example="clear")
#     traffic_level:     str   = Field(..., example="medium")
#     vehicle_type:      str   = Field(..., example="scooter")
#     distance_km:       float = Field(..., gt=0, le=50, example=3.5)
#     num_items:         int   = Field(..., ge=1, le=20, example=2)
#     prep_time_seconds: int   = Field(..., ge=0, example=600)

#     @validator('day_of_week')
#     def validate_day(cls, v):
#         valid = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
#         if v not in valid:
#             raise ValueError(f"day_of_week must be one of {valid}")
#         return v

#     @validator('weather_condition')
#     def validate_weather(cls, v):
#         valid = ['clear', 'cloudy', 'rain', 'heavy_rain', 'snow']
#         if v not in valid:
#             raise ValueError(f"weather_condition must be one of {valid}")
#         return v

#     @validator('traffic_level')
#     def validate_traffic(cls, v):
#         valid = ['low', 'medium', 'high', 'very_high']
#         if v not in valid:
#             raise ValueError(f"traffic_level must be one of {valid}")
#         return v

#     @validator('vehicle_type')
#     def validate_vehicle(cls, v):
#         valid = ['bicycle', 'scooter', 'car']
#         if v not in valid:
#             raise ValueError(f"vehicle_type must be one of {valid}")
#         return v


# class PredictionResponse(BaseModel):
#     edt_seconds:   int
#     edt_minutes:   float
#     model_version: str = "1.0.0"


# class BatchRequest(BaseModel):
#     orders: List[OrderFeatures]          # fixed: List not list


# class BatchResponse(BaseModel):
#     predictions: List[PredictionResponse]  # fixed: List not list
#     count: int


# # ── App lifecycle ─────────────────────────────────────────────────────────────
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     model_path = os.environ.get("MODEL_PATH", "outputs/edt_model.pkl")
#     load_model(model_path)
#     yield


# app = FastAPI(
#     title="EDT Prediction API",
#     description="Estimated Delivery Time prediction for Japan pizza stores",
#     version="1.0.0",
#     lifespan=lifespan,
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # ── Routes ────────────────────────────────────────────────────────────────────
# @app.get("/")
# @app.get("/health")
# def health():
#     return {"status": "healthy", "service": "edt-prediction-api", "version": "1.0.0"}


# @app.post("/predict", response_model=PredictionResponse)
# def predict_single(order: OrderFeatures):
#     """Predict EDT for a single delivery order."""
#     try:
#         t0     = time.time()
#         result = predict(order.dict())
#         result["model_version"] = "1.0.0"
#         result["latency_ms"]    = round((time.time() - t0) * 1000, 2)
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/predict/batch", response_model=BatchResponse)
# def predict_batch_endpoint(request: BatchRequest):
#     """Predict EDT for multiple delivery orders."""
#     if len(request.orders) > 100:
#         raise HTTPException(status_code=400, detail="Max 100 orders per batch")
#     try:
#         records = [o.dict() for o in request.orders]
#         preds   = predict_batch(records)
#         return {
#             "predictions": [dict(p, model_version="1.0.0") for p in preds],
#             "count": len(preds),
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

"""
main.py
-------
FastAPI prediction API for the EDT model.

Run locally:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /          -> health check
    GET  /health    -> health check
    POST /predict   -> single prediction
    POST /predict/batch -> batch predictions
"""

import os
import sys
import time
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.score import load_model, predict, predict_batch  # noqa: E402


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class OrderFeatures(BaseModel):
    store_id: str = Field(..., example="Tokyo-Shibuya")
    order_hour: int = Field(..., ge=0, le=23, example=19)
    day_of_week: str = Field(..., example="Friday")
    is_weekend: int = Field(..., ge=0, le=1, example=0)
    is_peak_hour: int = Field(..., ge=0, le=1, example=1)
    weather_condition: str = Field(..., example="clear")
    traffic_level: str = Field(..., example="medium")
    vehicle_type: str = Field(..., example="scooter")
    distance_km: float = Field(..., gt=0, le=50, example=3.5)
    num_items: int = Field(..., ge=1, le=20, example=2)
    prep_time_seconds: int = Field(..., ge=0, example=600)

    @validator('day_of_week')
    def validate_day(cls, v):
        valid = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        if v not in valid:
            raise ValueError(f"day_of_week must be one of {valid}")
        return v

    @validator('weather_condition')
    def validate_weather(cls, v):
        valid = ['clear', 'cloudy', 'rain', 'heavy_rain', 'snow']
        if v not in valid:
            raise ValueError(f"weather_condition must be one of {valid}")
        return v

    @validator('traffic_level')
    def validate_traffic(cls, v):
        valid = ['low', 'medium', 'high', 'very_high']
        if v not in valid:
            raise ValueError(f"traffic_level must be one of {valid}")
        return v

    @validator('vehicle_type')
    def validate_vehicle(cls, v):
        valid = ['bicycle', 'scooter', 'car']
        if v not in valid:
            raise ValueError(f"vehicle_type must be one of {valid}")
        return v


class PredictionResponse(BaseModel):
    edt_seconds: int
    edt_minutes: float
    model_version: str = "1.0.0"


class BatchRequest(BaseModel):
    orders: List[OrderFeatures]


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


# ── App lifecycle ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.environ.get("MODEL_PATH", "outputs/edt_model.pkl")
    load_model(model_path)
    yield


app = FastAPI(
    title="EDT Prediction API",
    description="Estimated Delivery Time prediction for Japan pizza stores",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
@app.get("/health")
def health():
    return {"status": "healthy", "service": "edt-prediction-api", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict_single(order: OrderFeatures):
    """Predict EDT for a single delivery order."""
    try:
        t0 = time.time()
        result = predict(order.dict())
        result["model_version"] = "1.0.0"
        result["latency_ms"] = round((time.time() - t0) * 1000, 2)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch_endpoint(request: BatchRequest):
    """Predict EDT for multiple delivery orders."""
    if len(request.orders) > 100:
        raise HTTPException(status_code=400, detail="Max 100 orders per batch")
    try:
        records = [o.dict() for o in request.orders]
        preds = predict_batch(records)
        return {
            "predictions": [dict(p, model_version="1.0.0") for p in preds],
            "count": len(preds),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

