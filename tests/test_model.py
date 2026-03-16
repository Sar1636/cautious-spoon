"""
tests/test_model.py
-------------------
Unit tests for preprocessing, training, and inference.
Run with: pytest tests/ -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import (
    load_data, clean_data, build_preprocessor,
    get_features_and_target, CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Small synthetic dataframe for testing."""
    return pd.DataFrame([
        {
            'order_id':          'JP000001',
            'store_id':          'Tokyo-Shibuya',
            'order_hour':        19,
            'day_of_week':       'Friday',
            'is_weekend':        0,
            'is_peak_hour':      1,
            'weather_condition': 'rain',
            'traffic_level':     'high',
            'vehicle_type':      'scooter',
            'distance_km':       4.2,
            'num_items':         3,
            'prep_time_seconds': 720,
            'edt_seconds':       2100,
        },
        {
            'order_id':          'JP000002',
            'store_id':          'Osaka-Namba',
            'order_hour':        12,
            'day_of_week':       'Saturday',
            'is_weekend':        1,
            'is_peak_hour':      1,
            'weather_condition': 'clear',
            'traffic_level':     'medium',
            'vehicle_type':      'bicycle',
            'distance_km':       1.5,
            'num_items':         1,
            'prep_time_seconds': 400,
            'edt_seconds':       900,
        },
    ] * 50)  # repeat to get enough rows for fitting


@pytest.fixture
def trained_model():
    """Load trained model if exists, else skip."""
    path = "outputs/edt_model.pkl"
    if not os.path.exists(path):
        pytest.skip("Trained model not found – run train.py first")
    return joblib.load(path)


# ── Preprocessing tests ───────────────────────────────────────────────────────
class TestPreprocessing:

    def test_clean_data_removes_nulls(self, sample_df):
        sample_df.loc[0, TARGET_COL] = None
        cleaned = clean_data(sample_df)
        assert cleaned[TARGET_COL].isnull().sum() == 0

    def test_clean_data_removes_duplicates(self, sample_df):
        dup_df  = pd.concat([sample_df, sample_df.iloc[:5]])
        cleaned = clean_data(dup_df)
        assert cleaned['order_id'].duplicated().sum() == 0

    def test_get_features_and_target_shapes(self, sample_df):
        X, y = get_features_and_target(sample_df)
        assert X.shape[1] == len(CATEGORICAL_COLS) + len(NUMERICAL_COLS)
        assert len(y) == len(sample_df)
        assert y.name == TARGET_COL

    def test_preprocessor_fit_transform(self, sample_df):
        X, _ = get_features_and_target(sample_df)
        preprocessor = build_preprocessor()
        X_transformed = preprocessor.fit_transform(X)
        assert X_transformed.shape[0] == len(sample_df)
        assert not np.isnan(X_transformed).any()

    def test_preprocessor_handles_unknown_categories(self, sample_df):
        X, _     = get_features_and_target(sample_df)
        preprocessor = build_preprocessor()
        preprocessor.fit(X)
        # Introduce unknown category
        X_new = X.copy()
        X_new.loc[0, 'store_id'] = 'UNKNOWN-STORE'
        result = preprocessor.transform(X_new)
        assert result is not None


# ── Inference tests ───────────────────────────────────────────────────────────
class TestInference:

    def test_single_prediction_returns_positive(self, trained_model):
        features = pd.DataFrame([{
            'store_id':          'Tokyo-Shibuya',
            'order_hour':        19,
            'day_of_week':       'Friday',
            'is_weekend':        0,
            'is_peak_hour':      1,
            'weather_condition': 'rain',
            'traffic_level':     'high',
            'vehicle_type':      'scooter',
            'distance_km':       4.2,
            'num_items':         3,
            'prep_time_seconds': 720,
        }])
        pred = trained_model.predict(features)[0]
        assert pred > 0, "Prediction must be positive"

    def test_batch_prediction_length(self, trained_model):
        features = pd.DataFrame([{
            'store_id': 'Osaka-Namba', 'order_hour': 12,
            'day_of_week': 'Saturday', 'is_weekend': 1, 'is_peak_hour': 1,
            'weather_condition': 'clear', 'traffic_level': 'medium',
            'vehicle_type': 'bicycle', 'distance_km': 1.5,
            'num_items': 1, 'prep_time_seconds': 400,
        }] * 10)
        preds = trained_model.predict(features)
        assert len(preds) == 10

    def test_longer_distance_means_longer_edt(self, trained_model):
        """Higher distance_km should generally predict higher EDT."""
        base = {
            'store_id': 'Tokyo-Shibuya', 'order_hour': 14,
            'day_of_week': 'Wednesday', 'is_weekend': 0, 'is_peak_hour': 0,
            'weather_condition': 'clear', 'traffic_level': 'low',
            'vehicle_type': 'scooter', 'num_items': 2, 'prep_time_seconds': 500,
        }
        short = pd.DataFrame([{**base, 'distance_km': 1.0}])
        long_ = pd.DataFrame([{**base, 'distance_km': 15.0}])
        assert trained_model.predict(long_)[0] > trained_model.predict(short)[0]

    def test_bad_weather_means_longer_edt(self, trained_model):
        """Snow should predict higher EDT than clear weather."""
        base = {
            'store_id': 'Sapporo-Odori', 'order_hour': 18,
            'day_of_week': 'Tuesday', 'is_weekend': 0, 'is_peak_hour': 1,
            'traffic_level': 'medium', 'vehicle_type': 'scooter',
            'distance_km': 5.0, 'num_items': 2, 'prep_time_seconds': 600,
        }
        clear = pd.DataFrame([{**base, 'weather_condition': 'clear'}])
        snow  = pd.DataFrame([{**base, 'weather_condition': 'snow'}])
        assert trained_model.predict(snow)[0] > trained_model.predict(clear)[0]
