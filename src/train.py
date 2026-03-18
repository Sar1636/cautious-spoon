# """
# train.py
# --------
# Train the EDT (Estimated Delivery Time) model.
# Uses only scikit-learn — no xgboost/lightgbm dependency.

# Local usage:
#     python src/train.py --data data/japan_pizza_delivery.csv --output outputs/

# Azure ML pipeline step:
#     python train.py --data <URI_FOLDER> --output <URI_FOLDER>
# """

# import os
# import sys
# import argparse
# import json
# import joblib
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from datetime import datetime, timezone

# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# # MLflow — optional, only used inside Azure ML
# try:
#     import mlflow
#     import mlflow.sklearn
#     MLFLOW_AVAILABLE = True
# except ImportError:
#     MLFLOW_AVAILABLE = False
#     print("MLflow not available — skipping tracking.", flush=True)

# # Import preprocessing helpers from same directory
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from preprocess import load_data, clean_data, build_preprocessor, get_features_and_target


# # ── Argument parsing ──────────────────────────────────────────────────────────
# def parse_args():
#     parser = argparse.ArgumentParser(description="Train EDT model")
#     parser.add_argument("--data",        default="data/japan_pizza_delivery.csv",
#                         help="Path to CSV file OR folder containing a CSV")
#     parser.add_argument("--output",      default="outputs",
#                         help="Output folder for model and metadata")
#     parser.add_argument("--test_size",   type=float, default=0.2)
#     parser.add_argument("--random_seed", type=int,   default=42)
#     return parser.parse_args()


# # ── Resolve folder → CSV ──────────────────────────────────────────────────────
# def resolve_data_path(data_arg: str) -> str:
#     """
#     Azure ML passes a URI_FOLDER (mounted directory) to --data.
#     This finds the actual CSV inside that folder.
#     """
#     p = Path(data_arg)
#     if p.is_dir():
#         print(f"--data is a directory. Contents: {[f.name for f in p.iterdir()]}", flush=True)
#         for name in ["cleaned_data.csv", "japan_pizza_delivery.csv"]:
#             candidate = p / name
#             if candidate.exists():
#                 print(f"Using: {candidate}", flush=True)
#                 return str(candidate)
#         csvs = sorted(p.glob("*.csv"))
#         if csvs:
#             print(f"Using first CSV found: {csvs[0]}", flush=True)
#             return str(csvs[0])
#         print(f"ERROR: No CSV files in {p}", flush=True)
#         raise FileNotFoundError(f"No CSV found in folder: {p}")
#     print(f"--data is a file: {p}", flush=True)
#     return str(p)


# # ── Model candidates (sklearn only) ──────────────────────────────────────────
# def get_model_candidates(seed: int) -> dict:
#     return {
#         "ridge": Ridge(alpha=10.0),
#         "random_forest": RandomForestRegressor(
#             n_estimators=100, max_depth=12,
#             min_samples_leaf=5, random_state=seed, n_jobs=-1,
#         ),
#         "gradient_boosting": GradientBoostingRegressor(
#             n_estimators=200, max_depth=5,
#             learning_rate=0.05, subsample=0.8, random_state=seed,
#         ),
#     }


# # ── Metrics ───────────────────────────────────────────────────────────────────
# def compute_metrics(y_true, y_pred) -> dict:
#     rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
#     mae  = float(mean_absolute_error(y_true, y_pred))
#     r2   = float(r2_score(y_true, y_pred))
#     mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100)
#     return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


# # ── Training ──────────────────────────────────────────────────────────────────
# def train(args):
#     os.makedirs(args.output, exist_ok=True)

#     # 1. Resolve and load data
#     data_path = resolve_data_path(args.data)
#     df = load_data(data_path)
#     df = clean_data(df)

#     # 2. Split
#     X, y = get_features_and_target(df)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=args.test_size, random_state=args.random_seed
#     )
#     print(f"Train: {len(X_train)}  Test: {len(X_test)}", flush=True)

#     # 3. Train all candidates
#     preprocessor  = build_preprocessor()
#     candidates    = get_model_candidates(args.random_seed)
#     results       = {}
#     best_rmse     = float("inf")
#     best_pipeline = None
#     best_name     = None

#     for name, model in candidates.items():
#         print(f"\nTraining {name}...", flush=True)
#         pipe = Pipeline([
#             ("preprocessor", preprocessor),
#             ("model", model),
#         ])
#         cv_scores = cross_val_score(
#             pipe, X_train, y_train,
#             scoring="neg_root_mean_squared_error", cv=3, n_jobs=-1,
#         )
#         cv_rmse = float(-cv_scores.mean())
#         print(f"  CV RMSE: {cv_rmse:.1f}s", flush=True)

#         pipe.fit(X_train, y_train)
#         y_pred  = pipe.predict(X_test)
#         metrics = compute_metrics(y_test.values, y_pred)
#         metrics["CV_RMSE"] = cv_rmse
#         results[name] = metrics
#         print(f"  Test RMSE={metrics['RMSE']:.1f}s  R2={metrics['R2']:.4f}", flush=True)

#         if metrics["RMSE"] < best_rmse:
#             best_rmse     = metrics["RMSE"]
#             best_pipeline = pipe
#             best_name     = name

#     # 4. Save model
#     print(f"\nBest model: {best_name}  RMSE={best_rmse:.1f}s", flush=True)
#     model_path = os.path.join(args.output, "edt_model.pkl")
#     joblib.dump(best_pipeline, model_path)
#     print(f"Model saved: {model_path}", flush=True)

#     # 5. Save metadata
#     final_metrics = compute_metrics(
#         y_test.values, best_pipeline.predict(X_test)
#     )
#     meta = {
#         "model_name":    best_name,
#         "trained_at":    datetime.now(timezone.utc).isoformat(),
#         "train_samples": len(X_train),
#         "test_samples":  len(X_test),
#         "metrics":       final_metrics,
#         "all_results":   results,
#     }
#     meta_path = os.path.join(args.output, "train_metadata.json")
#     with open(meta_path, "w") as f:
#         json.dump(meta, f, indent=2)
#     print(f"Metadata saved: {meta_path}", flush=True)

#     # 6. MLflow logging (inside Azure ML)
#     if MLFLOW_AVAILABLE:
#         try:
#             with mlflow.start_run():
#                 mlflow.log_param("model_type", best_name)
#                 mlflow.log_param("train_size", len(X_train))
#                 for k, v in final_metrics.items():
#                     mlflow.log_metric(k, v)
#                 mlflow.sklearn.log_model(
#                     best_pipeline, "edt_model",
#                     registered_model_name="edt-model-japan",
#                 )
#                 print("MLflow run logged.", flush=True)
#         except Exception as e:
#             print(f"MLflow logging skipped: {e}", flush=True)

#     return best_pipeline, final_metrics


# # ── Entry point ───────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     print("=== train.py starting ===", flush=True)
#     print(f"Python  : {sys.version}", flush=True)
#     print(f"numpy   : {np.__version__}", flush=True)
#     print(f"pandas  : {pd.__version__}", flush=True)

#     args = parse_args()
#     print(f"--data   : {args.data}",   flush=True)
#     print(f"--output : {args.output}", flush=True)

#     model, metrics = train(args)

#     print(f"\nFinal metrics: {metrics}", flush=True)
#     print("=== train.py done ===", flush=True)

"""
train.py
--------
Train the EDT (Estimated Delivery Time) model.
Uses only scikit-learn — no xgboost/lightgbm dependency.

Local usage:
    python src/train.py --data data/japan_pizza_delivery.csv --output outputs/

Azure ML pipeline step:
    python train.py --data <URI_FOLDER> --output <URI_FOLDER>
"""

import os
import sys
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# MLflow — optional, only used inside Azure ML
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available — skipping tracking.", flush=True)

# Import preprocessing helpers from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_data, clean_data, build_preprocessor, get_features_and_target  # noqa: E402


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train EDT model")
    parser.add_argument("--data", default="japan_pizza_delivery.csv",
                        help="Path to CSV file OR folder containing a CSV")
    parser.add_argument("--output", default="outputs",
                        help="Output folder for model and metadata")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


# ── Resolve folder → CSV ──────────────────────────────────────────────────────
def resolve_data_path(data_arg: str) -> str:
    """
    Azure ML passes a URI_FOLDER (mounted directory) to --data.
    This finds the actual CSV inside that folder.
    """
    p = Path(data_arg)
    if p.is_dir():
        print(f"--data is a directory. Contents: {[f.name for f in p.iterdir()]}", flush=True)
        for name in ["cleaned_data.csv", "japan_pizza_delivery.csv"]:
            candidate = p / name
            if candidate.exists():
                print(f"Using: {candidate}", flush=True)
                return str(candidate)
        csvs = sorted(p.glob("*.csv"))
        if csvs:
            print(f"Using first CSV found: {csvs[0]}", flush=True)
            return str(csvs[0])
        print(f"ERROR: No CSV files in {p}", flush=True)
        raise FileNotFoundError(f"No CSV found in folder: {p}")
    print(f"--data is a file: {p}", flush=True)
    return str(p)


# ── Model candidates (sklearn only) ──────────────────────────────────────────
def get_model_candidates(seed: int) -> dict:
    return {
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=12,
            min_samples_leaf=5, random_state=seed, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=5,
            learning_rate=0.05, subsample=0.8, random_state=seed,
        ),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    os.makedirs(args.output, exist_ok=True)

    # 1. Resolve and load data
    data_path = resolve_data_path(args.data)
    df = load_data(data_path)
    df = clean_data(df)

    # 2. Split
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}", flush=True)

    # 3. Train all candidates
    preprocessor = build_preprocessor()
    candidates = get_model_candidates(args.random_seed)
    results = {}
    best_rmse = float("inf")
    best_pipeline = None
    best_name = None

    for name, model in candidates.items():
        print(f"\nTraining {name}...", flush=True)
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        cv_scores = cross_val_score(
            pipe, X_train, y_train,
            scoring="neg_root_mean_squared_error", cv=3, n_jobs=-1,
        )
        cv_rmse = float(-cv_scores.mean())
        print(f"  CV RMSE: {cv_rmse:.1f}s", flush=True)

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = compute_metrics(y_test.values, y_pred)
        metrics["CV_RMSE"] = cv_rmse
        results[name] = metrics
        print(f"  Test RMSE={metrics['RMSE']:.1f}s  R2={metrics['R2']:.4f}", flush=True)

        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_pipeline = pipe
            best_name = name

    # 4. Save model
    print(f"\nBest model: {best_name}  RMSE={best_rmse:.1f}s", flush=True)
    model_path = os.path.join(args.output, "edt_model.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"Model saved: {model_path}", flush=True)

    # 5. Save metadata
    final_metrics = compute_metrics(
        y_test.values, best_pipeline.predict(X_test)
    )
    meta = {
        "model_name": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": final_metrics,
        "all_results": results,
    }
    meta_path = os.path.join(args.output, "train_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {meta_path}", flush=True)

    # 6. MLflow logging (inside Azure ML)
    if MLFLOW_AVAILABLE:
        try:
            with mlflow.start_run():
                mlflow.log_param("model_type", best_name)
                mlflow.log_param("train_size", len(X_train))
                for k, v in final_metrics.items():
                    mlflow.log_metric(k, v)
                mlflow.sklearn.log_model(
                    best_pipeline, "edt_model",
                    registered_model_name="edt-model-japan",
                )
                print("MLflow run logged.", flush=True)
        except Exception as e:
            print(f"MLflow logging skipped: {e}", flush=True)

    return best_pipeline, final_metrics


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== train.py starting ===", flush=True)
    print(f"Python  : {sys.version}", flush=True)
    print(f"numpy   : {np.__version__}", flush=True)
    print(f"pandas  : {pd.__version__}", flush=True)

    args = parse_args()
    print(f"--data   : {args.data}", flush=True)
    print(f"--output : {args.output}", flush=True)

    model, metrics = train(args)

    print(f"\nFinal metrics: {metrics}", flush=True)
    print("=== train.py done ===", flush=True)


