# """
# preprocess.py
# -------------
# Shared preprocessing logic — used by both the Azure ML pipeline step
# and the local training script.

# As a script (Azure ML step):
#     python preprocess.py --input <csv_file_or_folder> --output <folder>

# As a module (imported by train.py):
#     from preprocess import load_data, clean_data, build_preprocessor, ...
# """

# import pandas as pd
# import numpy as np
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OrdinalEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer


# # ── Column groups ─────────────────────────────────────────────────────────────
# CATEGORICAL_COLS = [
#     "store_id", "day_of_week", "weather_condition",
#     "traffic_level", "vehicle_type",
# ]
# NUMERICAL_COLS = [
#     "order_hour", "is_weekend", "is_peak_hour",
#     "distance_km", "num_items", "prep_time_seconds",
# ]
# TARGET_COL = "edt_seconds"


# # ── Helpers ───────────────────────────────────────────────────────────────────
# def load_data(filepath: str) -> pd.DataFrame:
#     df = pd.read_csv(filepath)
#     print(f"Loaded {len(df)} rows  cols={list(df.columns)}", flush=True)
#     return df


# def clean_data(df: pd.DataFrame) -> pd.DataFrame:
#     initial = len(df)
#     df = df.drop_duplicates(subset="order_id")
#     df = df.dropna(subset=[TARGET_COL])
#     df[NUMERICAL_COLS]   = df[NUMERICAL_COLS].fillna(df[NUMERICAL_COLS].median())
#     df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna("unknown")
#     print(f"Cleaned: {initial} -> {len(df)} rows", flush=True)
#     return df


# def build_preprocessor() -> ColumnTransformer:
#     cat_pipeline = Pipeline([
#         ("encoder", OrdinalEncoder(
#             handle_unknown="use_encoded_value", unknown_value=-1
#         ))
#     ])
#     num_pipeline = Pipeline([("scaler", StandardScaler())])
#     return ColumnTransformer(
#         transformers=[
#             ("cat", cat_pipeline, CATEGORICAL_COLS),
#             ("num", num_pipeline, NUMERICAL_COLS),
#         ],
#         remainder="drop",
#     )


# def get_features_and_target(df: pd.DataFrame):
#     X = df[CATEGORICAL_COLS + NUMERICAL_COLS]
#     y = df[TARGET_COL]
#     return X, y


# def feature_names_out(preprocessor) -> list:
#     return CATEGORICAL_COLS + NUMERICAL_COLS


# # ── CLI entry-point (Azure ML pipeline step) ──────────────────────────────────
# if __name__ == "__main__":
#     import sys
#     import argparse
#     from pathlib import Path

#     print("=== preprocess.py starting ===", flush=True)
#     print(f"Python: {sys.version}", flush=True)
#     print(f"pandas: {pd.__version__}  numpy: {np.__version__}", flush=True)

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input",  required=True,
#                         help="URI_FILE (csv) or URI_FOLDER containing a csv")
#     parser.add_argument("--output", required=True,
#                         help="Output folder — cleaned_data.csv will be written here")
#     args = parser.parse_args()

#     print(f"--input  = {args.input}",  flush=True)
#     print(f"--output = {args.output}", flush=True)

#     # ── Resolve input to an actual CSV file ───────────────────────────────────
#     input_path = Path(args.input)
#     print(f"input_path exists={input_path.exists()} is_dir={input_path.is_dir()}", flush=True)

#     if input_path.is_dir():
#         print(f"Contents of input dir: {list(input_path.iterdir())}", flush=True)
#         csv_files = sorted(input_path.glob("*.csv"))
#         if not csv_files:
#             print(f"ERROR: No CSV files found in {input_path}", flush=True)
#             sys.exit(1)
#         input_path = csv_files[0]
#         print(f"Using CSV from folder: {input_path}", flush=True)
#     elif not input_path.exists():
#         print(f"ERROR: Input path does not exist: {input_path}", flush=True)
#         sys.exit(1)

#     # ── Load, clean, save ─────────────────────────────────────────────────────
#     df = load_data(str(input_path))
#     df = clean_data(df)

#     output_path = Path(args.output)
#     output_path.mkdir(parents=True, exist_ok=True)

#     out_file = output_path / "cleaned_data.csv"
#     df.to_csv(out_file, index=False)
#     print(f"Saved -> {out_file}  ({len(df)} rows)", flush=True)
#     print("=== preprocess.py done ===", flush=True)

"""
preprocess.py
-------------
Shared preprocessing logic — used by both the Azure ML pipeline step
and the local training script.

As a script (Azure ML step):
    python preprocess.py --input <csv_file_or_folder> --output <folder>

As a module (imported by train.py):
    from preprocess import load_data, clean_data, build_preprocessor, ...
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


# ── Column groups ─────────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "store_id", "day_of_week", "weather_condition",
    "traffic_level", "vehicle_type",
]
NUMERICAL_COLS = [
    "order_hour", "is_weekend", "is_peak_hour",
    "distance_km", "num_items", "prep_time_seconds",
]
TARGET_COL = "edt_seconds"


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows  cols={list(df.columns)}", flush=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    initial = len(df)
    df = df.drop_duplicates(subset="order_id")
    df = df.dropna(subset=[TARGET_COL])
    df[NUMERICAL_COLS] = df[NUMERICAL_COLS].fillna(df[NUMERICAL_COLS].median())
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].fillna("unknown")
    print(f"Cleaned: {initial} -> {len(df)} rows", flush=True)
    return df


def build_preprocessor() -> ColumnTransformer:
    cat_pipeline = Pipeline([
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ))
    ])
    num_pipeline = Pipeline([("scaler", StandardScaler())])
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, CATEGORICAL_COLS),
            ("num", num_pipeline, NUMERICAL_COLS),
        ],
        remainder="drop",
    )


def get_features_and_target(df: pd.DataFrame):
    X = df[CATEGORICAL_COLS + NUMERICAL_COLS]
    y = df[TARGET_COL]
    return X, y


def feature_names_out(preprocessor) -> list:
    return CATEGORICAL_COLS + NUMERICAL_COLS


# ── CLI entry-point (Azure ML pipeline step) ──────────────────────────────────
if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path

    print("=== preprocess.py starting ===", flush=True)
    print(f"Python: {sys.version}", flush=True)
    print(f"pandas: {pd.__version__}  numpy: {np.__version__}", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="URI_FILE (csv) or URI_FOLDER containing a csv")
    parser.add_argument("--output", required=True,
                        help="Output folder — cleaned_data.csv will be written here")
    args = parser.parse_args()

    print(f"--input  = {args.input}", flush=True)
    print(f"--output = {args.output}", flush=True)

    # ── Resolve input to an actual CSV file ───────────────────────────────────
    input_path = Path(args.input)
    print(f"input_path exists={input_path.exists()} is_dir={input_path.is_dir()}", flush=True)

    if input_path.is_dir():
        print(f"Contents of input dir: {list(input_path.iterdir())}", flush=True)
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            print(f"ERROR: No CSV files found in {input_path}", flush=True)
            sys.exit(1)
        input_path = csv_files[0]
        print(f"Using CSV from folder: {input_path}", flush=True)
    elif not input_path.exists():
        print(f"ERROR: Input path does not exist: {input_path}", flush=True)
        sys.exit(1)

    # ── Load, clean, save ─────────────────────────────────────────────────────
    df = load_data(str(input_path))
    df = clean_data(df)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    out_file = output_path / "cleaned_data.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved -> {out_file}  ({len(df)} rows)", flush=True)
    print("=== preprocess.py done ===", flush=True)
