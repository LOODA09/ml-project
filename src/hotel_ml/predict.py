from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .config import CONFIG
from .features import add_engineered_features


def load_best_model():
    return joblib.load(CONFIG.artifacts_dir / "best_cancellation_model.joblib")


def load_raw_model():
    raw_path = CONFIG.artifacts_dir / "best_cancellation_model_raw.joblib"
    if raw_path.exists():
        return joblib.load(raw_path)
    return load_best_model()


def load_metadata() -> dict:
    path = CONFIG.artifacts_dir / "training_metadata.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def prepare_single_input(record: dict) -> pd.DataFrame:
    df = pd.DataFrame([record])
    return add_engineered_features(df)


def infer_training_columns_from_model(model) -> tuple[list[str], set[str]]:
    preprocess = _extract_preprocessor(model)
    if preprocess is None or not hasattr(preprocess, "transformers_"):
        return [], set()

    feature_columns: list[str] = []
    categorical_columns: set[str] = set()

    for name, _, columns in preprocess.transformers_:
        if name == "remainder":
            continue
        if isinstance(columns, list):
            feature_columns.extend(columns)
            if name in {"low_cat", "high_cat"}:
                categorical_columns.update(columns)

    return feature_columns, categorical_columns


def _extract_preprocessor(model):
    if hasattr(model, "named_steps"):
        return model.named_steps.get("preprocess")

    estimator = getattr(model, "estimator", None)
    if estimator is not None:
        preprocess = _extract_preprocessor(estimator)
        if preprocess is not None:
            return preprocess

    calibrated_estimators = getattr(model, "calibrated_classifiers_", None)
    if calibrated_estimators:
        first = calibrated_estimators[0]
        inner_estimator = getattr(first, "estimator", None)
        if inner_estimator is not None:
            preprocess = _extract_preprocessor(inner_estimator)
            if preprocess is not None:
                return preprocess

    return None


def align_to_training_features(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    feature_columns = metadata.get("feature_columns", [])
    categorical_columns = set(metadata.get("categorical_columns", []))
    aligned = df.copy()

    if not feature_columns:
        return aligned

    for column in feature_columns:
        if column not in aligned.columns:
            aligned[column] = "Unknown" if column in categorical_columns else 0

    return aligned.reindex(columns=feature_columns)


def align_to_model_schema(df: pd.DataFrame, model, metadata: dict | None = None) -> pd.DataFrame:
    metadata = metadata or {}
    feature_columns = metadata.get("feature_columns", [])
    categorical_columns = set(metadata.get("categorical_columns", []))

    if not feature_columns:
        feature_columns, categorical_columns = infer_training_columns_from_model(model)

    return align_to_training_features(
        df,
        {
            "feature_columns": feature_columns,
            "categorical_columns": list(categorical_columns),
        },
    )


def predict_booking(record: dict) -> dict:
    model = load_best_model()
    metadata = load_metadata()
    prepared = align_to_model_schema(prepare_single_input(record), model, metadata)
    probability = float(model.predict_proba(prepared)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "cancellation_probability": probability,
    }
