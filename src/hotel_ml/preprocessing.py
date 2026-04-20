from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CONFIG


@dataclass
class FeatureSchema:
    numeric_columns: list[str]
    low_cardinality_columns: list[str]
    high_cardinality_columns: list[str]


class RawFeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, schema: FeatureSchema):
        self.schema = schema

    def fit(self, x: pd.DataFrame, y=None):
        x_frame = x.copy()
        self.numeric_columns_ = list(self.schema.numeric_columns)
        self.categorical_columns_ = list(self.schema.low_cardinality_columns) + list(
            self.schema.high_cardinality_columns
        )

        self.numeric_fill_values_: dict[str, float] = {}
        for column in self.numeric_columns_:
            series = pd.to_numeric(x_frame[column], errors="coerce")
            median_value = series.median()
            self.numeric_fill_values_[column] = float(median_value) if pd.notna(median_value) else 0.0

        self.categorical_fill_values_: dict[str, str] = {}
        self.categorical_categories_: dict[str, list[str]] = {}
        for column in self.categorical_columns_:
            series = x_frame[column].astype("string")
            mode = series.mode(dropna=True)
            fill_value = str(mode.iloc[0]) if not mode.empty else "Unknown"
            cleaned = series.fillna(fill_value).astype(str).replace({"": fill_value})
            categories = sorted(cleaned.unique().tolist())
            if fill_value not in categories:
                categories.append(fill_value)
            if "Unknown" not in categories:
                categories.append("Unknown")
            self.categorical_fill_values_[column] = fill_value
            self.categorical_categories_[column] = categories

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x_frame = x.copy()

        for column in self.numeric_columns_:
            series = pd.to_numeric(x_frame[column], errors="coerce")
            x_frame[column] = series.fillna(self.numeric_fill_values_[column]).astype(float)

        for column in self.categorical_columns_:
            categories = self.categorical_categories_[column]
            fill_value = self.categorical_fill_values_[column]
            series = x_frame[column].astype("string").fillna(fill_value).astype(str)
            series = series.where(series.isin(categories), "Unknown")
            x_frame[column] = pd.Categorical(series, categories=categories)

        return x_frame


def infer_feature_schema(x: pd.DataFrame) -> FeatureSchema:
    numeric_columns = x.select_dtypes(
        exclude=["object", "string", "category"]
    ).columns.tolist()
    categorical_columns = x.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_columns = [column for column in numeric_columns if column not in categorical_columns]

    low_cardinality_columns = list(categorical_columns)
    high_cardinality_columns: list[str] = []

    return FeatureSchema(
        numeric_columns=numeric_columns,
        low_cardinality_columns=low_cardinality_columns,
        high_cardinality_columns=high_cardinality_columns,
    )


def build_preprocessor(schema: FeatureSchema) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    low_cardinality_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, schema.numeric_columns),
            ("low_cat", low_cardinality_pipeline, schema.low_cardinality_columns),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_sampler(schema: FeatureSchema) -> SMOTENC:
    _ = schema
    return SMOTENC(
        categorical_features="auto",
        sampling_strategy=CONFIG.smote_sampling_ratio,
        random_state=CONFIG.random_state,
        k_neighbors=CONFIG.smote_k_neighbors,
    )


def build_training_pipeline(model, schema: FeatureSchema) -> ImbPipeline:
    steps: list[tuple[str, object]] = [
        ("raw_prepare", RawFeaturePreprocessor(schema)),
    ]

    if CONFIG.smote_enabled:
        steps.append(("smote", build_sampler(schema)))

    steps.extend(
        [
            ("preprocess", build_preprocessor(schema)),
            ("model", model),
        ]
    )

    return ImbPipeline(steps=steps)
