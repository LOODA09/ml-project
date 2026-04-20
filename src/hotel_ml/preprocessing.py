from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
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


class RegularizedTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str] | None = None, min_samples_leaf: int = 20, smoothing: float = 10.0):
        self.columns = columns
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing

    def fit(self, x, y):
        x_frame = pd.DataFrame(x).copy()
        if self.columns is not None:
            x_frame.columns = list(self.columns)
        self.feature_names_in_ = x_frame.columns.astype(str).tolist()
        self.encoder_ = TargetEncoder(
            cols=self.feature_names_in_,
            min_samples_leaf=self.min_samples_leaf,
            smoothing=self.smoothing,
            handle_missing="value",
            handle_unknown="value",
            return_df=True,
        )
        self.encoder_.fit(x_frame, y)
        return self

    def transform(self, x):
        x_frame = pd.DataFrame(x).copy()
        x_frame.columns = self.feature_names_in_
        transformed = self.encoder_.transform(x_frame)
        return transformed.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray(list(input_features), dtype=object)


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
        self.high_cardinality_keep_values_: dict[str, set[str]] = {}
        rare_threshold = max(
            CONFIG.rare_category_min_count,
            int(len(x_frame) * CONFIG.rare_category_min_frequency_ratio),
        )
        for column in self.categorical_columns_:
            series = x_frame[column].astype("string")
            mode = series.mode(dropna=True)
            fill_value = str(mode.iloc[0]) if not mode.empty else "Unknown"
            cleaned = series.fillna(fill_value).astype(str).replace({"": fill_value})
            if column in self.schema.high_cardinality_columns:
                counts = cleaned.value_counts()
                keep_values = set(counts[counts >= rare_threshold].index.tolist())
                keep_values.add("Unknown")
                cleaned = cleaned.where(cleaned.isin(keep_values), "Other")
                self.high_cardinality_keep_values_[column] = keep_values
            categories = sorted(cleaned.unique().tolist())
            if fill_value not in categories:
                categories.append(fill_value)
            if "Unknown" not in categories:
                categories.append("Unknown")
            if column in self.schema.high_cardinality_columns and "Other" not in categories:
                categories.append("Other")
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
            if column in self.schema.high_cardinality_columns:
                keep_values = self.high_cardinality_keep_values_.get(column, set())
                series = series.where(series.isin(keep_values), "Other")
            else:
                series = series.where(series.isin(categories), "Unknown")
            x_frame[column] = pd.Categorical(series, categories=categories)

        return x_frame


def infer_feature_schema(x: pd.DataFrame) -> FeatureSchema:
    numeric_columns = x.select_dtypes(
        exclude=["object", "string", "category"]
    ).columns.tolist()
    categorical_columns = x.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_columns = [column for column in numeric_columns if column not in categorical_columns]

    low_cardinality_columns: list[str] = []
    high_cardinality_columns: list[str] = []
    for column in categorical_columns:
        cardinality = x[column].astype("string").nunique(dropna=True)
        if cardinality > CONFIG.low_cardinality_threshold:
            high_cardinality_columns.append(column)
        else:
            low_cardinality_columns.append(column)

    return FeatureSchema(
        numeric_columns=numeric_columns,
        low_cardinality_columns=low_cardinality_columns,
        high_cardinality_columns=high_cardinality_columns,
    )


def _is_tree_based_model(model) -> bool:
    return model.__class__.__name__ in {"DecisionTreeClassifier", "RandomForestClassifier"}


def build_preprocessor(schema: FeatureSchema, model=None) -> ColumnTransformer:
    if _is_tree_based_model(model):
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
    else:
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
    transformers: list[tuple[str, object, list[str]]] = [
        ("num", numeric_pipeline, schema.numeric_columns),
        ("low_cat", low_cardinality_pipeline, schema.low_cardinality_columns),
    ]
    if schema.high_cardinality_columns:
        transformers.append(
            (
                "high_cat",
                Pipeline(
                    steps=[
                        (
                            "target",
                            RegularizedTargetEncoder(
                                columns=schema.high_cardinality_columns,
                                min_samples_leaf=CONFIG.target_encoder_min_samples_leaf,
                                smoothing=CONFIG.target_encoder_smoothing,
                            ),
                        ),
                    ]
                ),
                schema.high_cardinality_columns,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
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
            ("preprocess", build_preprocessor(schema, model)),
            ("model", model),
        ]
    )

    return ImbPipeline(steps=steps)
