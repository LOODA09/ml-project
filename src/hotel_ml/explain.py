from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from .config import CONFIG


TREE_MODEL_CLASSNAMES = {
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "XGBClassifier",
}


@dataclass
class ShapSummary:
    explanation: shap.Explanation | None
    top_contributions: pd.DataFrame
    method: str


def _is_tree_model(estimator) -> bool:
    return estimator.__class__.__name__ in TREE_MODEL_CLASSNAMES


def _transform_with_feature_names(model, x: pd.DataFrame) -> pd.DataFrame:
    raw_prepare = model.named_steps.get("raw_prepare")
    if raw_prepare is not None:
        x = raw_prepare.transform(x)
    preprocess = model.named_steps["preprocess"]
    transformed = preprocess.transform(x)
    feature_names = preprocess.get_feature_names_out().tolist()
    return pd.DataFrame(transformed, columns=feature_names)


def _load_background_frame(metadata: dict) -> pd.DataFrame | None:
    path = CONFIG.artifacts_dir / "training_input_snapshot.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    feature_columns = metadata.get("feature_columns", [])
    if feature_columns:
        df = df.reindex(columns=feature_columns)

    if len(df) > 200:
        df = df.sample(n=200, random_state=CONFIG.random_state)
    return df


def _build_tree_summary(estimator, transformed_row: pd.DataFrame, transformed_background: pd.DataFrame) -> ShapSummary | None:
    explainer = shap.TreeExplainer(
        estimator,
        data=transformed_background,
        model_output="probability",
        feature_perturbation="interventional",
    )
    shap_values = explainer(transformed_row)

    values = shap_values.values
    base_values = shap_values.base_values

    if values.ndim == 3:
        row_values = values[0, :, 1]
        row_base = float(np.asarray(base_values)[0, 1])
    else:
        row_values = values[0]
        row_base = float(np.asarray(base_values)[0])

    return _finalize_summary(transformed_row, row_values, row_base, method="Tree SHAP")


def _build_model_agnostic_summary(estimator, transformed_row: pd.DataFrame, transformed_background: pd.DataFrame) -> ShapSummary | None:
    background_array = transformed_background.to_numpy()
    row_array = transformed_row.to_numpy()

    background_size = min(len(background_array), 40)
    if len(background_array) > background_size:
        background_array = shap.sample(background_array, background_size, random_state=CONFIG.random_state)

    explainer = shap.KernelExplainer(estimator.predict_proba, background_array)
    shap_values = explainer.shap_values(
        row_array,
        nsamples=min(120, max(40, transformed_row.shape[1] * 2 + 1)),
    )

    if isinstance(shap_values, list):
        row_values = np.asarray(shap_values[1])[0]
        expected_value = explainer.expected_value[1]
    else:
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            row_values = shap_array[0, :, 1]
        else:
            row_values = shap_array[0]
        expected = np.asarray(explainer.expected_value)
        expected_value = expected[1] if expected.ndim > 0 and len(expected) > 1 else expected.item()

    return _finalize_summary(transformed_row, row_values, float(expected_value), method="Kernel SHAP")


def _build_local_effect_fallback(model, prepared_row: pd.DataFrame, metadata: dict) -> ShapSummary | None:
    background_raw = _load_background_frame(metadata)
    if background_raw is None:
        return None

    base_probability = float(model.predict_proba(prepared_row)[0, 1])
    contributions: list[dict] = []
    for column in prepared_row.columns:
        candidate = prepared_row.copy()
        series = background_raw[column].dropna() if column in background_raw.columns else pd.Series(dtype=float)
        if series.empty:
            baseline_value = prepared_row.iloc[0][column]
        elif pd.api.types.is_numeric_dtype(series):
            baseline_value = float(series.median())
        else:
            mode = series.astype(str).mode(dropna=True)
            baseline_value = str(mode.iloc[0]) if not mode.empty else str(prepared_row.iloc[0][column])
        candidate.at[candidate.index[0], column] = baseline_value
        fallback_probability = float(model.predict_proba(candidate)[0, 1])
        contributions.append(
            {
                "feature": column,
                "feature_value": prepared_row.iloc[0][column],
                "shap_value": base_probability - fallback_probability,
            }
        )

    contribution_df = pd.DataFrame(contributions)
    contribution_df["abs_shap_value"] = contribution_df["shap_value"].abs()
    contribution_df = contribution_df.sort_values("abs_shap_value", ascending=False).head(12)
    return ShapSummary(
        explanation=None,
        top_contributions=contribution_df,
        method="Local Effect Fallback",
    )


def _finalize_summary(transformed_row: pd.DataFrame, row_values: np.ndarray, row_base: float, method: str) -> ShapSummary:
    explanation = shap.Explanation(
        values=row_values,
        base_values=row_base,
        data=transformed_row.iloc[0].to_numpy(),
        feature_names=transformed_row.columns.tolist(),
    )

    contributions = pd.DataFrame(
        {
            "feature": transformed_row.columns,
            "feature_value": transformed_row.iloc[0].values,
            "shap_value": row_values,
        }
    )
    contributions["abs_shap_value"] = contributions["shap_value"].abs()
    contributions = contributions.sort_values(
        by="abs_shap_value", ascending=False
    ).head(12)

    return ShapSummary(
        explanation=explanation,
        top_contributions=contributions,
        method=method,
    )


def explain_single_prediction(model, prepared_row: pd.DataFrame, metadata: dict) -> ShapSummary | None:
    estimator = model.named_steps.get("model")
    if estimator is None:
        return None

    try:
        if not SHAP_AVAILABLE:
            return _build_local_effect_fallback(model, prepared_row, metadata)
        transformed_row = _transform_with_feature_names(model, prepared_row)
        background_raw = _load_background_frame(metadata)
        if background_raw is None:
            return _build_local_effect_fallback(model, prepared_row, metadata)
        transformed_background = _transform_with_feature_names(model, background_raw)
        if _is_tree_model(estimator):
            return _build_tree_summary(estimator, transformed_row, transformed_background)
        return _build_model_agnostic_summary(estimator, transformed_row, transformed_background)
    except Exception:
        return _build_local_effect_fallback(model, prepared_row, metadata)
