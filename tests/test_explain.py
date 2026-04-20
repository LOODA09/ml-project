import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.hotel_ml import explain


def test_local_effect_fallback_returns_summary_when_shap_is_disabled(monkeypatch):
    x_train = pd.DataFrame(
        {
            "lead_time": [8, 14, 22, 30, 36, 44],
            "avg_price_per_room": [70.0, 82.0, 95.0, 120.0, 135.0, 160.0],
            "deposit_type": ["No Deposit", "No Deposit", "Refundable", "Refundable", "Non Refund", "Non Refund"],
        }
    )
    y_train = [0, 0, 0, 1, 1, 1]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["lead_time", "avg_price_per_room"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["deposit_type"]),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    model.fit(x_train, y_train)

    prepared_row = pd.DataFrame(
        [
            {
                "lead_time": 18,
                "avg_price_per_room": 110.0,
                "deposit_type": "Refundable",
            }
        ]
    )

    monkeypatch.setattr(explain, "SHAP_AVAILABLE", False)
    monkeypatch.setattr(explain, "_load_background_frame", lambda metadata: x_train.copy())

    summary = explain.explain_single_prediction(model, prepared_row, metadata={"feature_columns": list(prepared_row.columns)})

    assert summary is not None
    assert summary.method == "Local Effect Fallback"
    assert not summary.top_contributions.empty


def test_local_effect_fallback_handles_integer_columns_without_dtype_errors(monkeypatch):
    x_train = pd.DataFrame(
        {
            "lead_time": [10, 11, 12, 13],
            "no_of_previous_cancellations": [0, 1, 2, 3],
            "deposit_type": ["No Deposit", "Refundable", "Refundable", "Non Refund"],
        }
    )
    y_train = [0, 0, 1, 1]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["lead_time", "no_of_previous_cancellations"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["deposit_type"]),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    model.fit(x_train, y_train)

    prepared_row = pd.DataFrame(
        [
            {
                "lead_time": 12,
                "no_of_previous_cancellations": 1,
                "deposit_type": "Refundable",
            }
        ]
    )

    monkeypatch.setattr(explain, "SHAP_AVAILABLE", False)
    monkeypatch.setattr(explain, "_load_background_frame", lambda metadata: x_train.copy())

    summary = explain.explain_single_prediction(model, prepared_row, metadata={"feature_columns": list(prepared_row.columns)})

    assert summary is not None
    assert summary.method == "Local Effect Fallback"
    assert "no_of_previous_cancellations" in summary.top_contributions["feature"].values
