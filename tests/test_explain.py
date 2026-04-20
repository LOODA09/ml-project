import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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


def test_background_loader_falls_back_to_raw_dataset_when_snapshot_is_missing(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    artifacts_dir = tmp_path / "artifacts"
    raw_dir.mkdir()
    processed_dir.mkdir()
    artifacts_dir.mkdir()

    pd.DataFrame(
        {
            "booking_status": ["Canceled", "Not_Canceled"],
            "adults": [2, 1],
            "children": [0, 1],
            "lead_time": [25, 5],
            "arrival_date_month": ["July", "August"],
            "stays_in_weekend_nights": [1, 0],
            "stays_in_week_nights": [2, 3],
            "meal": ["Meal Plan 1", "Meal Plan 2"],
            "market_segment": ["Online", "Offline"],
            "deposit_type": ["Refundable", "No Deposit"],
            "is_repeated_guest": [0, 1],
            "required_car_parking_spaces": [0, 1],
            "adr": [120.0, 90.0],
            "total_of_special_requests": [1, 2],
            "previous_bookings_not_canceled": [0, 3],
            "previous_cancellations": [2, 0],
            "reserved_room_type": ["Room_Type 1", "Room_Type 2"],
        }
    ).to_csv(raw_dir / "hotels.csv", index=False)

    monkeypatch.setattr(explain.CONFIG, "artifacts_dir", artifacts_dir, raising=False)
    monkeypatch.setattr(explain.CONFIG, "processed_dir", processed_dir, raising=False)

    background = explain._load_background_frame(
        metadata={
            "feature_columns": [
                "lead_time",
                "deposit_type",
                "avg_price_per_room",
                "has_refundable_deposit",
                "historical_cancellation_ratio",
            ],
            "categorical_columns": ["deposit_type"],
        }
    )

    assert background is not None
    assert not background.empty
    assert list(background.columns) == [
        "lead_time",
        "deposit_type",
        "avg_price_per_room",
        "has_refundable_deposit",
        "historical_cancellation_ratio",
    ]


def test_tree_explanation_returns_fast_summary_without_background(monkeypatch):
    x_train = pd.DataFrame(
        {
            "lead_time": [8, 14, 22, 30, 36, 44, 60, 75],
            "avg_price_per_room": [70.0, 82.0, 95.0, 120.0, 135.0, 160.0, 175.0, 190.0],
            "deposit_type": ["No Deposit", "No Deposit", "Refundable", "Refundable", "Non Refund", "Non Refund", "Refundable", "Non Refund"],
        }
    )
    y_train = [0, 0, 0, 0, 1, 1, 1, 1]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["lead_time", "avg_price_per_room"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["deposit_type"]),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", RandomForestClassifier(n_estimators=20, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)

    prepared_row = pd.DataFrame(
        [
            {
                "lead_time": 40,
                "avg_price_per_room": 140.0,
                "deposit_type": "Refundable",
            }
        ]
    )

    monkeypatch.setattr(explain, "_load_background_frame", lambda metadata: None)

    summary = explain.explain_single_prediction(model, prepared_row, metadata={"feature_columns": list(prepared_row.columns)})

    assert summary is not None
    assert summary.method.startswith("Tree SHAP")
    assert not summary.top_contributions.empty
