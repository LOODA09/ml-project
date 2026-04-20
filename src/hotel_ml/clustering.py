from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CONFIG


def find_best_kmeans(x: pd.DataFrame) -> tuple[Pipeline, dict]:
    best_score = -1.0
    best_model = None
    diagnostics: list[dict] = []

    for k in range(CONFIG.clustering_k_min, CONFIG.clustering_k_max + 1):
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("kmeans", KMeans(n_clusters=k, random_state=CONFIG.random_state, n_init=10)),
            ]
        )
        labels = model.fit_predict(x)
        score = silhouette_score(StandardScaler().fit_transform(x), labels)
        diagnostics.append({"k": k, "silhouette_score": score})

        if score > best_score:
            best_score = score
            best_model = model

    assert best_model is not None
    return best_model, {"best_score": best_score, "diagnostics": diagnostics}


def profile_clusters(features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    profiled = features.copy()
    profiled["cluster"] = labels
    summary = profiled.groupby("cluster").mean(numeric_only=True).round(2)
    summary["cluster_size"] = profiled.groupby("cluster").size()
    summary["segment_name"] = [
        _describe_cluster(summary.loc[cluster_id]) for cluster_id in summary.index
    ]
    ordered_columns = ["segment_name", "cluster_size"] + [
        column for column in summary.columns if column not in {"segment_name", "cluster_size"}
    ]
    return summary[ordered_columns]


def _describe_cluster(row: pd.Series) -> str:
    lead_time = float(row.get("lead_time", 0))
    avg_price = float(row.get("avg_price_per_room", 0))
    total_guests = float(row.get("total_guests", 0))
    total_nights = float(row.get("total_nights", 0))
    previous_non_canceled = float(row.get("no_of_previous_bookings_not_canceled", 0))
    parking = float(row.get("required_car_parking_space", 0))
    special_requests = float(row.get("no_of_special_requests", 0))

    if previous_non_canceled >= 0.5 or parking >= 0.5:
        return "Loyal Planned Guests"
    if lead_time >= 90 and special_requests <= 0.7:
        return "Early Uncertain Bookers"
    if avg_price >= 120 and total_guests >= 2.2:
        return "Premium Group Stays"
    if total_nights >= 4:
        return "Extended Stay Guests"
    return "Core Standard Bookers"


def save_cluster_artifacts(model: Pipeline, profile: pd.DataFrame, diagnostics: dict) -> None:
    joblib.dump(model, CONFIG.artifacts_dir / "guest_segmentation.joblib")
    profile.to_csv(CONFIG.artifacts_dir / "cluster_profiles.csv")
    with open(CONFIG.artifacts_dir / "cluster_diagnostics.json", "w", encoding="utf-8") as file:
        json.dump(diagnostics, file, indent=2)
