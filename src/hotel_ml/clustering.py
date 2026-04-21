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
    total_rows = max(int(summary["cluster_size"].sum()), 1)
    summary["cluster_share"] = (summary["cluster_size"] / total_rows).round(4)
    raw_names = [_describe_cluster(summary.loc[cluster_id]) for cluster_id in summary.index]
    summary["segment_name"] = _make_unique_segment_names(summary, raw_names)
    ordered_columns = ["segment_name", "cluster_size", "cluster_share"] + [
        column for column in summary.columns if column not in {"segment_name", "cluster_size", "cluster_share"}
    ]
    return summary[ordered_columns]


def _describe_cluster(row: pd.Series) -> str:
    lead_time = float(row.get("lead_time", 0))
    avg_price = float(row.get("avg_price_per_room", 0))
    total_guests = float(row.get("total_guests", 0))
    total_nights = float(row.get("total_nights", 0))
    booking_changes = float(row.get("booking_changes", 0))
    waiting_list = float(row.get("days_in_waiting_list", 0))
    repeated_guest = float(row.get("repeated_guest", 0))
    refundable = float(row.get("has_refundable_deposit", 0))
    non_refundable = float(row.get("has_non_refundable_deposit", 0))
    previous_non_canceled = float(row.get("no_of_previous_bookings_not_canceled", 0))
    previous_cancellations = float(row.get("no_of_previous_cancellations", 0))
    cancellation_ratio = float(row.get("historical_cancellation_ratio", 0))
    parking = float(row.get("required_car_parking_space", 0))
    special_requests = float(row.get("no_of_special_requests", 0))

    if (previous_non_canceled >= 0.5 or repeated_guest >= 0.35) and special_requests >= 0.7:
        return "Loyal Planned Guests"
    if cancellation_ratio >= 0.35 or previous_cancellations >= 0.3:
        return "Repeat Cancellation Risk"
    if refundable >= 0.35 and (booking_changes >= 0.6 or waiting_list >= 5):
        return "Flexible Change-Prone Guests"
    if non_refundable >= 0.35 and special_requests >= 0.7:
        return "Committed Rate Guests"
    if lead_time >= 90 and special_requests <= 0.7:
        return "Early Uncertain Bookers"
    if avg_price >= 120 and total_guests >= 2.2:
        return "Premium Group Stays"
    if total_nights >= 4:
        return "Extended Stay Guests"
    if parking >= 0.5:
        return "Drive-In Practical Guests"
    return "Core Standard Bookers"


def _make_unique_segment_names(summary: pd.DataFrame, raw_names: list[str]) -> list[str]:
    name_counts: dict[str, int] = {}
    unique_names: list[str] = []

    for cluster_id, base_name in zip(summary.index, raw_names):
        seen = name_counts.get(base_name, 0)
        name_counts[base_name] = seen + 1
        if raw_names.count(base_name) == 1:
            unique_names.append(base_name)
            continue

        qualifier = _segment_qualifier(summary.loc[cluster_id], fallback=str(cluster_id))
        unique_names.append(f"{base_name} ({qualifier})")

    return unique_names


def _segment_qualifier(row: pd.Series, fallback: str) -> str:
    if float(row.get("repeated_guest", 0)) >= 0.5:
        return "repeat-heavy"
    if float(row.get("has_non_refundable_deposit", 0)) >= 0.4:
        return "non-refundable"
    if float(row.get("has_refundable_deposit", 0)) >= 0.4:
        return "refundable"
    if float(row.get("required_car_parking_space", 0)) >= 0.5:
        return "parking-led"
    if float(row.get("lead_time", 0)) >= 120:
        return "long-lead"
    return f"cluster-{fallback}"


def save_cluster_artifacts(model: Pipeline, profile: pd.DataFrame, diagnostics: dict) -> None:
    joblib.dump(model, CONFIG.artifacts_dir / "guest_segmentation.joblib")
    profile.to_csv(CONFIG.artifacts_dir / "cluster_profiles.csv")
    with open(CONFIG.artifacts_dir / "cluster_diagnostics.json", "w", encoding="utf-8") as file:
        json.dump(diagnostics, file, indent=2)
