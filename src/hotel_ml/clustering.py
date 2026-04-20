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
    return profiled.groupby("cluster").mean(numeric_only=True).round(2)


def save_cluster_artifacts(model: Pipeline, profile: pd.DataFrame, diagnostics: dict) -> None:
    joblib.dump(model, CONFIG.artifacts_dir / "guest_segmentation.joblib")
    profile.to_csv(CONFIG.artifacts_dir / "cluster_profiles.csv")
    with open(CONFIG.artifacts_dir / "cluster_diagnostics.json", "w", encoding="utf-8") as file:
        json.dump(diagnostics, file, indent=2)
