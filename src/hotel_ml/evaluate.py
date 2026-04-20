from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import CONFIG


@dataclass
class EvaluationResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    log_loss: float
    brier_score: float
    train_time_seconds: float
    inference_time_ms_per_100: float
    complexity: str


def evaluate_model(
    name: str,
    pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    complexity: str,
) -> tuple[EvaluationResult, object]:
    start = perf_counter()
    pipeline.fit(x_train, y_train)
    train_time = perf_counter() - start

    inference_batch = x_test.head(min(100, len(x_test)))
    start = perf_counter()
    preds = pipeline.predict(inference_batch)
    inference_time = (perf_counter() - start) * 1000

    y_pred = pipeline.predict(x_test)
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(x_test)[:, 1]
    else:
        y_proba = y_pred
    y_proba = y_proba.clip(1e-6, 1 - 1e-6)

    result = EvaluationResult(
        model_name=name,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1_score=f1_score(y_test, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, y_proba),
        log_loss=log_loss(y_test, y_proba),
        brier_score=brier_score_loss(y_test, y_proba),
        train_time_seconds=train_time,
        inference_time_ms_per_100=inference_time,
        complexity=complexity,
    )
    return result, pipeline


def cross_validation_overview(pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=CONFIG.cv_folds, shuffle=True, random_state=CONFIG.random_state)
    scores = cross_validate(
        pipeline,
        x_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1"],
        n_jobs=1,
        error_score="raise",
    )
    return {
        "cv_accuracy": float(scores["test_accuracy"].mean()),
        "cv_precision": float(scores["test_precision"].mean()),
        "cv_recall": float(scores["test_recall"].mean()),
        "cv_f1": float(scores["test_f1"].mean()),
    }


def results_to_dataframe(results: list[EvaluationResult]) -> pd.DataFrame:
    data = [
        {
            "model_name": result.model_name,
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1_score": result.f1_score,
            "roc_auc": result.roc_auc,
            "log_loss": result.log_loss,
            "brier_score": result.brier_score,
            "train_time_seconds": result.train_time_seconds,
            "inference_time_ms_per_100": result.inference_time_ms_per_100,
            "complexity": result.complexity,
        }
        for result in results
    ]
    df = pd.DataFrame(data)

    scoring_plan = {
        "accuracy": (0.14, True),
        "precision": (0.10, True),
        "recall": (0.10, True),
        "f1_score": (0.22, True),
        "roc_auc": (0.24, True),
        "log_loss": (0.08, False),
        "brier_score": (0.07, False),
        "train_time_seconds": (0.02, False),
        "inference_time_ms_per_100": (0.03, False),
    }

    composite_score = pd.Series(0.0, index=df.index, dtype=float)
    for column, (weight, higher_is_better) in scoring_plan.items():
        series = df[column].astype(float)
        span = float(series.max() - series.min())
        if span == 0:
            normalized = pd.Series(1.0, index=df.index, dtype=float)
        else:
            normalized = (series - series.min()) / span
            if not higher_is_better:
                normalized = 1.0 - normalized
        composite_score += normalized * weight

    df["composite_score"] = composite_score.round(6)
    return df.sort_values(
        by=["composite_score", "roc_auc", "f1_score", "accuracy"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def build_confusion_payload(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total": int(total),
        "accuracy": float((tp + tn) / total) if total else 0.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
    }
