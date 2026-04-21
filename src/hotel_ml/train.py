from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split

from .clustering import find_best_kmeans, profile_clusters, save_cluster_artifacts
from .config import CONFIG, ensure_directories
from .data import basic_cleaning, load_dataset, resolve_dataset_path, select_model_features, split_features_target
from .evaluate import build_confusion_payload, cross_validation_overview, evaluate_model, results_to_dataframe
from .features import add_engineered_features, get_segmentation_features
from .models import TENSORFLOW_AVAILABLE, get_model_specs
from .preprocessing import build_training_pipeline, infer_feature_schema


def _status_path() -> Path:
    return CONFIG.artifacts_dir / "training_status.json"


def write_status(payload: dict) -> None:
    with open(_status_path(), "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def save_partial_results(results: list, trained_pipelines: dict[str, object]) -> None:
    if not results:
        return

    results_df = results_to_dataframe(results)
    results_df.to_csv(CONFIG.artifacts_dir / "model_comparison.csv", index=False)

    best_model_name = results_df.iloc[0]["model_name"]
    best_pipeline = trained_pipelines[best_model_name]
    joblib.dump(best_pipeline, CONFIG.artifacts_dir / "best_cancellation_model.joblib")
    print(
        f"[checkpoint] Saved partial results. Current best model: {best_model_name}",
        flush=True,
    )


def fit_full_pipeline(spec, schema, x: pd.DataFrame, y: pd.Series):
    pipeline = build_training_pipeline(spec.estimator, schema)
    pipeline.fit(x, y)
    return pipeline


def fit_calibrated_pipeline(spec, schema, x: pd.DataFrame, y: pd.Series):
    x_fit, x_calibration, y_fit, y_calibration = train_test_split(
        x,
        y,
        test_size=0.15,
        random_state=CONFIG.random_state,
        stratify=y,
    )
    raw_pipeline = fit_full_pipeline(spec, schema, x_fit, y_fit)
    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(raw_pipeline),
        method="sigmoid",
    )
    calibrated.fit(x_calibration, y_calibration)
    calibration_info = {
        "fit_rows": int(len(x_fit)),
        "calibration_rows": int(len(x_calibration)),
        "calibration_fraction": 0.15,
    }
    return raw_pipeline, calibrated, calibration_info


def bootstrap_deployment_artifacts(data_path: str | Path | None = None, force: bool = False) -> dict:
    ensure_directories()

    model_path = CONFIG.artifacts_dir / "best_cancellation_model.joblib"
    metadata_path = CONFIG.artifacts_dir / "training_metadata.json"
    if not force and model_path.exists() and metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as file:
            return json.load(file)

    resolved_data_path = resolve_dataset_path(data_path or (CONFIG.processed_dir.parent / "raw" / "hotels.csv"))
    df = load_dataset(resolved_data_path)
    df = basic_cleaning(df)
    df = add_engineered_features(df)

    x, y = split_features_target(df)
    x = select_model_features(x)
    segmentation_features = get_segmentation_features(df)
    bootstrap_rows = min(len(x), 1500)
    if bootstrap_rows < len(x):
        x_bootstrap, _, y_bootstrap, _ = train_test_split(
            x,
            y,
            train_size=bootstrap_rows,
            random_state=CONFIG.random_state,
            stratify=y,
        )
    else:
        x_bootstrap, y_bootstrap = x, y

    schema = infer_feature_schema(x_bootstrap)
    x_train, x_test, y_train, y_test = train_test_split(
        x_bootstrap,
        y_bootstrap,
        test_size=CONFIG.test_size,
        random_state=CONFIG.random_state,
        stratify=y_bootstrap,
    )

    model_specs = get_model_specs(include_svm=True)
    for spec in model_specs:
        if spec.name == "1D-CNN" and hasattr(spec.estimator, "epochs"):
            spec.estimator.epochs = min(int(spec.estimator.epochs), 8)
    evaluation_results = []
    for spec in model_specs:
        evaluation_pipeline = build_training_pipeline(spec.estimator, schema)
        result, _ = evaluate_model(
            spec.name,
            evaluation_pipeline,
            x_train,
            x_test,
            y_train,
            y_test,
            spec.complexity,
        )
        evaluation_results.append(result)

    results_df = results_to_dataframe(evaluation_results)
    benchmark_leader_name = str(results_df.iloc[0]["model_name"])
    specs_by_name = {spec.name: spec for spec in model_specs}
    chosen_spec = specs_by_name.get(benchmark_leader_name)
    if chosen_spec is None:
        raise RuntimeError(f"No deployment model available for bootstrap leader: {benchmark_leader_name}")

    deployed_pipeline = build_training_pipeline(chosen_spec.estimator, schema)
    deployed_pipeline.fit(x_bootstrap, y_bootstrap)
    results_df.to_csv(CONFIG.artifacts_dir / "model_comparison.csv", index=False)
    joblib.dump(deployed_pipeline, CONFIG.artifacts_dir / "best_cancellation_model.joblib")
    joblib.dump(deployed_pipeline, CONFIG.artifacts_dir / "best_cancellation_model_raw.joblib")

    segmentation_rows = min(len(segmentation_features), 10000)
    if segmentation_rows < len(segmentation_features):
        segmentation_sample = segmentation_features.sample(
            n=segmentation_rows,
            random_state=CONFIG.random_state,
        )
    else:
        segmentation_sample = segmentation_features
    clustering_model, diagnostics = find_best_kmeans(segmentation_sample)
    cluster_labels = clustering_model.predict(segmentation_features)
    cluster_profile = profile_clusters(segmentation_features, pd.Series(cluster_labels, index=segmentation_features.index))
    diagnostics.update(
        {
            "fit_rows": int(len(segmentation_sample)),
            "profile_rows": int(len(segmentation_features)),
            "bootstrap_fit_sample": bool(len(segmentation_sample) < len(segmentation_features)),
        }
    )
    save_cluster_artifacts(clustering_model, cluster_profile, diagnostics)

    metadata = {
        "target_column": CONFIG.target_column,
        "best_model_name": chosen_spec.name,
        "benchmark_leader_name": benchmark_leader_name,
        "prediction_model_artifact": "best_cancellation_model.joblib",
        "raw_model_artifact": "best_cancellation_model_raw.joblib",
        "smote_enabled": CONFIG.smote_enabled,
        "sampler": "SMOTENC" if CONFIG.smote_enabled else "None",
        "smote_sampling_ratio": CONFIG.smote_sampling_ratio if CONFIG.smote_enabled else None,
        "smote_k_neighbors": CONFIG.smote_k_neighbors if CONFIG.smote_enabled else None,
        "calibrated_model_available": False,
        "probability_strategy": "raw_predict_proba",
        "probability_calibration": {},
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "feature_columns": x.columns.tolist(),
        "categorical_columns": x.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist(),
        "cv_summary": {},
        "testing_phase": {
            "method": "holdout_test_split",
            "test_size": CONFIG.test_size,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "bootstrap_training_rows": int(len(x_bootstrap)),
        },
        "segmentation_features": segmentation_features.columns.tolist(),
        "include_svm": True,
        "bootstrap_artifacts": True,
        "deployment_model_strategy": "composite_metric_leader",
        "source_dataset": str(resolved_data_path),
        "source_dataset_name": Path(resolved_data_path).name,
    }
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metadata


def train_all(
    data_path: str | Path,
    include_cv_report: bool = False,
    include_svm: bool = False,
) -> pd.DataFrame:
    ensure_directories()
    overall_start = perf_counter()
    write_status(
        {
            "state": "starting",
            "current_stage": "loading_data",
            "completed_models": [],
            "include_cv_report": include_cv_report,
            "include_svm": include_svm,
        }
    )
    print("[1/4] Loading and preparing dataset...", flush=True)
    resolved_data_path = resolve_dataset_path(data_path)
    df = load_dataset(resolved_data_path)
    df = basic_cleaning(df)
    df = add_engineered_features(df)

    x, y = split_features_target(df)
    x = select_model_features(x)
    schema = infer_feature_schema(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=CONFIG.test_size,
        random_state=CONFIG.random_state,
        stratify=y,
    )

    evaluation_results = []
    trained_pipelines = {}
    cv_summaries = {}
    model_specs = get_model_specs(include_svm=include_svm)
    model_specs_by_name = {spec.name: spec for spec in model_specs}
    total_models = len(model_specs)

    write_status(
        {
            "state": "running",
            "current_stage": "training_models",
            "total_models": total_models,
            "completed_models": [],
            "include_cv_report": include_cv_report,
            "include_svm": include_svm,
        }
    )

    for index, spec in enumerate(model_specs, start=1):
        print(f"[2/4] Training model {index}/{total_models}: {spec.name}", flush=True)
        write_status(
            {
                "state": "running",
                "current_stage": "training_models",
                "current_model": spec.name,
                "model_index": index,
                "total_models": total_models,
                "completed_models": [result.model_name for result in evaluation_results],
                "include_cv_report": include_cv_report,
                "include_svm": include_svm,
            }
        )
        pipeline = build_training_pipeline(spec.estimator, schema)
        result, trained = evaluate_model(
            spec.name,
            pipeline,
            x_train,
            x_test,
            y_train,
            y_test,
            spec.complexity,
        )
        evaluation_results.append(result)
        trained_pipelines[spec.name] = trained
        print(
            f"[2/4] Finished {spec.name}: accuracy={result.accuracy:.4f}, "
            f"f1={result.f1_score:.4f}, roc_auc={result.roc_auc:.4f}, "
            f"train_time={result.train_time_seconds:.1f}s",
            flush=True,
        )
        save_partial_results(evaluation_results, trained_pipelines)

        if include_cv_report:
            print(f"[2/4] Running CV summary for {spec.name}...", flush=True)
            cv_summaries[spec.name] = cross_validation_overview(
                build_training_pipeline(spec.estimator, schema),
                x_train,
                y_train,
            )
            print(f"[2/4] CV summary saved for {spec.name}", flush=True)

        write_status(
            {
                "state": "running",
                "current_stage": "training_models",
                "current_model": spec.name,
                "model_index": index,
                "total_models": total_models,
                "completed_models": [result.model_name for result in evaluation_results],
                "latest_result": {
                    "model_name": result.model_name,
                    "accuracy": result.accuracy,
                    "f1_score": result.f1_score,
                    "roc_auc": result.roc_auc,
                    "train_time_seconds": result.train_time_seconds,
                },
                "include_cv_report": include_cv_report,
                "include_svm": include_svm,
            }
        )

    print("[3/4] Running guest segmentation with K-Means...", flush=True)
    results_df = results_to_dataframe(evaluation_results)
    best_model_name = results_df.iloc[0]["model_name"]
    best_holdout_pipeline = trained_pipelines[best_model_name]
    best_holdout_predictions = pd.Series(best_holdout_pipeline.predict(x_test), index=y_test.index)
    holdout_confusion = build_confusion_payload(y_test, best_holdout_predictions)
    print(
        f"[3/4] Re-fitting best model on the full dataset: {best_model_name}",
        flush=True,
    )
    best_raw_pipeline = fit_full_pipeline(
        model_specs_by_name[best_model_name],
        schema,
        x,
        y,
    )
    best_prediction_pipeline = best_raw_pipeline
    calibrated_model_available = False
    calibration_info = {}

    if best_model_name != "1D-CNN":
        print(
            f"[3/4] Calibrating probability outputs for deployment: {best_model_name}",
            flush=True,
        )
        calibration_raw_pipeline, best_prediction_pipeline, calibration_info = fit_calibrated_pipeline(
            model_specs_by_name[best_model_name],
            schema,
            x,
            y,
        )
        best_raw_pipeline = calibration_raw_pipeline
        calibrated_model_available = True

    results_df.to_csv(CONFIG.artifacts_dir / "model_comparison.csv", index=False)
    joblib.dump(best_raw_pipeline, CONFIG.artifacts_dir / "best_cancellation_model_raw.joblib")
    joblib.dump(best_prediction_pipeline, CONFIG.artifacts_dir / "best_cancellation_model.joblib")

    segmentation_features = get_segmentation_features(df)
    clustering_model, diagnostics = find_best_kmeans(segmentation_features)
    cluster_labels = clustering_model.predict(segmentation_features)
    cluster_profile = profile_clusters(segmentation_features, pd.Series(cluster_labels))
    diagnostics.update(
        {
            "fit_rows": int(len(segmentation_features)),
            "profile_rows": int(len(segmentation_features)),
            "bootstrap_fit_sample": False,
        }
    )
    save_cluster_artifacts(clustering_model, cluster_profile, diagnostics)

    print("[4/4] Saving metadata and processed dataset...", flush=True)
    metadata = {
        "target_column": CONFIG.target_column,
        "best_model_name": best_model_name,
        "prediction_model_artifact": "best_cancellation_model.joblib",
        "raw_model_artifact": "best_cancellation_model_raw.joblib",
        "smote_enabled": CONFIG.smote_enabled,
        "sampler": "SMOTENC" if CONFIG.smote_enabled else "None",
        "smote_sampling_ratio": CONFIG.smote_sampling_ratio if CONFIG.smote_enabled else None,
        "smote_k_neighbors": CONFIG.smote_k_neighbors if CONFIG.smote_enabled else None,
        "calibrated_model_available": calibrated_model_available,
        "probability_strategy": "sigmoid_holdout_calibration" if calibrated_model_available else "raw_predict_proba",
        "probability_calibration": calibration_info,
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "feature_columns": x.columns.tolist(),
        "categorical_columns": x.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist(),
        "cv_summary": cv_summaries,
        "testing_phase": {
            "method": "holdout_test_split",
            "test_size": CONFIG.test_size,
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "best_model_confusion_matrix": holdout_confusion,
        },
        "segmentation_features": segmentation_features.columns.tolist(),
        "include_svm": include_svm,
        "source_dataset": str(resolved_data_path),
        "source_dataset_name": Path(resolved_data_path).name,
    }
    with open(CONFIG.artifacts_dir / "training_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    x.to_csv(CONFIG.artifacts_dir / "training_input_snapshot.csv", index=False)

    processed_copy = df.copy()
    processed_copy["guest_cluster"] = cluster_labels
    processed_copy.to_csv(CONFIG.processed_dir / "hotel_bookings_enriched.csv", index=False)

    elapsed = perf_counter() - overall_start
    write_status(
        {
            "state": "completed",
            "current_stage": "done",
            "best_model_name": best_model_name,
            "total_models": total_models,
            "completed_models": [result.model_name for result in evaluation_results],
            "include_cv_report": include_cv_report,
            "include_svm": include_svm,
            "elapsed_seconds": elapsed,
        }
    )
    print(
        f"Training completed in {elapsed / 60:.2f} minutes. Best model: {best_model_name}",
        flush=True,
    )

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hotel cancellation and segmentation models.")
    parser.add_argument("--data", required=True, help="Path to hotel_bookings.csv")
    parser.add_argument(
        "--include-cv-report",
        action="store_true",
        help="Run extra cross-validation summaries for each model. Slower, but useful for reporting.",
    )
    parser.add_argument(
        "--include-svm",
        action="store_true",
        help="Include SVM training. Disabled by default because it is the slowest model.",
    )
    args = parser.parse_args()

    results = train_all(
        args.data,
        include_cv_report=args.include_cv_report,
        include_svm=args.include_svm,
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
