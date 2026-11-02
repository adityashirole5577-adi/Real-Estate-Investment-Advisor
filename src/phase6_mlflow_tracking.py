"""Phase 6: track the existing Phase 5 artifacts in local MLflow.

This script does not retrain models. It loads the saved Phase 5 joblib files,
parses the exact metrics from the Phase 5 report, logs them to a file-backed
MLflow experiment, and verifies the resulting runs and artifacts.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import joblib

# MLflow 3.x protects its maintenance-mode filesystem backend by default.
# Phase 6 intentionally uses a local file store, so opt into that backend.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[1]
PHASE5_REPORT = ROOT / "reports" / "phase5_model_evaluation.md"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures" / "phase5"
TRACKING_DIR = ROOT / "mlruns"
EXPERIMENT_NAME = "Real_Estate_Investment_Advisor"
REPORT_BASE = ROOT / "reports" / "phase6_mlflow_tracking.md"
RANDOM_STATE = 42
TEST_SIZE = 0.20

CLASSIFICATION_MODELS = {
    "Logistic Regression": "phase5_logistic_regression.joblib",
    "Decision Tree Classifier": "phase5_decision_tree_classifier.joblib",
    "Majority Baseline": "phase5_majority_baseline.joblib",
}
REGRESSION_MODELS = {
    "Linear Regression": "phase5_linear_regression.joblib",
    "Random Forest Regressor": "phase5_random_forest_regressor.joblib",
    "Mean Baseline": "phase5_mean_baseline.joblib",
}


def unused(path: Path) -> Path:
    if not path.exists():
        return path
    for number in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{number}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"No unused output name available for {path}")


def parse_phase5_metrics(report: str) -> tuple[dict[str, dict[str, float | str]], dict[str, dict[str, float]]]:
    """Parse model rows from the persisted Phase 5 Markdown report."""
    classification: dict[str, dict[str, float | str]] = {}
    regression: dict[str, dict[str, float]] = {}
    class_pattern = re.compile(
        r"^\| (Logistic Regression|Decision Tree Classifier|Majority Baseline) "
        r"\| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| "
        r"(N/A|[0-9.]+) \|"
    )
    reg_pattern = re.compile(
        r"^\| (Linear Regression|Random Forest Regressor|Mean Baseline) "
        r"\| (-?[0-9.]+) \| (-?[0-9.]+) \| (-?[0-9.]+) \|"
    )
    for line in report.splitlines():
        class_match = class_pattern.match(line)
        if class_match:
            name, accuracy, precision, recall, f1, roc_auc = class_match.groups()
            classification[name] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "roc_auc": roc_auc if roc_auc == "N/A" else float(roc_auc),
            }
        reg_match = reg_pattern.match(line)
        if reg_match:
            name, mae, rmse, r2 = reg_match.groups()
            regression[name] = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}
    expected_classification = set(CLASSIFICATION_MODELS)
    expected_regression = set(REGRESSION_MODELS)
    if set(classification) != expected_classification:
        raise ValueError(f"Could not parse all classification metrics: {classification}")
    if set(regression) != expected_regression:
        raise ValueError(f"Could not parse all regression metrics: {regression}")
    return classification, regression


def log_params_from_model(model: object) -> None:
    """Log compact, reproducible model parameters without logging large objects."""
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model")
    else:
        estimator = model
    if estimator is None or not hasattr(estimator, "get_params"):
        return
    selected = {
        key: value
        for key, value in estimator.get_params(deep=False).items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    mlflow.log_params({f"model_{key}": "None" if value is None else value for key, value in selected.items()})


def log_relevant_artifacts(task: str, model_name: str) -> list[str]:
    artifacts = [PHASE5_REPORT, MODELS_DIR / "phase5_metadata.json"]
    if task == "classification":
        artifacts.extend([
            FIGURES_DIR / "classification_metrics.png",
            FIGURES_DIR / "classification_confusion_matrices.png",
        ])
    else:
        artifacts.extend([
            FIGURES_DIR / "regression_metrics.png",
            FIGURES_DIR / "regression_actual_vs_predicted.png",
            FIGURES_DIR / "target_distribution.png",
        ])
    missing = [str(path) for path in artifacts if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required artifact(s) missing for {model_name}: {missing}")
    for path in artifacts:
        if path == PHASE5_REPORT:
            mlflow.log_artifact(str(path), artifact_path="reports")
        elif path.suffix == ".png":
            mlflow.log_artifact(str(path), artifact_path="figures")
        else:
            mlflow.log_artifact(str(path), artifact_path="metadata")
    return [str(path.relative_to(ROOT)) for path in artifacts]


def track_run(
    name: str,
    task: str,
    model_filename: str,
    metrics: dict[str, float | str],
    target: str,
    feature_count: int,
) -> dict[str, object]:
    model_path = MODELS_DIR / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    model = joblib.load(model_path)
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({
            "task": task,
            "model_role": "baseline" if "Baseline" in name else "model",
            "data_provenance": "synthetic",
            "training_status": "existing_saved_model_not_retrained",
        })
        mlflow.log_params({
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "target": target,
            "feature_count": feature_count,
            "source_model_file": model_filename,
            "metrics_source": "reports/phase5_model_evaluation.md",
        })
        log_params_from_model(model)
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(metric_name, float(value))
        mlflow.log_artifact(str(model_path), artifact_path="model")
        artifact_paths = log_relevant_artifacts(task, name)
        run_id = run.info.run_id
    return {
        "name": name,
        "task": task,
        "run_id": run_id,
        "metrics": metrics,
        "artifacts": [f"model/{model_filename}"] + artifact_paths,
    }


def verify_run(client: MlflowClient, run_id: str) -> dict[str, object]:
    run = client.get_run(run_id)
    artifacts = client.list_artifacts(run_id)
    artifact_names = [artifact.path for artifact in artifacts]
    nested_count = sum(1 + len(client.list_artifacts(run_id, artifact.path)) for artifact in artifacts if artifact.is_dir)
    return {
        "run_id": run_id,
        "status": run.info.status,
        "metric_count": len(run.data.metrics),
        "parameter_count": len(run.data.params),
        "top_level_artifact_paths": artifact_names,
        "artifact_entry_count": len(artifacts) + nested_count,
    }


def main() -> None:
    if not PHASE5_REPORT.exists():
        raise FileNotFoundError(f"Phase 5 report not found: {PHASE5_REPORT}")
    report_text = PHASE5_REPORT.read_text(encoding="utf-8")
    classification_metrics, regression_metrics = parse_phase5_metrics(report_text)
    metadata = json.loads((MODELS_DIR / "phase5_metadata.json").read_text(encoding="utf-8"))
    feature_count = len(metadata["feature_columns"])
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    tracking_uri = TRACKING_DIR.as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"Experiment was not created: {EXPERIMENT_NAME}")

    results: list[dict[str, object]] = []
    for name, filename in CLASSIFICATION_MODELS.items():
        results.append(track_run(name, "classification", filename, classification_metrics[name], metadata["classification_target"], feature_count))
    for name, filename in REGRESSION_MODELS.items():
        results.append(track_run(name, "regression", filename, regression_metrics[name], metadata["regression_target"], feature_count))

    client = MlflowClient(tracking_uri=tracking_uri)
    verifications = [verify_run(client, str(result["run_id"])) for result in results]
    failed = [verification for verification in verifications if verification["status"] != "FINISHED" or verification["artifact_entry_count"] < 4]
    if failed:
        raise RuntimeError(f"MLflow verification failed: {failed}")

    report_path = unused(REPORT_BASE)
    result_lines = "\n".join(
        f"| {result['name']} | {result['task']} | `{result['run_id']}` | {verification['metric_count']} | {verification['parameter_count']} | {verification['artifact_entry_count']} |"
        for result, verification in zip(results, verifications)
    )
    metric_lines = []
    for result in results:
        metrics_text = ", ".join(f"`{key}`={value}" for key, value in result["metrics"].items())
        metric_lines.append(f"- **{result['name']}**: {metrics_text}")
    report = f"""# Phase 6: MLflow Experiment Tracking

## Tracking configuration

- Tracking store: `{TRACKING_DIR.relative_to(ROOT)}/` using MLflow's local file backend
- Experiment: **{EXPERIMENT_NAME}**
- Tracking URI: `{tracking_uri}`
- MLflow file-store opt-in: `MLFLOW_ALLOW_FILE_STORE=true` (required by MLflow 3.x for this local backend)
- Source metrics: `reports/phase5_model_evaluation.md`
- Existing saved pipelines were loaded from `models/`; no model was retrained.

The models were trained on synthetic, assumption-based data. MLflow records experiment provenance and reproducibility metadata; it does not validate these models as real-world property investment predictors.

## Runs and verification

| Run | Task | Run ID | Logged metrics | Logged artifact entries |
|---|---|---|---:|---:|
{result_lines}

All six runs finished with status **FINISHED**. The MLflow client successfully retrieved every run and its logged metrics/artifacts. Each model run includes the corresponding saved pipeline, the Phase 5 evaluation report, the Phase 5 metadata feature/target definition, and the relevant evaluation charts. Baseline metrics were logged for the majority-class and mean-prediction baselines; the majority baseline has no ROC-AUC because it produces one prediction class, so ROC-AUC was not logged for that metric.

## Exact metrics logged

{chr(10).join(metric_lines)}

The values above were parsed from the persisted Phase 5 report rather than recomputed or guessed. The four requested model runs are Logistic Regression, Decision Tree Classifier, Linear Regression, and Random Forest Regressor; the two additional runs are the Phase 5 majority and mean baselines.

## Artifacts

Each classification run logged:

- Its saved joblib pipeline or baseline under `model/`.
- `reports/phase5_model_evaluation.md` under `reports/`.
- `models/phase5_metadata.json` under `metadata/`.
- `classification_metrics.png` and `classification_confusion_matrices.png` under `figures/`.

Each regression run logged its saved joblib pipeline or baseline, the same report and metadata, and `regression_metrics.png`, `regression_actual_vs_predicted.png`, and `target_distribution.png` under `figures/`.

## Local MLflow UI

From the project root, activate the project virtual environment and run:

```powershell
mlflow ui --backend-store-uri .\\mlruns --port 5000
```

Then open `http://127.0.0.1:5000` and select the `{EXPERIMENT_NAME}` experiment. No cloud account or external tracking server is required.

## Limitations

The tracked models and metrics reflect one fixed split of a synthetic dataset. The classification target is an educational rule, and the regression target is an assumption-based five-year projection. Strong metrics here must not be interpreted as evidence of actual Indian property-price or investment-quality prediction. No raw or prepared dataset was modified, and no Streamlit application was created.
"""
    report_path.write_text(report, encoding="utf-8")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Runs verified: {len(verifications)}")
    print(f"Report: {report_path}")
    print(f"Run IDs: {', '.join(str(result['run_id']) for result in results)}")


if __name__ == "__main__":
    main()
