"""Phase 5: train and evaluate baseline ML models on the Phase 4 data.

All preprocessing is fitted inside scikit-learn pipelines on training data
only. The prepared synthetic dataset and all existing outputs are read-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "real_estate_phase4_prepared_2.csv"
REPORT_BASE = ROOT / "reports" / "phase5_model_evaluation.md"
FIGURES_DIR = ROOT / "reports" / "figures" / "phase5"
MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC_FEATURES = [
    "BHK", "Area_SqFt", "Property_Age_Years", "Floor_No", "Total_Floors",
    "Distance_to_CBD_km", "Distance_to_Metro_km", "Distance_to_School_km",
    "Distance_to_Hospital_km", "Amenities_Score_1_to_10", "Price_INR_Lakhs",
]
CLASSIFICATION_TARGET = "Target_Good_Investment_Label"
REGRESSION_TARGET = "Target_Projected_Value_After_5Y_INR_Lakhs"
EXCLUDED_FROM_PREDICTORS = {
    "Property_ID",
    "Price_Per_SqFt_INR_Corrected",
    "Net_Rental_Yield_Pct_Corrected",
    CLASSIFICATION_TARGET,
    REGRESSION_TARGET,
}


def unused(path: Path) -> Path:
    if not path.exists():
        return path
    for number in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{number}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"No unused output name available for {path}")


def save_figure(filename: str) -> str:
    target = unused(FIGURES_DIR / filename)
    plt.tight_layout()
    plt.savefig(target, dpi=160, bbox_inches="tight")
    plt.close()
    return target.name


def make_preprocessor(numeric_columns: list[str], categorical_columns: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical_one_hot", "passthrough", categorical_columns),
        ],
        remainder="drop",
    )


def classification_metrics(y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray | None) -> dict[str, float | str]:
    roc_auc: float | str
    if probabilities is not None and len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, probabilities))
    else:
        roc_auc = "N/A"
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": roc_auc,
    }


def regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def fmt_metric(value: float | str) -> str:
    return value if isinstance(value, str) else f"{value:.4f}"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {DATA_PATH}")
    report_path = unused(REPORT_BASE)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    required_targets = {CLASSIFICATION_TARGET, REGRESSION_TARGET}
    missing_targets = sorted(required_targets.difference(df.columns))
    if missing_targets:
        raise ValueError(f"Expected target columns are missing: {missing_targets}")
    if df.isna().any().any():
        raise ValueError("Prepared dataset contains missing values; stopping before training.")
    if int(df.duplicated().sum()) or int(df["Property_ID"].duplicated().sum()):
        raise ValueError("Prepared dataset contains duplicate rows or property IDs.")
    for target in [CLASSIFICATION_TARGET, REGRESSION_TARGET]:
        if not np.isfinite(df[target].to_numpy(dtype=float)).all():
            raise ValueError(f"Target {target} contains non-finite values.")
    if not set(df[CLASSIFICATION_TARGET].unique()).issubset({0, 1}):
        raise ValueError("Classification target is not binary.")
    if (df[REGRESSION_TARGET] <= 0).any():
        raise ValueError("Regression target contains non-positive values.")

    feature_columns = [column for column in df.columns if column not in EXCLUDED_FROM_PREDICTORS]
    categorical_columns = [column for column in feature_columns if column not in NUMERIC_FEATURES]
    missing_numeric = sorted(set(NUMERIC_FEATURES).difference(feature_columns))
    if missing_numeric:
        raise ValueError(f"Expected numeric predictor columns are missing: {missing_numeric}")
    if not categorical_columns:
        raise ValueError("No categorical/encoded predictor columns found.")
    X = df[feature_columns]
    y_class = df[CLASSIFICATION_TARGET]
    y_value = df[REGRESSION_TARGET]

    class_counts = y_class.value_counts().sort_index()
    class_distribution = {str(int(key)): int(value) for key, value in class_counts.items()}
    class_train_X, class_test_X, class_train_y, class_test_y = train_test_split(
        X, y_class, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_class
    )
    reg_train_X, reg_test_X, reg_train_y, reg_test_y = train_test_split(
        X, y_value, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    classification_models = {
        "Logistic Regression": Pipeline([
            ("preprocess", make_preprocessor(NUMERIC_FEATURES, categorical_columns, True)),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree Classifier": Pipeline([
            ("preprocess", make_preprocessor(NUMERIC_FEATURES, categorical_columns, False)),
            ("model", DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=RANDOM_STATE)),
        ]),
    }
    class_predictions: dict[str, np.ndarray] = {}
    class_probabilities: dict[str, np.ndarray | None] = {}
    class_results: dict[str, dict[str, float | str]] = {}
    class_confusion: dict[str, np.ndarray] = {}
    for name, model in classification_models.items():
        model.fit(class_train_X, class_train_y)
        prediction = model.predict(class_test_X)
        probability = model.predict_proba(class_test_X)[:, 1] if hasattr(model, "predict_proba") else None
        class_predictions[name] = prediction
        class_probabilities[name] = probability
        class_results[name] = classification_metrics(class_test_y, prediction, probability)
        class_confusion[name] = confusion_matrix(class_test_y, prediction, labels=[0, 1])
        joblib.dump(model, unused(MODELS_DIR / f"phase5_{name.lower().replace(' ', '_')}.joblib"))

    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(class_train_X, class_train_y)
    majority_prediction = majority.predict(class_test_X)
    class_results["Majority Baseline"] = classification_metrics(class_test_y, majority_prediction, None)
    class_confusion["Majority Baseline"] = confusion_matrix(class_test_y, majority_prediction, labels=[0, 1])
    joblib.dump(majority, unused(MODELS_DIR / "phase5_majority_baseline.joblib"))

    regression_models = {
        "Linear Regression": Pipeline([
            ("preprocess", make_preprocessor(NUMERIC_FEATURES, categorical_columns, True)),
            ("model", LinearRegression()),
        ]),
        "Random Forest Regressor": Pipeline([
            ("preprocess", make_preprocessor(NUMERIC_FEATURES, categorical_columns, False)),
            ("model", RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }
    regression_predictions: dict[str, np.ndarray] = {}
    regression_results: dict[str, dict[str, float]] = {}
    for name, model in regression_models.items():
        model.fit(reg_train_X, reg_train_y)
        prediction = model.predict(reg_test_X)
        regression_predictions[name] = prediction
        regression_results[name] = regression_metrics(reg_test_y, prediction)
        joblib.dump(model, unused(MODELS_DIR / f"phase5_{name.lower().replace(' ', '_')}.joblib"))

    mean_baseline = DummyRegressor(strategy="mean")
    mean_baseline.fit(reg_train_X, reg_train_y)
    mean_prediction = mean_baseline.predict(reg_test_X)
    regression_results["Mean Baseline"] = regression_metrics(reg_test_y, mean_prediction)
    joblib.dump(mean_baseline, unused(MODELS_DIR / "phase5_mean_baseline.joblib"))

    sns.set_theme(style="whitegrid")
    figure_names: list[str] = []
    plt.figure(figsize=(8, 5))
    class_plot = pd.DataFrame(class_results).T.replace("N/A", np.nan)
    class_plot[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(kind="bar", ax=plt.gca())
    plt.title("Classification Test Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right")
    figure_names.append(save_figure("classification_metrics.png"))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, (name, matrix) in zip(axes, class_confusion.items()):
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(name)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_xticklabels(["0", "1"])
        axis.set_yticklabels(["0", "1"], rotation=0)
    figure_names.append(save_figure("classification_confusion_matrices.png"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, (name, prediction) in zip(axes, regression_predictions.items()):
        axis.scatter(reg_test_y, prediction, alpha=0.55, s=18)
        low = min(float(reg_test_y.min()), float(prediction.min()))
        high = max(float(reg_test_y.max()), float(prediction.max()))
        axis.plot([low, high], [low, high], color="black", linestyle="--")
        axis.set_title(name)
        axis.set_xlabel("Actual projected value (INR lakhs)")
        axis.set_ylabel("Predicted value (INR lakhs)")
    figure_names.append(save_figure("regression_actual_vs_predicted.png"))

    reg_plot = pd.DataFrame(regression_results).T
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, metric in zip(axes, ["mae", "rmse", "r2"]):
        reg_plot[metric].plot(kind="bar", ax=axis, color="#0f766e")
        axis.set_title(metric.upper())
        axis.set_ylabel(metric.upper())
        axis.tick_params(axis="x", rotation=20)
    figure_names.append(save_figure("regression_metrics.png"))

    plt.figure(figsize=(9, 5))
    sns.histplot(reg_train_y, color="#2563eb", label="Train", kde=False, alpha=0.45)
    sns.histplot(reg_test_y, color="#b45309", label="Test", kde=False, alpha=0.45)
    plt.title("Projected Five-Year Value Target Distribution")
    plt.xlabel("Target projected value (INR lakhs)")
    plt.legend()
    figure_names.append(save_figure("target_distribution.png"))

    model_files = sorted(str(path.relative_to(ROOT)) for path in MODELS_DIR.glob("phase5_*.joblib"))
    metadata_path = unused(MODELS_DIR / "phase5_metadata.json")
    metadata_path.write_text(json.dumps({
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "classification_target": CLASSIFICATION_TARGET,
        "regression_target": REGRESSION_TARGET,
        "feature_columns": feature_columns,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_encoded_features": categorical_columns,
        "excluded_from_predictors": sorted(EXCLUDED_FROM_PREDICTORS),
    }, indent=2), encoding="utf-8")

    class_table = "\n".join(
        f"| {name} | {fmt_metric(metrics['accuracy'])} | {fmt_metric(metrics['precision'])} | {fmt_metric(metrics['recall'])} | {fmt_metric(metrics['f1'])} | {fmt_metric(metrics['roc_auc'])} | `{class_confusion[name].tolist()}` |"
        for name, metrics in class_results.items()
    )
    reg_table = "\n".join(
        f"| {name} | {metrics['mae']:,.4f} | {metrics['rmse']:,.4f} | {metrics['r2']:.4f} |"
        for name, metrics in regression_results.items()
    )
    report = f"""# Phase 5: Machine Learning Model Development and Evaluation

## Scope and provenance

Models were trained from `data/processed/real_estate_phase4_prepared_2.csv`, which is derived from a **synthetic, assumption-based dataset**. These results describe performance on this file only. They do not establish real-world accuracy for Indian property prices or investment quality.

- Random seed: **{RANDOM_STATE}**
- Test size: **{TEST_SIZE:.0%}**
- Prepared rows / columns: **{len(df):,} / {len(df.columns):,}**
- Missing cells: **{int(df.isna().sum().sum()):,}**
- Duplicate rows / IDs: **{int(df.duplicated().sum()):,} / {int(df['Property_ID'].duplicated().sum()):,}**
- Classification target: `{CLASSIFICATION_TARGET}`
- Regression target: `{REGRESSION_TARGET}`

## Feature and leakage policy

The predictor matrix contains current property characteristics: BHK, area, age, floor information, distances, amenities, current price, and Phase 4 one-hot category indicators for city, locality tier, property type, parking, gated community, furnishing, and occupancy. `Property_ID` is excluded.

The corrected rental-yield and price-per-square-foot fields are excluded as derived fields. Future projected value, projected capital gain, investment labels, and any target columns are excluded. The prepared file contains no assumed appreciation-rate column, so no future appreciation assumption was used as a predictor. Current property price is included for the value task because it is available at prediction time and is a legitimate current-value input; this means the model evaluates conditional value projection rather than price discovery from zero information.

Preprocessing is inside each pipeline: numeric columns are standardized for Logistic Regression and Linear Regression, while one-hot indicators are passed through. Tree models receive the same one-hot representation without numeric scaling. All preprocessing is fitted on training folds only; the test data was not used for fitting or model selection.

## Pre-training validation

- Classification class counts: `{json.dumps(class_distribution)}`; positive class share: **{float(y_class.mean()):.2%}**.
- Regression target range: **{float(y_value.min()):,.4f} to {float(y_value.max()):,.4f} INR lakhs**; mean: **{float(y_value.mean()):,.4f}**; median: **{float(y_value.median()):,.4f}**.
- Both classification classes were present in the stratified train/test split: **{len(np.unique(class_train_y)) == 2 and len(np.unique(class_test_y)) == 2}**.
- No missing, duplicate, non-finite, or non-positive target values were found.

## Model 1: Good Investment Classification

The target is the transparent synthetic rule label. The four rule inputs used to construct that label are intentionally not predictors, so these models do not simply reproduce the rule through direct leakage.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion matrix `[[TN, FP], [FN, TP]]` |
|---|---:|---:|---:|---:|---:|---|
{class_table}

The majority baseline predicts the most frequent class for every test property. The label distribution is imbalanced, so accuracy alone can look acceptable without identifying positive properties; precision, recall, F1, ROC-AUC, and the confusion matrices give a fuller view. The five source-label mismatches corrected in Phase 4 also mean this target is a rule-defined educational target, not an observed investment outcome.

## Model 2: Five-Year Property Value Regression

The target is the Phase 4 corrected five-year projected value. Linear Regression and Random Forest Regressor were evaluated against a mean-prediction baseline.

| Model | MAE (INR lakhs) | RMSE (INR lakhs) | R² |
|---|---:|---:|---:|
{reg_table}

Lower MAE/RMSE and higher R² are better. Because the target is generated from current price and an assumption-based appreciation formula, these metrics measure fit to synthetic calculations rather than validated future market behavior. The current-price feature is allowed because it is known at prediction time; assumed appreciation is excluded because it would directly provide unavailable future scenario information in the intended setting.

## Evaluation figures and saved models

Figures were written to `reports/figures/phase5/` without overwriting existing files:

{chr(10).join(f'- `{name}`' for name in figure_names)}

Serialized pipelines and baselines were written under `models/`:

{chr(10).join(f'- `{name}`' for name in model_files)}
- `{metadata_path.relative_to(ROOT)}`

No MLflow configuration or Streamlit application was created. These models are reproducible artifacts for this synthetic dataset, not evidence of deployable real-world investment prediction.
"""
    report_path.write_text(report, encoding="utf-8")
    print(f"Report: {report_path}")
    print(f"Figures written: {len(figure_names)}")
    print(f"Models written: {len(model_files) + 1}")
    print(f"Classification target/classes: {CLASSIFICATION_TARGET}/{class_distribution}")
    print(f"Regression target range: {float(y_value.min()):.4f} to {float(y_value.max()):.4f}")
    print(f"Classification metrics: {class_results}")
    print(f"Regression metrics: {regression_results}")


if __name__ == "__main__":
    main()
