# Phase 5: Machine Learning Model Development and Evaluation

## Scope and provenance

Models were trained from `data/processed/real_estate_phase4_prepared_2.csv`, which is derived from a **synthetic, assumption-based dataset**. These results describe performance on this file only. They do not establish real-world accuracy for Indian property prices or investment quality.

- Random seed: **42**
- Test size: **20%**
- Prepared rows / columns: **5,000 / 42**
- Missing cells: **0**
- Duplicate rows / IDs: **0 / 0**
- Classification target: `Target_Good_Investment_Label`
- Regression target: `Target_Projected_Value_After_5Y_INR_Lakhs`

## Feature and leakage policy

The predictor matrix contains current property characteristics: BHK, area, age, floor information, distances, amenities, current price, and Phase 4 one-hot category indicators for city, locality tier, property type, parking, gated community, furnishing, and occupancy. `Property_ID` is excluded.

The corrected rental-yield and price-per-square-foot fields are excluded as derived fields. Future projected value, projected capital gain, investment labels, and any target columns are excluded. The prepared file contains no assumed appreciation-rate column, so no future appreciation assumption was used as a predictor. Current property price is included for the value task because it is available at prediction time and is a legitimate current-value input; this means the model evaluates conditional value projection rather than price discovery from zero information.

Preprocessing is inside each pipeline: numeric columns are standardized for Logistic Regression and Linear Regression, while one-hot indicators are passed through. Tree models receive the same one-hot representation without numeric scaling. All preprocessing is fitted on training folds only; the test data was not used for fitting or model selection.

## Pre-training validation

- Classification class counts: `{"0": 3829, "1": 1171}`; positive class share: **23.42%**.
- Regression target range: **20.7731 to 1,014.8068 INR lakhs**; mean: **173.4445**; median: **146.2529**.
- Both classification classes were present in the stratified train/test split: **True**.
- No missing, duplicate, non-finite, or non-positive target values were found.

## Model 1: Good Investment Classification

The target is the transparent synthetic rule label. The four rule inputs used to construct that label are intentionally not predictors, so these models do not simply reproduce the rule through direct leakage.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion matrix `[[TN, FP], [FN, TP]]` |
|---|---:|---:|---:|---:|---:|---|
| Logistic Regression | 0.8050 | 0.6258 | 0.4145 | 0.4987 | 0.8226 | `[[708, 58], [137, 97]]` |
| Decision Tree Classifier | 0.7960 | 0.5789 | 0.4701 | 0.5189 | 0.8362 | `[[686, 80], [124, 110]]` |
| Majority Baseline | 0.7660 | 0.0000 | 0.0000 | 0.0000 | N/A | `[[766, 0], [234, 0]]` |

The majority baseline predicts the most frequent class for every test property. The label distribution is imbalanced, so accuracy alone can look acceptable without identifying positive properties; precision, recall, F1, ROC-AUC, and the confusion matrices give a fuller view. The five source-label mismatches corrected in Phase 4 also mean this target is a rule-defined educational target, not an observed investment outcome.

## Model 2: Five-Year Property Value Regression

The target is the Phase 4 corrected five-year projected value. Linear Regression and Random Forest Regressor were evaluated against a mean-prediction baseline.

| Model | MAE (INR lakhs) | RMSE (INR lakhs) | R² |
|---|---:|---:|---:|
| Linear Regression | 11.5834 | 17.2580 | 0.9689 |
| Random Forest Regressor | 11.9680 | 17.8242 | 0.9669 |
| Mean Baseline | 73.3024 | 98.1463 | -0.0045 |

Lower MAE/RMSE and higher R² are better. Because the target is generated from current price and an assumption-based appreciation formula, these metrics measure fit to synthetic calculations rather than validated future market behavior. The current-price feature is allowed because it is known at prediction time; assumed appreciation is excluded because it would directly provide unavailable future scenario information in the intended setting.

## Evaluation figures and saved models

Figures were written to `reports/figures/phase5/` without overwriting existing files:

- `classification_metrics.png`
- `classification_confusion_matrices.png`
- `regression_actual_vs_predicted.png`
- `regression_metrics.png`
- `target_distribution.png`

Serialized pipelines and baselines were written under `models/`:

- `models\phase5_decision_tree_classifier.joblib`
- `models\phase5_linear_regression.joblib`
- `models\phase5_logistic_regression.joblib`
- `models\phase5_majority_baseline.joblib`
- `models\phase5_mean_baseline.joblib`
- `models\phase5_random_forest_regressor.joblib`
- `models\phase5_metadata.json`

No MLflow configuration or Streamlit application was created. These models are reproducible artifacts for this synthetic dataset, not evidence of deployable real-world investment prediction.
