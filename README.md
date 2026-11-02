# Real Estate Investment Advisor

Predicting Property Profitability & Future Value



## Live application and project handbook

- **Public Streamlit application:** https://real-estate-investment-advisor-8z7ppsrhfksyfdihc3f5e4.streamlit.app/
- **Project handbook:** [Read the full project handbook](./docs/PROJECT_HANDBOOK.md)

The handbook documents the project problem statement, business use cases, phase-wise methodology, data flow, dataset fields, target-label logic, 20-question EDA guide, models, recorded evaluation results, MLflow, Streamlit deployment, setup steps, deliverables and limitations.

## Overview

This project is an educational real-estate analytics application for exploring two property scenarios:

1. Whether a property matches the project’s synthetic `Good_Investment` label pattern.
2. A model-based estimate of the property’s projected value after five years.

The project includes data validation, exploratory analysis, feature preparation, model training and evaluation, local MLflow tracking, and a Streamlit dashboard. The investment dataset is synthetic and assumption-based. Its predictions are not verified market forecasts, investment recommendations, or financial advice.

## Problem statement and objectives

The project investigates whether commonly available property characteristics can be used to support a simple investment-scenario analysis. Its objectives are to:

- Validate the structure and calculated fields in the synthetic investment dataset.
- Prepare current-property features without using target-derived or future-only information.
- Evaluate classification and regression models against simple baselines.
- Track the existing model artifacts and metrics in a local MLflow experiment.
- Provide a beginner-friendly Streamlit interface for local demonstration.

## Streamlit application

The application is implemented in [`app.py`](./app.py). It provides:

- City, locality tier, property type, BHK, area, property age, and floor inputs.
- Current price and distance inputs for the city centre, metro, school, and hospital.
- Amenities, parking, gated-community, furnishing, and occupancy inputs.
- Input validation, including a check that the floor number does not exceed total floors.
- Predict and reset controls.
- A synthetic investment-label result and model probability when available.
- A five-year projected-value estimate in INR lakhs and INR.
- A scenario summary and a chart comparing current price with the five-year estimate.
- A visible synthetic-data disclaimer.

The application loads existing saved pipelines and does not retrain models when it starts:

- `models/phase5_logistic_regression.joblib`
- `models/phase5_linear_regression.joblib`
- `models/phase5_metadata.json`

## Datasets

### Synthetic investment dataset

[`data/raw/india_real_estate_investment_synthetic_dataset.csv`](./data/raw/india_real_estate_investment_synthetic_dataset.csv) contains 5,000 rows and 28 columns. It includes property characteristics, prices, rent and cost assumptions, assumed appreciation rates, projected five-year values, and an investment label.

The dataset is explicitly synthetic and assumption-based. Prices, rents, appreciation rates, projected values, and labels are not verified market observations. The label is an educational rule requiring all of the following:

- Net rental yield ≥ 3%.
- Assumed annual appreciation rate ≥ 7.5%.
- Distance to the city centre ≤ 15 km.
- Amenities score ≥ 4.

The Phase 4 preparation created corrected derived fields and targets in [`data/processed/real_estate_phase4_prepared_2.csv`](./data/processed/real_estate_phase4_prepared_2.csv) without modifying the raw CSV. The prepared file contains 5,000 rows and 42 columns, with no missing cells or duplicate property IDs according to the Phase 4 report.

### Original housing-price dataset

[`data/raw/india_housing_prices.csv`](./data/raw/india_housing_prices.csv) is a separate housing-price dataset with 250,000 rows and 23 columns, including state, city, locality, property type, size, price, year built, furnishing, floors, nearby facilities, parking, security, amenities, ownership, and availability fields. Its cleaned version is [`data/processed/india_housing_prices_cleaned.csv`](./data/processed/india_housing_prices_cleaned.csv), with 250,000 rows and 28 columns.

This original housing dataset is used for project inspection and earlier exploratory work. The project’s trained investment models use the prepared synthetic investment dataset described above.

## Technology stack

- Python 3.13.5 in the project virtual environment.
- pandas
- matplotlib
- seaborn
- scikit-learn
- joblib, used by the saved model artifacts and scripts
- Streamlit
- MLflow with a local file-based tracking store
- Python `unittest` and Streamlit `AppTest` for application tests

The dependency ranges are maintained in [`requirements.txt`](./requirements.txt).

## Project structure

```text
Real_Estate_Investment_Advisor/
├── app.py
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── india_housing_prices.csv
│   │   └── india_real_estate_investment_synthetic_dataset.csv
│   └── processed/
│       ├── india_housing_prices_cleaned.csv
│       ├── real_estate_phase4_prepared.csv
│       └── real_estate_phase4_prepared_2.csv
├── models/
│   ├── phase5_decision_tree_classifier.joblib
│   ├── phase5_linear_regression.joblib
│   ├── phase5_logistic_regression.joblib
│   ├── phase5_majority_baseline.joblib
│   ├── phase5_mean_baseline.joblib
│   ├── phase5_metadata.json
│   └── phase5_random_forest_regressor.joblib
├── reports/
│   ├── phase3_dataset_validation*.md
│   ├── phase4_data_preparation_2.md
│   ├── phase5_model_evaluation.md
│   ├── phase6_mlflow_tracking.md
│   ├── phase7_streamlit_app.md
│   └── figures/
├── src/
│   ├── phase2_5_suitability.py
│   ├── phase2_eda.py
│   ├── phase3_eda.py
│   ├── phase3_eda_2.py
│   ├── phase4_prepare_data.py
│   ├── phase5_train_models.py
│   └── phase6_mlflow_tracking.py
├── tests/
│   └── test_phase7_app.py
└── mlruns/
```

`mlruns/` contains local tracking data and is excluded from Git by the project’s [`.gitignore`](./.gitignore). The two prepared Phase 4 filenames are retained because both exist in the project; the Phase 5 report identifies `real_estate_phase4_prepared_2.csv` as its source.

## Windows PowerShell setup

Open PowerShell and move to the project directory:

```powershell
Set-Location "C:\Users\Admin\OneDrive\Desktop\Real_Estate_Investment_Advisor"
```

Create the virtual environment if it does not already exist:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks local script execution for the current user, use the project-approved execution-policy command appropriate for your environment, or invoke the environment’s interpreter directly instead of activating it.

Install the project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The existing project environment was verified with Python 3.13.5. The dependency file specifies minimum versions rather than a fully locked environment.

## Run the Streamlit app

From the project root with the virtual environment active:

```powershell
streamlit run app.py
```

The application uses the existing artifacts in `models/`. If a required model artifact is missing, the app reports the missing artifact and points to the Phase 5 training script.

## Run the existing tests

From the project root:

```powershell
python -m unittest discover -s tests -p 'test*.py' -v
```

The existing test suite covers saved-model compatibility, Streamlit rendering and prediction, and the school/hospital input ranges. The test file is [`tests/test_phase7_app.py`](./tests/test_phase7_app.py).

## Local MLflow UI

The project uses the local file-based tracking store in `mlruns/` and the experiment `Real_Estate_Investment_Advisor`. MLflow 3.x requires the file-store opt-in environment variable for this local backend.

From the project root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$env:MLFLOW_ALLOW_FILE_STORE = "true"
mlflow ui --backend-store-uri .\mlruns --port 5000
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) and select the `Real_Estate_Investment_Advisor` experiment.

The Phase 6 tracking script also sets `MLFLOW_ALLOW_FILE_STORE=true` when it runs:

```powershell
python src/phase6_mlflow_tracking.py
```

The existing Phase 6 report records six verified finished runs: Logistic Regression, Decision Tree Classifier, Majority Baseline, Linear Regression, Random Forest Regressor, and Mean Baseline.

## Models and reported metrics

### Classification: synthetic good-investment label

The classification target is `Target_Good_Investment_Label`. Phase 5 trained Logistic Regression and Decision Tree Classifier models using a stratified 80/20 split with random seed 42. The rule inputs used to define the label were excluded from the predictors to avoid directly reproducing the rule.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8050 | 0.6258 | 0.4145 | 0.4987 | 0.8226 |
| Decision Tree Classifier | 0.7960 | 0.5789 | 0.4701 | 0.5189 | 0.8362 |
| Majority Baseline | 0.7660 | 0.0000 | 0.0000 | 0.0000 | N/A |

The positive class represents 23.42% of the prepared records, so accuracy should not be interpreted alone.

### Regression: five-year projected property value

The regression target is `Target_Projected_Value_After_5Y_INR_Lakhs`. Phase 5 trained Linear Regression and Random Forest Regressor models using the same fixed 80/20 split and random seed. Current property price is included because it is available at prediction time. Assumed appreciation is excluded from the general predictor set because it directly participates in the assumption-based target formula.

| Model | MAE (INR lakhs) | RMSE (INR lakhs) | R² |
|---|---:|---:|---:|
| Linear Regression | 11.5834 | 17.2580 | 0.9689 |
| Random Forest Regressor | 11.9680 | 17.8242 | 0.9669 |
| Mean Baseline | 73.3024 | 98.1463 | -0.0045 |

These metrics describe fit to one synthetic, assumption-based dataset and split. They do not establish real-world accuracy for Indian property prices, rental yields, investment quality, or future market values.

## Known limitations and future improvements

- The investment dataset is synthetic and assumption-based, not verified market data.
- The classification label is an educational rule rather than an observed investment outcome.
- The five-year value target is generated from current price and assumed appreciation, so the regression result is a scenario estimate rather than a validated market forecast.
- Results come from one fixed train/test split and have no external or temporal validation.
- The classification data is imbalanced; precision, recall, F1, ROC-AUC, and confusion matrices should be considered together.
- The project has minimum dependency versions rather than a locked dependency file.
- The local MLflow store is intended for local experimentation and is excluded from Git by default.

Potential future improvements include adding verified and time-aware market data, external validation, calibration and imbalance analysis, stronger input-domain validation, dependency locking, and a documented production deployment process.

## Credits and acknowledgments

No individual authors, external citations, or third-party project credits are documented in the inspected project files. This README therefore does not assign authorship or add unsupported acknowledgments.

## Additional project documentation

- [Phase 3 dataset validation](./reports/phase3_dataset_validation_2.md)
- [Phase 4 data preparation](./reports/phase4_data_preparation_2.md)
- [Phase 5 model evaluation](./reports/phase5_model_evaluation.md)
- [Phase 6 MLflow tracking](./reports/phase6_mlflow_tracking.md)
- [Phase 7 Streamlit application](./reports/phase7_streamlit_app.md)
