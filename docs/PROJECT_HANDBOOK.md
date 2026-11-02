# Real Estate Investment Advisor — Project Handbook
## Predicting Property Profitability & Future Value

**Live Streamlit application:** [Open the deployed app](https://real-estate-investment-advisor-8z7ppsrhfksyfdihc3f5e4.streamlit.app/)  
**GitHub repository:** [Open the source project](https://github.com/adityashirole5577-adi/Real-Estate-Investment-Advisor)

**Domain:** Real Estate / Investment / Financial Analytics  
**Skills:** Python, ML, EDA, data analysis, feature engineering, regression, classification, model evaluation, feature scaling, Streamlit and MLflow.

> **Important:** The 5,000-row investment-model dataset is synthetic and assumption-based. The project demonstrates an ML workflow; it is not a verified market forecast, professional property valuation or financial advice.

## Contents
1. Project Overview and Skills Takeaway · 2. Problem Statement and Objectives · 3. Business Use Cases · 4. Project Approach · 5. Data Flow and Architecture · 6. Dataset and Features · 7. Target Variables · 8. EDA Guide · 9. Model Development · 10. Results · 11. MLflow · 12. Streamlit App · 13. How to Run · 14. Evaluation Metrics · 15. Deliverables · 16. Limitations · 17. References and Links

---

## 1. Project Overview and Skills Takeaway

The Real Estate Investment Advisor is an end-to-end ML demonstration with two prediction tasks:
- **Classification:** determine whether a property matches the project-defined `Good_Investment_Label`.
- **Regression:** estimate the property's projected value after five years using the project's prepared target.

| Skill area | Concepts used |
|---|---|
| Programming / data | Python, Pandas, NumPy, validation and transformation |
| Analytics | EDA, descriptive statistics, distributions, correlations and charts |
| Feature engineering | Derived variables, categorical encoding and target construction |
| Machine learning | Logistic Regression, Decision Tree, Linear Regression and Random Forest |
| Evaluation | ROC-AUC, MAE, RMSE, R² and baseline comparison |
| Deployment | Joblib, Streamlit and MLflow |

## 2. Problem Statement and Objectives

Property decisions involve asking price, size, rental potential, appreciation assumptions, location, connectivity and amenities. The project demonstrates a repeatable analysis workflow and an interactive interface to:
1. Classify whether a property meets the project's Good Investment rule (1/0).
2. Estimate a property's value after five years.
3. Inspect, explore and prepare data for modelling.
4. Compare models against metrics and baselines.
5. Track experiments with MLflow and deploy an interactive Streamlit app.

## 3. Business Use Cases

| Potential user | Possible use | Qualification |
|---|---|---|
| Property buyers | Compare selected property characteristics consistently. | Requires independent due diligence and verified market data. |
| Real estate analysts | Use the workflow as a screening / EDA starting point. | Current investment data is synthetic. |
| Property platforms | Demonstrate prediction outputs in a listing workflow. | Production use requires reliable data, monitoring and validation. |
| Investment advisers | Explain a repeatable framework for rental and location factors. | Not a substitute for professional advice. |

## 4. Project Approach and Phase-wise Methodology

| Phase | Work completed |
|---|---|
| 1 — Setup | Project folders, environment, requirements and data organization. |
| 2 — Initial EDA | Inspected the housing dataset, columns and data quality. |
| 2.5 — Suitability | Reviewed whether housing data suited the intended investment task. |
| 3 — Investment data validation | Explored distributions / relationships and validated synthetic data. |
| 4 — Preparation | Created derived features, encoded categories and prepared labels / model table. |
| 5 — Modelling | Trained classification / regression models, compared baselines and saved artifacts. |
| 6 — MLflow | Recorded experiment runs and metrics in the local tracking store. |
| 7 — Streamlit | Loaded saved pipelines in an interactive prediction interface. |

**Key concepts:** data inspection (shape, types, missing values, duplicates); feature engineering (derive useful inputs); categorical encoding (turn text categories into numeric features); pipelines (keep preprocessing and model together); persistence (save fitted pipelines for reuse without retraining).

## 5. Data Flow and System Architecture

```text
Data sources → Inspection / EDA (2–3) → Feature engineering / label (4)
        → Classification + regression (5) → Saved model pipelines
        → Streamlit app (7) → Class / probability + five-year value
                          ↘ MLflow experiment tracking (6)
```

The app collects a property profile, prepares the expected feature row, loads the saved models and displays results. MLflow supports development tracking; it is not the app interface.

## 6. Dataset and Feature Understanding

There are two separate dataset tracks; do not confuse the housing-price data explored earlier with the synthetic data used to train the final investment models.

| Dataset artifact | Known details | Role |
|---|---|---|
| `data/raw/india_housing_prices.csv` | About 250,000 rows. | Initial exploration / understanding. |
| `data/processed/india_housing_prices_cleaned.csv` | Cleaned housing counterpart. | Earlier cleaning / EDA. |
| `data/raw/india_real_estate_investment_synthetic_dataset.csv` | 5,000 rows × 28 columns. | Main investment-model source. |
| `data/processed/real_estate_phase4_prepared*.csv` | Prepared table, 5,000 rows × 42 columns. | Model-ready data and targets. |

### Investment variables

| Variable | Meaning |
|---|---|
| Property price (INR lakh) | Property price in lakh units. |
| Net rental yield (%) | Net rental income as a percentage measure. |
| Assumed annual appreciation (%) | Scenario assumption used in target construction. |
| Distance to CBD (km) | Distance to central business district. |
| Amenities score (1–10) | Numeric amenities summary. |
| Five-year future value | Assumption-based regression target. |

### Housing field guide from the project brief

Confirm actual CSV headers before coding. These are housing-data fields and are not necessarily present in the synthetic investment dataset.

| Field | Description |
|---|---|
| ID; State; City; Locality | Identifier and property geography. |
| Property_Type; BHK | Property category and bedrooms/hall/kitchen. |
| Size_in_SqFt; Price_in_Lakhs; Price_per_SqFt | Area, asking/listing price and price divided by area. |
| Year_Built; Age_of_Property | Construction year and age. |
| Furnished_Status; Floor_No; Total_Floors | Furnishing and floor details. |
| Nearby_Schools; Nearby_Hospitals | Nearby education / medical facilities. |
| Public_Transport_Accessibility | Bus / metro / train access. |
| Parking_Space; Security; Amenities | Parking, security and available facilities. |
| Facing; Owner_Type; Availability_Status | Direction, owner category and listing status. |

## 7. Target Variables and Label Logic

### Classification — `Good_Investment_Label`

A property is labelled **1 only if all four conditions** are true; otherwise it is labelled 0.

| Condition | Threshold |
|---|---:|
| `Net_Rental_Yield_Pct` | ≥ 3 |
| `Assumed_Annual_Appreciation_Rate_Pct` | ≥ 7.5 |
| `Distance_to_CBD_km` | ≤ 15 |
| `Amenities_Score_1_to_10` | ≥ 4 |

The classifier learns this rule-generated label; it does not independently establish profitability.

### Regression — five-year value

The target is `Target_Projected_Value_After_5Y_INR_Lakhs`. It is generated from the project's assumed appreciation logic, not an observed future sale price.

## 8. Exploratory Data Analysis (EDA) Guide

These 20 questions follow the supplied project brief. Use them only where the chosen dataset has the required columns. The price-per-square-foot, schools, hospitals, furnishing, facing, parking and ownership questions belong to the housing-data field guide above.

**Price and size (1–5)**
1. What is the distribution of property prices?
2. What is the distribution of property sizes?
3. How does price per sq ft vary by property type?
4. Is there a relationship between property size and price?
5. Are there outliers in price per sq ft or property size?

**Location (6–10)**
6. What is the average price per sq ft by state?
7. What is the average property price by city?
8. What is the median age of properties by locality?
9. How is BHK distributed across cities?
10. What are the price trends for the top 5 most expensive localities?

**Feature relationships (11–15)**
11. How are numeric features correlated with each other?
12. How do nearby schools relate to price per sq ft?
13. How do nearby hospitals relate to price per sq ft?
14. How does price vary by furnished status?
15. How does price per sq ft vary by property facing direction?

**Investment / amenities / ownership (16–20)**
16. How many properties belong to each owner type?
17. How many properties are available under each availability status?
18. Does parking space affect property price?
19. How do amenities affect price per sq ft?
20. How does public transport accessibility relate to price per sq ft or investment potential?

**Additional synthetic-data EDA:** inspect property price, net rental yield and assumed appreciation distributions; compare amenities / CBD distance with the label; inspect class balance and the four label thresholds; review correlations and price/value relationships. Histograms show distributions, bar charts compare categories, scatter plots show two-variable patterns, and heatmaps show numeric correlations. Correlation does not prove causation.

## 9. Model Development and Evaluation

| Task | Model | Role / metric |
|---|---|---|
| Classification | Logistic Regression | Positive-class probability; ROC-AUC. |
| Classification | Decision Tree Classifier | Rule-based splits; ROC-AUC. |
| Regression | Linear Regression | Linear numeric estimate; MAE / R². |
| Regression | Random Forest Regressor | Ensemble of trees; MAE / R². |

The project also saves a majority-class baseline and a mean-value baseline. Saved files in `models/` include the four model pipelines, `phase5_majority_baseline.joblib`, `phase5_mean_baseline.joblib` and `phase5_metadata.json`.

## 10. Results and Interpretation

Recorded Phase 5 results on the synthetic evaluation split:

| Classification model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8050 | 0.6258 | 0.4145 | 0.4987 | 0.8226 |
| Decision Tree Classifier | 0.7960 | 0.5789 | 0.4701 | 0.5189 | 0.8362 |
| Majority Baseline | 0.7660 | 0.0000 | 0.0000 | 0.0000 | N/A |

The positive class is 23.42% of prepared records, so accuracy should not be used alone. Decision Tree had the higher recorded ROC-AUC in this run.

| Regression model | MAE (INR lakh) | RMSE (INR lakh) | R² |
|---|---:|---:|---:|
| Linear Regression | 11.5834 | 17.2580 | 0.9689 |
| Random Forest Regressor | 11.9680 | 17.8242 | 0.9669 |
| Mean Baseline | 73.3024 | 98.1463 | -0.0045 |

MAE is average absolute prediction error; 11.5834 lakh is about 11.58 lakh on that evaluation sample. Linear Regression had the lower recorded MAE and higher R² in this run. These metrics do not prove real-world forecast accuracy.

## 11. MLflow Experiment Tracking

MLflow records experiment/run identifiers, parameters, metrics and logged artifacts. The local project tracking store is `mlruns/`; the project history records six completed runs.

Start the local UI from the project root in PowerShell:
```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
mlflow ui --backend-store-uri ./mlruns
```
Then open `http://localhost:5000`. This local UI is separate from the public Streamlit app.

## 12. Streamlit Application

The user interface is `app.py`. It loads the saved classification and regression pipelines; it does not retrain models for every interaction. The flow is: user inputs → validation / feature row → classification and probability → five-year value estimate → displayed summary.

**Public application:** [https://real-estate-investment-advisor-8z7ppsrhfksyfdihc3f5e4.streamlit.app/](https://real-estate-investment-advisor-8z7ppsrhfksyfdihc3f5e4.streamlit.app/)

For local use, run `streamlit run app.py`; the local address is normally `http://localhost:8501` and is only available on the computer running the app.

## 13. How to Run the Project

Windows PowerShell, from the project folder:
```powershell
Set-Location "C:\Users\Admin\OneDrive\Desktop\Real_Estate_Investment_Advisor"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```
Run tests with:
```powershell
python -m pytest -q
```
The project environment was checked with Python 3.13.5; `requirements.txt` pins package versions used in that environment.

## 14. Project Evaluation Metrics

| Area | What to check |
|---|---|
| Data handling | Structure, missing values, duplicates, encoding and engineered features. |
| Classification | ROC-AUC, precision, recall, F1 and confusion matrix. |
| Regression | MAE, R², actual-vs-predicted values and residuals. |
| Baselines | Improvement over simple baseline predictors. |
| Deployment | App runs, validates input and loads saved pipelines. |
| Tracking | MLflow runs / metrics in local UI. |
| Reproducibility | Data, models, scripts, requirements and instructions stay aligned. |

## 15. Deliverables and Repository Structure

**Deliverables:** cleaned/prepared datasets; EDA, validation, preparation and training scripts; saved classification/regression models; Phase 5 evaluation report; Phase 6 MLflow notes; Streamlit app, Phase 7 report and tests; public GitHub repository and deployed application.

```text
Real_Estate_Investment_Advisor/
├── app.py                 # Streamlit app
├── README.md
├── requirements.txt
├── data/                  # raw and processed datasets
├── models/                # saved model pipelines and metadata
├── reports/               # phase reports and figures
├── src/                   # EDA, preparation, training and tracking scripts
└── tests/                 # app tests
```
The local `mlruns/` store and `.venv/` environment are excluded from Git by `.gitignore`.

## 16. Limitations, Risks and Responsible Use

1. The investment dataset is synthetic and does not establish real market patterns.
2. The classifier learns a label generated from four thresholds, not observed investment outcomes.
3. The five-year target depends on assumed appreciation; actual growth can differ.
4. Asking prices / generated values are not verified transaction prices or realized returns.
5. Real investment analysis may require taxes, registration, maintenance, vacancy, financing, insurance, legal-title checks, liquidity and local demand.
6. A real product would need verified data, time/location-based validation, monitoring, calibration, uncertainty ranges and professional review.

> **Use disclaimer:** This is an educational demonstration. Do not use its outputs as a stand-alone basis to buy, sell, finance or value real estate. It is not financial advice or a professional valuation.

## 17. References and Project Links

- **Live Streamlit application:** [Open the deployed app](https://real-estate-investment-advisor-8z7ppsrhfksyfdihc3f5e4.streamlit.app/)
- **GitHub repository:** [Open the source project](https://github.com/adityashirole5577-adi/Real-Estate-Investment-Advisor)
- **Local Streamlit:** `http://localhost:8501` while running on the development computer.
- **Local MLflow:** `http://localhost:5000` while the MLflow server is running.
