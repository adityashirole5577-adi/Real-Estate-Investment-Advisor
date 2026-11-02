# Phase 6: MLflow Experiment Tracking

## Tracking configuration

- Tracking store: `mlruns/` using MLflow's local file backend
- Experiment: **Real_Estate_Investment_Advisor**
- Tracking URI: `file:///C:/Users/Admin/OneDrive/Desktop/Real_Estate_Investment_Advisor/mlruns`
- MLflow file-store opt-in: `MLFLOW_ALLOW_FILE_STORE=true` (required by MLflow 3.x for this local backend)
- Source metrics: `reports/phase5_model_evaluation.md`
- Existing saved pipelines were loaded from `models/`; no model was retrained.

The models were trained on synthetic, assumption-based data. MLflow records experiment provenance and reproducibility metadata; it does not validate these models as real-world property investment predictors.

## Runs and verification

| Run | Task | Run ID | Logged metrics | Logged artifact entries |
|---|---|---|---:|---:|
| Logistic Regression | classification | `f7835fb67b0d4783a81701eeb12dde50` | 5 | 20 | 13 |
| Decision Tree Classifier | classification | `4fdacff20ee84b39a5308249ac180a38` | 5 | 19 | 13 |
| Majority Baseline | classification | `c6329d2ed1bf420ab5b02dea830fcda9` | 4 | 9 | 13 |
| Linear Regression | regression | `ca5f73eeb6934fa986918b7eeaf8a135` | 3 | 11 | 14 |
| Random Forest Regressor | regression | `fa4bf14a084947e1807e29d7ee8bddda` | 3 | 24 | 14 |
| Mean Baseline | regression | `3d11532cbfd04532b568086559fbd18a` | 3 | 9 | 14 |

All six runs finished with status **FINISHED**. The MLflow client successfully retrieved every run and its logged metrics/artifacts. Each model run includes the corresponding saved pipeline, the Phase 5 evaluation report, the Phase 5 metadata feature/target definition, and the relevant evaluation charts. Baseline metrics were logged for the majority-class and mean-prediction baselines; the majority baseline has no ROC-AUC because it produces one prediction class, so ROC-AUC was not logged for that metric.

## Exact metrics logged

- **Logistic Regression**: `accuracy`=0.805, `precision`=0.6258, `recall`=0.4145, `f1`=0.4987, `roc_auc`=0.8226
- **Decision Tree Classifier**: `accuracy`=0.796, `precision`=0.5789, `recall`=0.4701, `f1`=0.5189, `roc_auc`=0.8362
- **Majority Baseline**: `accuracy`=0.766, `precision`=0.0, `recall`=0.0, `f1`=0.0, `roc_auc`=N/A
- **Linear Regression**: `mae`=11.5834, `rmse`=17.258, `r2`=0.9689
- **Random Forest Regressor**: `mae`=11.968, `rmse`=17.8242, `r2`=0.9669
- **Mean Baseline**: `mae`=73.3024, `rmse`=98.1463, `r2`=-0.0045

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
mlflow ui --backend-store-uri .\mlruns --port 5000
```

Then open `http://127.0.0.1:5000` and select the `Real_Estate_Investment_Advisor` experiment. No cloud account or external tracking server is required.

## Limitations

The tracked models and metrics reflect one fixed split of a synthetic dataset. The classification target is an educational rule, and the regression target is an assumption-based five-year projection. Strong metrics here must not be interpreted as evidence of actual Indian property-price or investment-quality prediction. No raw or prepared dataset was modified, and no Streamlit application was created.
