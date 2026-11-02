# Phase 7: Streamlit Web Application

## Application

The dashboard is implemented in `app.py` at the project root. It is a beginner-friendly, responsive real-estate scenario interface with:

- Project title, description, and a persistent synthetic-data disclaimer.
- Sidebar form for city, locality tier, property type, BHK, area, age, floor information, current price, distances, amenities, parking, gated community, furnishing, and occupancy status.
- Input ranges based on observed source-dataset ranges, plus floor validation.
- Predict and reset controls.
- Bordered metric cards for the synthetic classification result, probability, and five-year value estimate.
- Entered-property summary table.
- Native Streamlit bar chart comparing current price with the five-year scenario estimate.
- Friendly missing-model and invalid-input messages.

## Model integration

The app loads these existing Phase 5 pipelines with `st.cache_resource`; it does not retrain them:

- `models/phase5_logistic_regression.joblib` for the classification result and probability.
- `models/phase5_linear_regression.joblib` for the five-year value estimate.

The app reads `models/phase5_metadata.json` and checks that both saved pipelines expose the same 37 feature names and order before accepting predictions. The feature row contains the 11 numeric model inputs and one-hot indicators for all seven categorical feature groups. Current price is included because it is available at prediction time and required by the saved value model. No target, rental-yield, appreciation, projected-gain, or other future-derived field is constructed in the app.

Floor number and total floors are included because they are part of the saved model feature definition. The app validates `Floor_No <= Total_Floors` before prediction.

## Dependency and launch command

Streamlit was installed in the project virtual environment and added to `requirements.txt` as `streamlit>=1.57`.

From the project root, run:

```powershell
streamlit run app.py
```

The app uses the existing saved models under `models/`; it does not modify datasets, Phase 5 results, or MLflow runs.

## Verification performed

Executed successfully:

1. Workspace diagnostics for `app.py` and `tests/test_phase7_app.py`: no errors reported.
2. `python -m unittest discover -s tests -p 'test_phase7_app.py' -v`:
   - Saved model compatibility test passed.
   - Streamlit AppTest render-and-predict test passed.
3. Local Streamlit server smoke test using `streamlit run app.py --server.headless true --server.port 8501`:
   - Page title and dashboard heading rendered.
   - Sidebar inputs rendered.
   - Default prediction submitted successfully.
   - Classification result, probability, five-year value, summary table, disclaimer, and comparison chart were visible.
   - Server was stopped after verification.

The browser smoke test produced the default scenario result of `Does not meet synthetic criteria`, a `37.4%` classification probability, and an `INR 153.10 lakh` five-year estimate. This is only a UI smoke-test observation, not a general model claim.

## Limitations

The application presents predictions from models trained on synthetic, assumption-based data. The classification target is an educational rule and the value target is an assumption-based five-year projection. These outputs are not verified market forecasts, guaranteed investment outcomes, or financial advice. The app is local-only and was not deployed publicly.
