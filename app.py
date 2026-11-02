"""Streamlit dashboard for the synthetic Real Estate Investment Advisor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "phase5_logistic_regression.joblib"
REGRESSION_MODEL_PATH = MODELS_DIR / "phase5_linear_regression.joblib"
METADATA_PATH = MODELS_DIR / "phase5_metadata.json"

NUMERIC_FEATURES = [
    "BHK",
    "Area_SqFt",
    "Property_Age_Years",
    "Floor_No",
    "Total_Floors",
    "Distance_to_CBD_km",
    "Distance_to_Metro_km",
    "Distance_to_School_km",
    "Distance_to_Hospital_km",
    "Amenities_Score_1_to_10",
    "Price_INR_Lakhs",
]
CATEGORICAL_FEATURES = [
    "City",
    "Locality_Tier",
    "Property_Type",
    "Parking_Available",
    "Gated_Community",
    "Furnishing_Status",
    "Occupancy_Status",
]
DEFAULTS: dict[str, Any] = {
    "city": "Bengaluru",
    "locality_tier": "Established",
    "property_type": "Apartment",
    "bhk": 2,
    "area_sqft": 1000,
    "property_age_years": 5,
    "floor_no": 1,
    "total_floors": 5,
    "price_inr_lakhs": 100.0,
    "distance_cbd_km": 5.0,
    "distance_metro_km": 2.0,
    "distance_school_km": 1.0,
    "distance_hospital_km": 2.0,
    "amenities_score": 7,
    "parking": "Yes",
    "gated_community": "Yes",
    "furnishing_status": "Semi-Furnished",
    "occupancy_status": "Tenant Occupied",
}


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> tuple[Any, Any, list[str]]:
    """Load the Phase 5 pipelines and their exact feature order once."""
    required = [CLASSIFICATION_MODEL_PATH, REGRESSION_MODEL_PATH, METADATA_PATH]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s): "
            + ", ".join(missing)
            + ". Run `c:/Users/Admin/OneDrive/Desktop/Real_Estate_Investment_Advisor/.venv/Scripts/python.exe src/phase5_train_models.py` first."
        )
    classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
    regression_model = joblib.load(REGRESSION_MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    feature_columns = metadata.get("feature_columns", [])
    if not feature_columns:
        raise ValueError("Phase 5 metadata does not contain feature_columns.")
    for model_name, model in {
        "classification": classification_model,
        "regression": regression_model,
    }.items():
        model_features = list(getattr(model, "feature_names_in_", []))
        if model_features != feature_columns:
            raise ValueError(
                f"{model_name.title()} model feature order does not match phase5_metadata.json."
            )
    return classification_model, regression_model, feature_columns


def build_feature_row(values: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    """Create one model-compatible row in the saved pipeline's exact order."""
    row = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    numeric_values = {
        "BHK": values["bhk"],
        "Area_SqFt": values["area_sqft"],
        "Property_Age_Years": values["property_age_years"],
        "Floor_No": values["floor_no"],
        "Total_Floors": values["total_floors"],
        "Distance_to_CBD_km": values["distance_cbd_km"],
        "Distance_to_Metro_km": values["distance_metro_km"],
        "Distance_to_School_km": values["distance_school_km"],
        "Distance_to_Hospital_km": values["distance_hospital_km"],
        "Amenities_Score_1_to_10": values["amenities_score"],
        "Price_INR_Lakhs": values["price_inr_lakhs"],
    }
    for column, value in numeric_values.items():
        row.at[0, column] = value

    for feature_name, value_key in {
        "City": "city",
        "Locality_Tier": "locality_tier",
        "Property_Type": "property_type",
        "Parking_Available": "parking",
        "Gated_Community": "gated_community",
        "Furnishing_Status": "furnishing_status",
        "Occupancy_Status": "occupancy_status",
    }.items():
        encoded_column = f"{feature_name}__{values[value_key]}"
        if encoded_column not in feature_columns:
            raise ValueError(f"Unsupported category for saved model: {encoded_column}")
        row.at[0, encoded_column] = 1.0
    return row.loc[:, feature_columns]


def reset_inputs() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.pop("prediction_result", None)


def format_inr_lakhs(value: float) -> str:
    return f"INR {value:,.2f} lakh"


def format_inr(value_in_lakhs: float) -> str:
    return f"INR {value_in_lakhs * 100_000:,.0f}"


def render_prediction(values: dict[str, Any], classification_model: Any, regression_model: Any, feature_columns: list[str]) -> None:
    try:
        features = build_feature_row(values, feature_columns)
        predicted_label = int(classification_model.predict(features)[0])
        probability = None
        if hasattr(classification_model, "predict_proba"):
            probability = float(classification_model.predict_proba(features)[0, 1])
        projected_value = float(regression_model.predict(features)[0])
    except (ValueError, TypeError, KeyError) as error:
        st.error(f"The property inputs could not be prepared for the saved models: {error}", icon=":material/error:")
        return

    st.session_state.prediction_result = {
        "label": predicted_label,
        "probability": probability,
        "projected_value": projected_value,
        "values": values,
    }


def main() -> None:
    st.set_page_config(
        page_title="Real Estate Investment Advisor",
        page_icon=":material/home_work:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)

    st.title("Real Estate Investment Advisor")
    st.caption("A synthetic scenario dashboard for property profitability and five-year value estimation")
    st.warning(
        "Synthetic-data disclaimer: predictions are based on assumption-driven training data and are for educational demonstration only. They are not verified market forecasts or financial advice.",
        icon=":material/info:",
    )

    try:
        classification_model, regression_model, feature_columns = load_model_bundle()
    except (FileNotFoundError, ValueError, OSError) as error:
        st.error(str(error), icon=":material/error:")
        st.stop()

    with st.sidebar:
        st.header("Property inputs")
        st.caption("Enter the current information available at decision time.")
        if st.button("Reset inputs", on_click=reset_inputs, width="stretch"):
            st.rerun()
        with st.form("property_inputs", border=True):
            city = st.selectbox("City", ["Ahmedabad", "Bengaluru", "Chennai", "Delhi", "Hyderabad", "Jaipur", "Kochi", "Kolkata", "Mumbai", "Pune"], key="city")
            locality_tier = st.selectbox("Locality tier", ["Developing", "Established", "Prime"], key="locality_tier")
            property_type = st.selectbox("Property type", ["Apartment", "Independent House", "Villa"], key="property_type")
            bhk = st.number_input("BHK", min_value=1, max_value=5, step=1, key="bhk")
            area_sqft = st.number_input("Area (sq ft)", min_value=328, max_value=2664, step=1, key="area_sqft")
            property_age_years = st.number_input("Property age (years)", min_value=0, max_value=30, step=1, key="property_age_years")
            floor_no = st.number_input("Floor number", min_value=0, max_value=50, step=1, key="floor_no")
            total_floors = st.number_input("Total floors", min_value=0, max_value=50, step=1, key="total_floors")
            price_inr_lakhs = st.number_input("Current price (INR lakhs)", min_value=15.37, max_value=670.70, step=1.0, key="price_inr_lakhs")
            st.divider()
            distance_cbd_km = st.number_input("Distance to city centre (km)", min_value=0.5, max_value=30.0, step=0.1, key="distance_cbd_km")
            distance_metro_km = st.number_input("Distance to metro (km)", min_value=0.1, max_value=12.0, step=0.1, key="distance_metro_km")
            distance_school_km = st.number_input("Distance to school (km)", min_value=0.0, max_value=8.0, step=0.1, key="distance_school_km")
            distance_hospital_km = st.number_input("Distance to hospital (km)", min_value=0.0, max_value=9.6, step=0.1, key="distance_hospital_km")
            amenities_score = st.slider("Amenities score", min_value=1, max_value=10, step=1, key="amenities_score")
            parking = st.selectbox("Parking available", ["No", "Yes"], key="parking")
            gated_community = st.selectbox("Gated community", ["No", "Yes"], key="gated_community")
            furnishing_status = st.selectbox("Furnishing status", ["Furnished", "Semi-Furnished", "Unfurnished"], key="furnishing_status")
            occupancy_status = st.selectbox("Occupancy status", ["Owner Occupied", "Tenant Occupied", "Vacant"], key="occupancy_status")
            submitted = st.form_submit_button("Predict property outlook", type="primary", width="stretch")

    if submitted:
        if floor_no > total_floors:
            st.error("Floor number cannot be greater than total floors.", icon=":material/error:")
        else:
            render_prediction(
                {
                    "city": city,
                    "locality_tier": locality_tier,
                    "property_type": property_type,
                    "bhk": bhk,
                    "area_sqft": area_sqft,
                    "property_age_years": property_age_years,
                    "floor_no": floor_no,
                    "total_floors": total_floors,
                    "price_inr_lakhs": price_inr_lakhs,
                    "distance_cbd_km": distance_cbd_km,
                    "distance_metro_km": distance_metro_km,
                    "distance_school_km": distance_school_km,
                    "distance_hospital_km": distance_hospital_km,
                    "amenities_score": amenities_score,
                    "parking": parking,
                    "gated_community": gated_community,
                    "furnishing_status": furnishing_status,
                    "occupancy_status": occupancy_status,
                },
                classification_model,
                regression_model,
                feature_columns,
            )

    result = st.session_state.get("prediction_result")
    if result is None:
        with st.container(border=True):
            st.subheader("Ready for a scenario")
            st.write("Complete the property inputs in the sidebar, then select **Predict property outlook** to view the two model estimates.")
        return

    label_text = "Meets synthetic criteria" if result["label"] == 1 else "Does not meet synthetic criteria"
    probability_text = "Not available"
    if result["probability"] is not None:
        probability_text = f"{result['probability']:.1%} model probability"
    projected_value = result["projected_value"]
    current_price = float(result["values"]["price_inr_lakhs"])

    with st.container(horizontal=True):
        st.metric("Synthetic investment label", label_text, border=True)
        st.metric("Five-year value estimate", format_inr_lakhs(projected_value), border=True)
        st.metric("Classification probability", probability_text, border=True)

    left, right = st.columns([1.05, 1.4])
    with left:
        with st.container(border=True):
            st.subheader("Scenario summary")
            summary = pd.DataFrame(
                {
                    "Property detail": [
                        "City", "Locality tier", "Property type", "BHK", "Area", "Current price", "Amenities",
                    ],
                    "Entered value": [
                        result["values"]["city"], result["values"]["locality_tier"], result["values"]["property_type"],
                        f"{result['values']['bhk']} BHK", f"{result['values']['area_sqft']:,} sq ft",
                        format_inr_lakhs(current_price), f"{result['values']['amenities_score']}/10",
                    ],
                }
            )
            st.table(summary)
    with right:
        with st.container(border=True):
            st.subheader("Current versus five-year scenario")
            chart_data = pd.DataFrame(
                {"Value (INR lakhs)": [current_price, projected_value]},
                index=["Current price", "Five-year estimate"],
            )
            st.bar_chart(chart_data, horizontal=True, alt="Current property price compared with the five-year scenario estimate")
            st.caption(f"Five-year estimate: {format_inr(projected_value)} ({format_inr_lakhs(projected_value)}).")

    st.info(
        "Interpretation: the classification model predicts the synthetic label pattern, while the value model follows an assumption-based five-year target. Neither result guarantees profitability or future market performance.",
        icon=":material/insights:",
    )


if __name__ == "__main__":
    main()
