"""Phase 2.5: evidence-based data suitability audit.

This script does not modify either dataset. It writes only a diagnostic report.
The regression experiment is intentionally diagnostic, not a production model.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "india_housing_prices.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "india_housing_prices_cleaned.csv"
REPORT_PATH = ROOT / "reports" / "phase2_5_suitability_report.md"
REFERENCE_YEAR = 2025
RANDOM_STATE = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def fmt(value: float) -> str:
    return f"{value:,.4f}"


def main() -> None:
    raw_hash_before = sha256(RAW_PATH)
    clean_hash_before = sha256(CLEAN_PATH)
    raw = pd.read_csv(RAW_PATH)
    df = pd.read_csv(CLEAN_PATH)
    raw_hash_after = sha256(RAW_PATH)
    clean_hash_after = sha256(CLEAN_PATH)

    if len(raw) != len(df):
        raise ValueError("Raw and cleaned row counts differ; stop before suitability analysis.")

    target = "Price_in_Lakhs"
    numeric = [
        "BHK",
        "Size_in_SqFt",
        "Year_Built",
        "Floor_No",
        "Total_Floors",
        "Age_of_Property",
        "Nearby_Schools",
        "Nearby_Hospitals",
    ]
    categorical = [
        "State",
        "City",
        "Locality",
        "Property_Type",
        "Furnished_Status",
        "Public_Transport_Accessibility",
        "Parking_Space",
        "Security",
        "Amenities",
        "Facing",
        "Owner_Type",
        "Availability_Status",
    ]
    feature_columns = numeric + categorical

    y = df[target]
    X = df[feature_columns]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ]
    )
    ridge = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", Ridge(alpha=10.0)),
        ]
    )
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    dummy_predictions = dummy.predict(np.zeros((len(y_test), 1)))
    ridge.fit(X_train, y_train)
    ridge_predictions = ridge.predict(X_test)

    baseline_metrics = metrics(y_test, dummy_predictions)
    ridge_metrics = metrics(y_test, ridge_predictions)

    numeric_unique = df[numeric].nunique().sort_values()
    numeric_summary = df[numeric].describe().T
    categorical_summary = pd.DataFrame(
        {
            "unique": df[categorical].nunique(),
            "most_common_value": [df[column].value_counts().index[0] for column in categorical],
            "most_common_count": [df[column].value_counts().iloc[0] for column in categorical],
            "most_common_share": [df[column].value_counts(normalize=True).iloc[0] for column in categorical],
        }
    ).sort_values("unique")

    repeated_numeric = []
    for column in numeric:
        counts = df[column].value_counts()
        repeated_numeric.append(
            {
                "column": column,
                "unique": int(df[column].nunique()),
                "most_common_value": counts.index[0],
                "most_common_count": int(counts.iloc[0]),
                "most_common_share": float(counts.iloc[0] / len(df)),
            }
        )
    repeated_numeric_df = pd.DataFrame(repeated_numeric).sort_values("most_common_share", ascending=False)

    price_per_sqft_expected = df["Price_in_Lakhs"] / df["Size_in_SqFt"]
    price_per_sqft_consistent = (df["Price_per_SqFt"] - price_per_sqft_expected).abs() <= 0.011
    age_consistent = df["Age_of_Property"] == REFERENCE_YEAR - df["Year_Built"]
    floor_flag = df["Floor_Consistency_Check"].eq("Floor_No_exceeds_Total_Floors")

    size_price_corr = df["Size_in_SqFt"].corr(df[target])
    size_price_spearman = df[["Size_in_SqFt", target]].corr(method="spearman").iloc[0, 1]
    size_price_by_type = df.groupby("Property_Type").apply(
        lambda group: group["Size_in_SqFt"].corr(group[target]), include_groups=False
    )
    size_price_by_state = df.groupby("State").apply(
        lambda group: group["Size_in_SqFt"].corr(group[target]), include_groups=False
    )
    price_by_type = df.groupby("Property_Type")[target].agg(["count", "mean", "median", "std"])
    price_by_bhk = df.groupby("BHK")[target].agg(["count", "mean", "median", "std"])
    price_by_city = df.groupby("City")[target].agg(["count", "mean", "median", "std"])
    price_per_sqft_by_city = df.groupby("City")["Price_per_SqFt"].mean().sort_values(ascending=False)

    historical_columns = [
        column
        for column in df.columns
        if any(token in column.lower() for token in ["date", "year", "time", "sold", "sale", "listing"])
    ]
    repeated_property_candidates = int(df.duplicated(subset=["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Year_Built"]).sum())

    report = f"""# Phase 2.5: Data Suitability Check

## Executive conclusion

The cleaned file is technically usable for an educational current-price experiment, but it does not provide strong evidence for a trustworthy investment classifier or a genuine five-year property-value forecasting model.

The strongest measured warning is that the data has almost no relationship between size and price: Pearson correlation is **{size_price_corr:.4f}** and Spearman correlation is **{size_price_spearman:.4f}**. This is not a calculation mistake. The price-per-square-foot calculation is internally consistent, but it is derived from price and size. The pattern appears to be a property of the supplied data, which is consistent with the earlier provenance concerns.

## Dataset integrity and preservation

- Raw rows: **{len(raw):,}**
- Cleaned rows: **{len(df):,}**
- Raw SHA-256 unchanged during this audit: **{raw_hash_before == raw_hash_after}**
- Cleaned SHA-256 unchanged during this audit: **{clean_hash_before == clean_hash_after}**
- Raw and cleaned row counts match: **{len(raw) == len(df)}**

No dataset file was written by this script.

## 1. Variation and data-quality observations

### Numerical variation

The least-variable numerical features by number of distinct values are:

{chr(10).join(f"- `{column}`: {int(count):,} unique values; range {numeric_summary.loc[column, 'min']:,.2f} to {numeric_summary.loc[column, 'max']:,.2f}" for column, count in numeric_unique.items())}

Repeated-value inspection shows these largest single-value shares among the numerical features:

{chr(10).join(f"- `{row.column}`: value `{row.most_common_value}` appears {row.most_common_count:,} times ({row.most_common_share:.2%})" for row in repeated_numeric_df.itertuples())}

There are no constant numerical columns, but the ranges are unusually neat: BHK is 1–5, size is 500–5,000 sq ft, price is 10–500 lakhs, and price per sq ft is reported to two decimal places. These patterns are not automatically wrong, but they support treating the file as possibly simulated or heavily constrained.

### Categorical variation

{chr(10).join(f"- `{column}`: {int(row['unique']):,} unique values; most common value `{row['most_common_value']}` represents {row['most_common_share']:.2%}" for column, row in categorical_summary.iterrows())}

All locality names match `Locality_<number>`. This is artificial-looking and removes meaningful locality identity. It is evidence of generated or anonymized labels, not proof of synthetic generation.

### Relationships and consistency checks

- Size versus price: Pearson **{size_price_corr:.4f}**; Spearman **{size_price_spearman:.4f}**.
- Size-price Pearson correlations by property type: {', '.join(f'{k}: {v:.4f}' for k, v in size_price_by_type.items())}.
- Across cities, the minimum and maximum size-price correlations are **{size_price_by_state.min():.4f}** and **{size_price_by_state.max():.4f}**.
- `Price_per_SqFt` agrees with `Price_in_Lakhs / Size_in_SqFt` within a two-decimal rounding tolerance for **{int(price_per_sqft_consistent.sum()):,} / {len(df):,}** rows.
- `Age_of_Property` equals `2025 - Year_Built` for **{int(age_consistent.sum()):,} / {len(df):,}** rows.
- Existing floor flags mark **{int(floor_flag.sum()):,} / {len(df):,}** rows where `Floor_No > Total_Floors`.
- Repeated property-description combinations (State, City, Locality, type, BHK, size, and year) occur in **{repeated_property_candidates:,}** rows after the first occurrence. This is not proof of repeated observations because there is no property identifier beyond the unique ID.

The near-zero size-price relationship is therefore not caused by the price-per-square-foot calculation. That calculation is a derived consistency check, not an independent feature. The source data itself shows nearly constant price distributions across BHK and property type, and the location labels are generic. Those facts make the signal weak and the model risk high.

## 2. Diagnostic current-price regression

### Features used

The diagnostic used an 80/20 random split with `random_state=42`.

Used features:

- Numeric: BHK, size, year built, floor number, total floors, property age, nearby schools, nearby hospitals.
- Categorical: state, city, locality, property type, furnishing, transport access, parking, security, amenities, facing, owner type, availability status.

Excluded features:

- `Price_in_Lakhs`: target.
- `Price_per_SqFt`: target leakage because it is calculated from price and size.
- `ID`: identifier, not a property characteristic.
- Phase 2 diagnostic flag columns.

The model pipeline uses median/mode imputation, standard scaling for numeric features, one-hot encoding for categorical features, and Ridge regression. The naive baseline predicts the training-set mean price for every test row.

### Metrics

MAE means average absolute error in lakhs. RMSE also measures error in lakhs but penalizes large errors more heavily. R² measures improvement over the mean baseline; a value near zero means the model explains little additional variation, and a negative value is worse than the baseline.

| Diagnostic | MAE (lakhs) | RMSE (lakhs) | R² |
|---|---:|---:|---:|
| Naive mean baseline | {fmt(baseline_metrics['MAE'])} | {fmt(baseline_metrics['RMSE'])} | {fmt(baseline_metrics['R2'])} |
| Leakage-safe Ridge pipeline | {fmt(ridge_metrics['MAE'])} | {fmt(ridge_metrics['RMSE'])} | {fmt(ridge_metrics['R2'])} |

These are diagnostic holdout results, not final-model performance. A random split may overestimate performance when properties share locations or when the data is generated from simple rules. A time-based or external-market validation would be needed for stronger evidence.

## 3. Investment classification suitability

The dataset contains no genuine `Good_Investment`, profitability, return, rental-yield, or observed-investment-outcome column. Therefore, a supervised classifier cannot be trained to predict verified investment profitability from this file.

An arbitrary rule such as “price growth above 10% means good investment” would create assumed labels. A model trained on those labels would learn the rule, not discover real-world profitability.

Additional data needed for a stronger investment assessment includes:

- Monthly or annual rental income.
- Purchase price and transaction costs.
- Maintenance, tax, insurance, and vacancy costs.
- Financing terms, interest rate, and down payment.
- Verified historical resale prices.
- Holding period and expected selling costs.
- Reliable market appreciation data matched to city and time period.

A defensible educational alternative is a transparent rule-based score. For example, separately score affordability, rental yield, market appreciation scenario, location features, and data-quality warnings, then show the component scores and assumptions. This is a decision aid, not a machine-learning classifier and not financial advice.

## 4. Five-year value-estimation suitability

The property file has `Year_Built`, but it does not have sale dates, listing dates, future prices, repeated observations of the same property over time, or a five-year target. The unique IDs also mean that the file does not provide a verified property-level time series.

Therefore, this dataset alone cannot train a genuine model for actual five-year property values.

NHB RESIDEX could support a city-level appreciation assumption if compatible historical observations are obtained. The required fields are:

- City covered by the index.
- Index observation date or quarter.
- Index value.
- Beginning and ending dates that span the intended historical period.
- A clear match between the property city and the RESIDEX city definition.

For compatible observations, annualized appreciation would be calculated as:

```text
(Ending index / Beginning index) ** (1 / number_of_years) - 1
```

An illustrative scenario could then be calculated as:

```text
Estimated five-year value = Current property value * (1 + annual growth rate) ** 5
```

This becomes defensible only after verifying that the city exists in both datasets, the index dates are comparable, the period is long enough, and the index is being used as a city-level scenario rather than a property-specific prediction. RESIDEX is not included in the current project files, so no appreciation rate was calculated here.

## Recommendation

Proceed cautiously with current-price modeling only as an educational diagnostic. Do not present it as a validated market-price predictor until the data source is improved or independently validated.

For the investment and five-year parts, use a transparent scenario/scoring framework and clearly display the missing inputs. Do not train a `Good_Investment` classifier or a five-year regression model from this dataset alone.

No final model, production training pipeline, MLflow configuration, or Streamlit application was created in Phase 2.5.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Report: {REPORT_PATH}")
    print(f"Baseline: {baseline_metrics}")
    print(f"Ridge: {ridge_metrics}")
    print(f"Size-price Pearson: {size_price_corr:.6f}")
    print(f"Price-per-square-foot consistent: {int(price_per_sqft_consistent.sum()):,}/{len(df):,}")
    print(f"Age consistent: {int(age_consistent.sum()):,}/{len(df):,}")
    print(f"Floor flags: {int(floor_flag.sum()):,}/{len(df):,}")


if __name__ == "__main__":
    main()
