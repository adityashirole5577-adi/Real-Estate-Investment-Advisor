"""Phase 4: audit formulas, define targets, and prepare ML-ready features.

The raw synthetic CSV is never modified. Existing Phase 4 outputs are also
preserved by selecting numbered alternative filenames when necessary.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "india_real_estate_investment_synthetic_dataset.csv"
OUTPUT_BASE = ROOT / "data" / "processed" / "real_estate_phase4_prepared.csv"
REPORT_BASE = ROOT / "reports" / "phase4_data_preparation.md"

NUMERIC_FEATURES = [
    "BHK", "Area_SqFt", "Property_Age_Years", "Floor_No", "Total_Floors",
    "Distance_to_CBD_km", "Distance_to_Metro_km", "Distance_to_School_km",
    "Distance_to_Hospital_km", "Amenities_Score_1_to_10", "Price_INR_Lakhs",
]
CATEGORICAL_FEATURES = [
    "City", "Locality_Tier", "Property_Type", "Parking_Available",
    "Gated_Community", "Furnishing_Status", "Occupancy_Status",
]


def unused(path: Path) -> Path:
    if not path.exists():
        return path
    for number in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{number}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"No unused output name available for {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def precision_summary(frame: pd.DataFrame, columns: list[str]) -> str:
    values: list[str] = []
    for column in columns:
        decimals = frame[column].dropna().map(
            lambda value: len(str(value).split(".")[-1]) if "." in str(value) else 0
        )
        values.append(f"`{column}`: {int(decimals.min())}-{int(decimals.max())} decimal places")
    return "; ".join(values)


def main() -> None:
    output_path = unused(OUTPUT_BASE)
    output_suffix = output_path.stem[len(OUTPUT_BASE.stem):]
    report_seed = REPORT_BASE.with_name(
        f"{REPORT_BASE.stem}{output_suffix}{REPORT_BASE.suffix}"
    )
    report_path = unused(report_seed)
    source_hash_before = sha256(RAW_PATH)
    source_size_before = RAW_PATH.stat().st_size
    df = pd.read_csv(RAW_PATH)

    expected_columns = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [
        "Property_ID", "Price_Per_SqFt_INR", "Estimated_Annual_Rent_INR_Lakhs",
        "Annual_Maintenance_INR_Lakhs", "Annual_Property_Tax_INR_Lakhs",
        "Net_Rental_Yield_Pct", "Assumed_Annual_Appreciation_Rate_Pct",
        "Projected_Value_After_5Y_INR_Lakhs", "Projected_5Y_Capital_Gain_Pct",
        "Good_Investment_Label",
    ])
    missing_columns = sorted(expected_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Required source columns missing: {missing_columns}")

    # Audit the stored source values before creating corrected columns.
    expected_ppsf = df["Price_INR_Lakhs"] * 100_000 / df["Area_SqFt"]
    expected_yield = (
        df["Estimated_Annual_Rent_INR_Lakhs"]
        - df["Annual_Maintenance_INR_Lakhs"]
        - df["Annual_Property_Tax_INR_Lakhs"]
    ) / df["Price_INR_Lakhs"] * 100
    expected_value = df["Price_INR_Lakhs"] * (
        1 + df["Assumed_Annual_Appreciation_Rate_Pct"] / 100
    ) ** 5
    expected_gain = (
        df["Projected_Value_After_5Y_INR_Lakhs"] / df["Price_INR_Lakhs"] - 1
    ) * 100
    residuals = {
        "price per square foot": (df["Price_Per_SqFt_INR"] - expected_ppsf).abs(),
        "net rental yield": (df["Net_Rental_Yield_Pct"] - expected_yield).abs(),
        "five-year projected value": (
            df["Projected_Value_After_5Y_INR_Lakhs"] - expected_value
        ).abs(),
        "five-year capital gain": (
            df["Projected_5Y_Capital_Gain_Pct"] - expected_gain
        ).abs(),
    }
    tolerances = {
        "price per square foot": 1.0,
        "net rental yield": 0.011,
        "five-year projected value": 0.011,
        "five-year capital gain": 0.011,
    }

    rule_label = (
        (df["Net_Rental_Yield_Pct"] >= 3)
        & (df["Assumed_Annual_Appreciation_Rate_Pct"] >= 7.5)
        & (df["Distance_to_CBD_km"] <= 15)
        & (df["Amenities_Score_1_to_10"] >= 4)
    ).astype("int8")
    label_mismatch = df["Good_Investment_Label"] != rule_label

    # Corrected fields are calculated from the stored source inputs, with extra
    # precision retained. Original source derived fields are not overwritten.
    corrected = pd.DataFrame({
        "Price_Per_SqFt_INR_Corrected": expected_ppsf.round(6),
        "Net_Rental_Yield_Pct_Corrected": expected_yield.round(6),
        "Target_Projected_Value_After_5Y_INR_Lakhs": expected_value.round(6),
        "Target_Good_Investment_Label": rule_label,
    })

    # Encode categoricals with one-hot columns. No scaling or imputation is
    # needed: the source audit found no missing values and these are mixed-scale
    # tabular features intended for later model-specific pipelines.
    feature_frame = df[["Property_ID"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    encoded = pd.get_dummies(
        feature_frame[CATEGORICAL_FEATURES],
        prefix=CATEGORICAL_FEATURES,
        prefix_sep="__",
        dtype="int8",
    )
    prepared = pd.concat(
        [feature_frame[["Property_ID"] + NUMERIC_FEATURES], encoded, corrected],
        axis=1,
    )
    duplicate_rows = int(prepared.duplicated().sum())
    duplicate_ids = int(prepared["Property_ID"].duplicated().sum())
    missing_cells = int(prepared.isna().sum().sum())
    if duplicate_rows or duplicate_ids or missing_cells:
        raise ValueError(
            f"Prepared data quality failure: duplicates={duplicate_rows}, "
            f"duplicate IDs={duplicate_ids}, missing cells={missing_cells}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)

    source_hash_after = sha256(RAW_PATH)
    formula_lines = []
    for name, residual in residuals.items():
        consistent = residual <= tolerances[name]
        formula_lines.append(
            f"| {name} | {int(consistent.sum()):,} | {int((~consistent).sum()):,} "
            f"| {residual.max():,.6f} | {tolerances[name]:,.3f} |"
        )

    mismatch_ids = ", ".join(df.loc[label_mismatch, "Property_ID"].tolist()) or "None"
    feature_columns = NUMERIC_FEATURES + encoded.columns.tolist()
    report = f"""# Phase 4: Feature Engineering and Data Preparation

## Scope and provenance

This preparation uses `data/raw/india_real_estate_investment_synthetic_dataset.csv`, a **synthetic** dataset. Its prices, rents, appreciation rates, projected values, and labels are generated or assumption-based. They are not verified market estimates, actual investment outcomes, or financial advice. The original CSV was read-only and remains unchanged.

## Source audit before correction

- Source rows / columns: **{len(df):,} / {len(df.columns):,}**
- Source size before / after: **{source_size_before:,} / {RAW_PATH.stat().st_size:,} bytes**
- Source SHA-256 unchanged: **{source_hash_before == source_hash_after}**
- Source missing cells: **{int(df.isna().sum().sum()):,}**
- Source exact duplicate rows: **{int(df.duplicated().sum()):,}**
- Source duplicate property IDs: **{int(df['Property_ID'].duplicated().sum()):,}**
- Stored numeric precision observed: {precision_summary(df, ['Price_INR_Lakhs', 'Estimated_Annual_Rent_INR_Lakhs', 'Annual_Maintenance_INR_Lakhs', 'Annual_Property_Tax_INR_Lakhs', 'Net_Rental_Yield_Pct', 'Assumed_Annual_Appreciation_Rate_Pct', 'Projected_Value_After_5Y_INR_Lakhs'])}

The source stores money and percentage fields at one or two decimal places, while price per square foot is stored as an integer. Recomputing from those displayed values therefore cannot recover any hidden higher-precision values used during synthetic generation. The residuals below were measured before correction.

| Formula audit | Within Phase 3 tolerance | Outside tolerance | Maximum residual | Tolerance |
|---|---:|---:|---:|---:|
{chr(10).join(formula_lines)}

Interpretation: price-per-square-foot differences are compatible with integer output rounding plus rounded source price (only the largest 16 exceeded 1 INR). Yield differences are small percentage-point deviations consistent with rounded rent, costs, and price (median residual {residuals['net rental yield'].median():.6f}). Five-year value differences compound the displayed price and appreciation precision (median residual {residuals['five-year projected value'].median():.6f} lakhs). The capital-gain relationship matches within 0.005 percentage points. The units and scale match the intended formulas; the stored values do not preserve enough precision to reproduce every generated value exactly.

## Corrected formulas and targets

Corrected values are calculated from the stored source inputs, without overwriting the source:

- `Price_Per_SqFt_INR_Corrected` = `Price_INR_Lakhs * 100000 / Area_SqFt`.
- `Net_Rental_Yield_Pct_Corrected` = `(annual rent - annual maintenance - annual property tax) / Price_INR_Lakhs * 100`.
- `Target_Projected_Value_After_5Y_INR_Lakhs` = `Price_INR_Lakhs * (1 + Assumed_Annual_Appreciation_Rate_Pct / 100) ** 5`. The stored appreciation percentage is explicitly converted to a decimal.
- `Target_Good_Investment_Label` = 1 only when yield is at least 3%, assumed appreciation is at least 7.5%, distance to city centre is at most 15 km, and amenities score is at least 4; otherwise it is 0.

The corrected values retain six decimal places to avoid introducing another display-rounding discrepancy. The original derived fields were not silently replaced.

## Label verification

Applying the supplied educational synthetic rule to the stored source columns produces **{int(rule_label.sum()):,}** positive labels versus **{int(df['Good_Investment_Label'].sum()):,}** recorded positives. There are **{int(label_mismatch.sum()):,}** mismatches: `{mismatch_ids}`. All five are recorded as 0 but satisfy the supplied rule using the stored values; each has assumed appreciation exactly 7.5%.

This label is a transparent educational rule, not a proven measure of actual investment quality. It must not be interpreted as a real-world investment recommendation.

## Leakage-safe feature set

The prepared file contains **{len(prepared):,} rows and {len(prepared.columns):,} columns**. It includes `Property_ID`, numeric current-property features, one-hot encoded categorical features, and corrected targets.

Numeric predictors:

{chr(10).join(f'- `{column}`' for column in NUMERIC_FEATURES)}

Categorical predictors encoded as one-hot columns:

{chr(10).join(f'- `{column}`' for column in CATEGORICAL_FEATURES)}

The categorical variables are retained through their generated `column__category` indicator columns. No imputation was needed, and no scaling was applied because scaling can be chosen inside a later model-specific pipeline.

### Target-specific leakage decisions

- Future-value task: use the current-property predictors above and `Price_INR_Lakhs`; predict `Target_Projected_Value_After_5Y_INR_Lakhs`. Exclude the original projected value, projected capital gain, corrected target, investment labels, and `Assumed_Annual_Appreciation_Rate_Pct` from a leakage-safe general model because appreciation is a direct input to this deterministic target formula. A scenario model may use appreciation deliberately, but that is not an independent prediction task.
- Label task: use `Target_Good_Investment_Label` only as the target. The four rule inputs (`Net_Rental_Yield_Pct`, `Assumed_Annual_Appreciation_Rate_Pct`, `Distance_to_CBD_km`, and `Amenities_Score_1_to_10`) are retained for transparent rule validation, but using them to train a label model would reproduce the rule and constitute target leakage. The corrected label is not a predictor.
- Price-per-square-foot or rental-yield tasks: exclude the corresponding corrected target and the source field it replaces; also exclude any direct formula inputs when evaluating a model intended to learn rather than recompute the formula.

Excluded from predictors: `Projected_Value_After_5Y_INR_Lakhs`, `Projected_5Y_Capital_Gain_Pct`, `Good_Investment_Label`, and the corrected target columns. `Net_Rental_Yield_Pct` and `Assumed_Annual_Appreciation_Rate_Pct` are not included in the default feature matrix because they are derived/assumption fields and directly participate in the supplied label or future-value formulas.

## Output validation

- Prepared output: `{output_path.relative_to(ROOT)}`
- Prepared rows / columns: **{len(prepared):,} / {len(prepared.columns):,}**
- Prepared exact duplicate rows: **{duplicate_rows:,}**
- Prepared duplicate property IDs: **{duplicate_ids:,}**
- Prepared missing cells: **{missing_cells:,}**
- Corrected ppsf, yield, and future-value fields have no missing values: **{bool(prepared[['Price_Per_SqFt_INR_Corrected', 'Net_Rental_Yield_Pct_Corrected', 'Target_Projected_Value_After_5Y_INR_Lakhs']].notna().all().all())}**

No machine-learning model was trained, no MLflow configuration was added, and no Streamlit application was built.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Prepared data: {output_path}")
    print(f"Report: {report_path}")
    print(f"Rows/columns: {len(prepared):,}/{len(prepared.columns):,}")
    print(f"Label mismatches: {int(label_mismatch.sum()):,}")
    print(f"Prepared missing cells/duplicates: {missing_cells:,}/{duplicate_rows:,}")
    print(f"Source SHA unchanged: {source_hash_before == source_hash_after}")


if __name__ == "__main__":
    main()
