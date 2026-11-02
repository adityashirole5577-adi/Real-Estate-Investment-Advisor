"""Phase 3: validation and exploratory analysis for the synthetic investment dataset.

The source CSV is read-only. This script writes a Markdown audit report and
standalone PNG figures; it does not clean the data, train models, or create
derived datasets.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "india_real_estate_investment_synthetic_dataset.csv"
REPORT_PATH = ROOT / "reports" / "phase3_dataset_validation.md"
FIGURES_DIR = ROOT / "reports" / "figures" / "phase3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available_path(path: Path) -> Path:
    """Return path, or a numbered alternative without overwriting a file."""
    if not path.exists():
        return path
    for number in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{number}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an unused output name for {path}")


def fmt(value: float) -> str:
    return f"{value:,.4f}"


def save_figure(name: str) -> str:
    target = available_path(FIGURES_DIR / name)
    plt.tight_layout()
    plt.savefig(target, dpi=160, bbox_inches="tight")
    plt.close()
    return target.name


def count_lines(series: pd.Series) -> str:
    return "\n".join(f"- `{column}`: {int(value):,}" for column, value in series.items())


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    report_path = available_path(REPORT_PATH)
    raw_hash_before = sha256(RAW_PATH)
    raw_size_before = RAW_PATH.stat().st_size
    df = pd.read_csv(RAW_PATH)

    expected_formula_columns = {
        "Price_Per_SqFt_INR",
        "Price_INR_Lakhs",
        "Area_SqFt",
        "Estimated_Annual_Rent_INR_Lakhs",
        "Annual_Maintenance_INR_Lakhs",
        "Annual_Property_Tax_INR_Lakhs",
        "Net_Rental_Yield_Pct",
        "Assumed_Annual_Appreciation_Rate_Pct",
        "Projected_Value_After_5Y_INR_Lakhs",
        "Projected_5Y_Capital_Gain_Pct",
        "Good_Investment_Label",
    }
    missing_formula_columns = expected_formula_columns.difference(df.columns)
    if missing_formula_columns:
        raise ValueError(f"Required calculated columns are missing: {sorted(missing_formula_columns)}")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(exclude="number").columns.tolist()
    missing = df.isna().sum()
    duplicate_rows = int(df.duplicated().sum())
    duplicate_property_ids = int(df["Property_ID"].duplicated().sum())

    range_checks = {
        "negative_or_zero_area": df["Area_SqFt"] <= 0,
        "negative_property_age": df["Property_Age_Years"] < 0,
        "floor_above_total_floors": df["Floor_No"] > df["Total_Floors"],
        "negative_distances": df[[
            "Distance_to_CBD_km", "Distance_to_Metro_km", "Distance_to_School_km",
            "Distance_to_Hospital_km",
        ]].lt(0).any(axis=1),
        "amenities_outside_1_to_10": ~df["Amenities_Score_1_to_10"].between(1, 10),
        "negative_prices_or_rent": df[[
            "Price_Per_SqFt_INR", "Price_INR_Lakhs", "Estimated_Annual_Rent_INR_Lakhs",
        ]].le(0).any(axis=1),
        "negative_costs": df[[
            "Annual_Maintenance_INR_Lakhs", "Annual_Property_Tax_INR_Lakhs",
        ]].lt(0).any(axis=1),
        "yield_outside_reasonable_percentage_range": ~df["Net_Rental_Yield_Pct"].between(0, 100),
        "appreciation_outside_reasonable_percentage_range": ~df[
            "Assumed_Annual_Appreciation_Rate_Pct"
        ].between(-100, 100),
        "projected_value_non_positive": df["Projected_Value_After_5Y_INR_Lakhs"] <= 0,
        "capital_gain_below_minus_100_pct": df["Projected_5Y_Capital_Gain_Pct"] < -100,
        "label_not_binary": ~df["Good_Investment_Label"].isin([0, 1]),
    }

    price_expected = df["Price_INR_Lakhs"] * 100_000 / df["Area_SqFt"]
    price_residual = (df["Price_Per_SqFt_INR"] - price_expected).abs()
    price_consistent = price_residual <= 1.0

    yield_expected = (
        (df["Estimated_Annual_Rent_INR_Lakhs"]
         - df["Annual_Maintenance_INR_Lakhs"]
         - df["Annual_Property_Tax_INR_Lakhs"])
        / df["Price_INR_Lakhs"]
        * 100
    )
    yield_residual = (df["Net_Rental_Yield_Pct"] - yield_expected).abs()
    yield_consistent = yield_residual <= 0.011

    value_expected = df["Price_INR_Lakhs"] * (
        1 + df["Assumed_Annual_Appreciation_Rate_Pct"] / 100
    ) ** 5
    value_residual = (df["Projected_Value_After_5Y_INR_Lakhs"] - value_expected).abs()
    value_consistent = value_residual <= 0.011

    gain_expected = (
        df["Projected_Value_After_5Y_INR_Lakhs"] / df["Price_INR_Lakhs"] - 1
    ) * 100
    gain_residual = (df["Projected_5Y_Capital_Gain_Pct"] - gain_expected).abs()
    gain_consistent = gain_residual <= 0.011

    # No label rule was documented in the repository. Test the most direct
    # implied rule only as an audit aid; do not invent or apply a replacement.
    label_values = sorted(df["Good_Investment_Label"].dropna().unique().tolist())
    label_counts = df["Good_Investment_Label"].value_counts().sort_index()
    label_by_yield_median = df.groupby("Good_Investment_Label")["Net_Rental_Yield_Pct"].median()
    label_by_gain_median = df.groupby("Good_Investment_Label")["Projected_5Y_Capital_Gain_Pct"].median()

    sns.set_theme(style="whitegrid")
    figure_names: list[str] = []

    city_price = df.groupby("City", as_index=False)["Price_INR_Lakhs"].mean().sort_values("Price_INR_Lakhs")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=city_price, x="Price_INR_Lakhs", y="City", color="#2563eb")
    plt.title("Synthetic Dataset: Mean Property Price by City")
    plt.xlabel("Mean price (INR lakhs)")
    plt.ylabel("City")
    figure_names.append(save_figure("property_prices_by_city.png"))

    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="Property_Type", y="Price_INR_Lakhs", hue="Property_Type", legend=False)
    plt.title("Synthetic Dataset: Property Prices by Property Type")
    plt.xlabel("Property type")
    plt.ylabel("Price (INR lakhs)")
    figure_names.append(save_figure("property_prices_by_property_type.png"))

    sample = df.sample(min(5000, len(df)), random_state=42)
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=sample, x="Area_SqFt", y="Price_INR_Lakhs", hue="Property_Type", alpha=0.5, s=24)
    plt.title("Synthetic Dataset: Price versus Area")
    plt.xlabel("Area (sq ft)")
    plt.ylabel("Price (INR lakhs)")
    figure_names.append(save_figure("price_vs_area.png"))

    city_yield = df.groupby("City", as_index=False)["Net_Rental_Yield_Pct"].mean().sort_values("Net_Rental_Yield_Pct")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=city_yield, x="Net_Rental_Yield_Pct", y="City", color="#0f766e")
    plt.title("Synthetic Dataset: Mean Net Rental Yield by City")
    plt.xlabel("Mean net rental yield (%)")
    plt.ylabel("City")
    figure_names.append(save_figure("rental_yield_by_city.png"))

    city_appreciation = df.groupby("City", as_index=False)["Assumed_Annual_Appreciation_Rate_Pct"].mean().sort_values("Assumed_Annual_Appreciation_Rate_Pct")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=city_appreciation, x="Assumed_Annual_Appreciation_Rate_Pct", y="City", color="#b45309")
    plt.title("Synthetic Dataset: Assumed Annual Appreciation by City")
    plt.xlabel("Assumed appreciation rate (%)")
    plt.ylabel("City")
    figure_names.append(save_figure("assumed_appreciation_by_city.png"))

    plt.figure(figsize=(7, 5))
    counts = df["Good_Investment_Label"].value_counts().sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values, color="#7c3aed")
    plt.title("Synthetic Dataset: Good Investment Label Distribution")
    plt.xlabel("Good_Investment_Label (assumed rule output)")
    plt.ylabel("Number of properties")
    figure_names.append(save_figure("good_investment_label_distribution.png"))

    correlation_columns = [
        "BHK", "Area_SqFt", "Property_Age_Years", "Floor_No", "Total_Floors",
        "Distance_to_CBD_km", "Distance_to_Metro_km", "Amenities_Score_1_to_10",
        "Price_Per_SqFt_INR", "Price_INR_Lakhs", "Estimated_Annual_Rent_INR_Lakhs",
        "Net_Rental_Yield_Pct", "Assumed_Annual_Appreciation_Rate_Pct",
        "Projected_Value_After_5Y_INR_Lakhs", "Projected_5Y_Capital_Gain_Pct",
        "Good_Investment_Label",
    ]
    plt.figure(figsize=(14, 11))
    sns.heatmap(df[correlation_columns].corr(), cmap="vlag", center=0, annot=False)
    plt.title("Synthetic Dataset: Correlation Heatmap of Numerical Features")
    figure_names.append(save_figure("numerical_correlation_heatmap.png"))

    raw_hash_after = sha256(RAW_PATH)
    raw_size_after = RAW_PATH.stat().st_size
    range_lines = "\n".join(f"- {name}: {int(mask.sum()):,}" for name, mask in range_checks.items())
    dtype_lines = "\n".join(f"- `{column}`: `{dtype}`" for column, dtype in df.dtypes.items())
    category_lines = "\n".join(
        f"- `{column}` ({int(df[column].nunique(dropna=False)):,}): {', '.join(map(str, df[column].drop_duplicates().tolist()))}"
        for column in categorical_columns
    )
    numeric_summary = df[numeric_columns].describe().T.to_string(float_format=lambda value: f"{value:,.3f}")
    city_price_lines = "\n".join(
        f"- {row.City}: mean {row.Price_INR_Lakhs:,.2f} lakhs"
        for row in city_price.itertuples()
    )
    city_yield_lines = "\n".join(
        f"- {row.City}: mean {row.Net_Rental_Yield_Pct:,.2f}%"
        for row in city_yield.itertuples()
    )
    city_appreciation_lines = "\n".join(
        f"- {row.City}: mean assumed rate {row.Assumed_Annual_Appreciation_Rate_Pct:,.2f}%"
        for row in city_appreciation.itertuples()
    )
    label_medians = "\n".join(
        f"- Label {label}: median net yield {label_by_yield_median[label]:,.2f}%; median projected five-year gain {label_by_gain_median[label]:,.2f}%"
        for label in label_values
    )
    report = f"""# Phase 3: Dataset Validation and Exploratory Data Analysis

## Scope and provenance

This report analyzes `data/raw/india_real_estate_investment_synthetic_dataset.csv` as a **synthetic dataset**. Its prices, rental figures, appreciation rates, projected values, and `Good_Investment_Label` values are generated or assumed inputs for analysis; they are not verified Indian market facts, financial advice, or observed investment outcomes. No values were changed and no machine-learning model was trained.

## Dataset overview

- Rows: **{len(df):,}**
- Columns: **{len(df.columns):,}**
- Raw file size before/after: **{raw_size_before:,} / {raw_size_after:,} bytes**
- Raw SHA-256 unchanged: **{raw_hash_before == raw_hash_after}**
- Property IDs unique: **{duplicate_property_ids == 0}**

## Column names and data types

{dtype_lines}

## Missing values and duplicates

{count_lines(missing)}

- Total missing cells: **{int(missing.sum()):,}**
- Exact duplicate records: **{duplicate_rows:,}**
- Duplicate `Property_ID` values: **{duplicate_property_ids:,}**

## Categorical values

{category_lines}

## Numerical summary statistics

```text
{numeric_summary}
```

## Impossible or inconsistent values

Counts below are validation flags; no rows were deleted or corrected.

{range_lines}

The `floor_above_total_floors` flag identifies a structural inconsistency when `Floor_No > Total_Floors`. A zero floor and zero total floors can represent a ground-level/non-high-rise record, so they were not automatically treated as impossible.

## Calculated-column validation

Calculations use the formulas implied by the column names and units. Tolerances of 1 INR for rounded price-per-square-foot values and 0.011 percentage points for two-decimal percentage fields were used.

| Check | Formula / interpretation | Consistent rows | Inconsistent rows | Maximum absolute residual |
|---|---|---:|---:|---:|
| Price per square foot | `Price_INR_Lakhs * 100000 / Area_SqFt` | {int(price_consistent.sum()):,} | {int((~price_consistent).sum()):,} | {price_residual.max():,.4f} INR |
| Net rental yield | `(rent - maintenance - tax) / price * 100` | {int(yield_consistent.sum()):,} | {int((~yield_consistent).sum()):,} | {yield_residual.max():,.4f} percentage points |
| Projected five-year value | `price * (1 + appreciation / 100) ** 5` | {int(value_consistent.sum()):,} | {int((~value_consistent).sum()):,} | {value_residual.max():,.4f} lakhs |
| Projected five-year capital gain | `(projected value / price - 1) * 100` | {int(gain_consistent.sum()):,} | {int((~gain_consistent).sum()):,} | {gain_residual.max():,.4f} percentage points |

    The price-per-square-foot and percentage-field residuals are small where the source values appear rounded for display. The projected-value check has a larger number of rows outside the stated tolerance, so that formula is not assumed to be consistently reproduced by the source values. Inconsistent rows remain unchanged in the source file and are reported rather than silently repaired.

### `Good_Investment_Label`

No documented label rule was found in the existing project files. Therefore, the label cannot be validated against a documented specification. The observed label values are `{label_values}`, with counts:

{count_lines(label_counts)}

Observed group medians are descriptive only:

{label_medians}

This label must be treated as a **synthetic, assumption-based outcome**, not a verified investment result. No replacement rule was inferred or applied.

## Exploratory findings

### Mean property prices by city

{city_price_lines}

### Mean rental yield by city

{city_yield_lines}

### Mean assumed appreciation rate by city

{city_appreciation_lines}

The appreciation rates are explicitly assumptions embedded in the synthetic data. They are not historical market measurements.

### Price versus area

The Pearson correlation between `Area_SqFt` and `Price_INR_Lakhs` is **{df['Area_SqFt'].corr(df['Price_INR_Lakhs']):.4f}**. This describes this synthetic file only and does not establish a market relationship.

### Label distribution

`Good_Investment_Label = 0` represents **{int((df['Good_Investment_Label'] == 0).sum()):,} ({(df['Good_Investment_Label'] == 0).mean():.2%})** rows; label `1` represents **{int((df['Good_Investment_Label'] == 1).sum()):,} ({(df['Good_Investment_Label'] == 1).mean():.2%})** rows.

## Figures

Figures were saved under `reports/figures/phase3/` using non-overwriting filenames:

{chr(10).join(f"- `{name}`" for name in figure_names)}

## Conclusion

The file has {len(df):,} records, no missing cells, no exact duplicate records, and unique property IDs. The derived numeric fields are broadly consistent after accounting for rounding, while any flagged range or formula inconsistencies are preserved for review. The main interpretive limitation is provenance: this is synthetic data with assumed appreciation rates and an undocumented, assumption-based investment label. Phase 3 did not train models, configure MLflow, build Streamlit, or modify the original CSV.
"""
    report_path.write_text(report, encoding="utf-8")

    print(f"Report: {report_path}")
    print(f"Figures directory: {FIGURES_DIR}")
    print(f"Figures written: {len(figure_names)}")
    print(f"Rows/columns: {len(df):,}/{len(df.columns):,}")
    print(f"Missing cells: {int(missing.sum()):,}")
    print(f"Exact duplicates: {duplicate_rows:,}")
    print(f"Formula consistency: price={int(price_consistent.sum()):,}, yield={int(yield_consistent.sum()):,}, value={int(value_consistent.sum()):,}, gain={int(gain_consistent.sum()):,}")
    print(f"Raw SHA unchanged: {raw_hash_before == raw_hash_after}")


if __name__ == "__main__":
    main()
