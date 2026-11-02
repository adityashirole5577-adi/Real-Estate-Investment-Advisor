"""Phase 3 validation and EDA for the synthetic real-estate dataset.

The raw CSV is read-only. Existing Phase 3 outputs are preserved; this run
writes numbered alternatives when an output name is already present.
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
REPORT_BASE = ROOT / "reports" / "phase3_dataset_validation.md"
FIGURES_DIR = ROOT / "reports" / "figures" / "phase3"


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


def bullets(values: pd.Series) -> str:
    return "\n".join(f"- `{key}`: {int(value):,}" for key, value in values.items())


def save_figure(filename: str) -> str:
    target = unused(FIGURES_DIR / filename)
    plt.tight_layout()
    plt.savefig(target, dpi=160, bbox_inches="tight")
    plt.close()
    return target.name


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    report_path = unused(REPORT_BASE)
    before_hash = sha256(RAW_PATH)
    before_size = RAW_PATH.stat().st_size
    df = pd.read_csv(RAW_PATH)

    required = {
        "Area_SqFt", "Price_INR_Lakhs", "Price_Per_SqFt_INR",
        "Estimated_Annual_Rent_INR_Lakhs", "Annual_Maintenance_INR_Lakhs",
        "Annual_Property_Tax_INR_Lakhs", "Net_Rental_Yield_Pct",
        "Assumed_Annual_Appreciation_Rate_Pct", "Projected_Value_After_5Y_INR_Lakhs",
        "Projected_5Y_Capital_Gain_Pct", "Good_Investment_Label",
    }
    missing_required = sorted(required.difference(df.columns))
    if missing_required:
        raise ValueError(f"Required columns missing: {missing_required}")

    numeric = df.select_dtypes(include="number").columns.tolist()
    missing = df.isna().sum()
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df["Property_ID"].duplicated().sum())

    flags = {
        "area_non_positive": df["Area_SqFt"] <= 0,
        "property_age_negative": df["Property_Age_Years"] < 0,
        "floor_above_total_floors": df["Floor_No"] > df["Total_Floors"],
        "distance_negative": df[["Distance_to_CBD_km", "Distance_to_Metro_km", "Distance_to_School_km", "Distance_to_Hospital_km"]].lt(0).any(axis=1),
        "amenities_outside_1_to_10": ~df["Amenities_Score_1_to_10"].between(1, 10),
        "price_or_rent_non_positive": df[["Price_Per_SqFt_INR", "Price_INR_Lakhs", "Estimated_Annual_Rent_INR_Lakhs"]].le(0).any(axis=1),
        "maintenance_or_tax_negative": df[["Annual_Maintenance_INR_Lakhs", "Annual_Property_Tax_INR_Lakhs"]].lt(0).any(axis=1),
        "net_yield_outside_0_to_100_pct": ~df["Net_Rental_Yield_Pct"].between(0, 100),
        "appreciation_outside_minus_100_to_100_pct": ~df["Assumed_Annual_Appreciation_Rate_Pct"].between(-100, 100),
        "projected_value_non_positive": df["Projected_Value_After_5Y_INR_Lakhs"] <= 0,
        "capital_gain_below_minus_100_pct": df["Projected_5Y_Capital_Gain_Pct"] < -100,
        "label_not_binary": ~df["Good_Investment_Label"].isin([0, 1]),
    }

    expected_price = df["Price_INR_Lakhs"] * 100_000 / df["Area_SqFt"]
    expected_yield = (df["Estimated_Annual_Rent_INR_Lakhs"] - df["Annual_Maintenance_INR_Lakhs"] - df["Annual_Property_Tax_INR_Lakhs"]) / df["Price_INR_Lakhs"] * 100
    expected_value = df["Price_INR_Lakhs"] * (1 + df["Assumed_Annual_Appreciation_Rate_Pct"] / 100) ** 5
    expected_gain = (df["Projected_Value_After_5Y_INR_Lakhs"] / df["Price_INR_Lakhs"] - 1) * 100
    checks = {
        "Price per square foot": (df["Price_Per_SqFt_INR"] - expected_price).abs(),
        "Net rental yield": (df["Net_Rental_Yield_Pct"] - expected_yield).abs(),
        "Projected five-year value": (df["Projected_Value_After_5Y_INR_Lakhs"] - expected_value).abs(),
        "Projected five-year capital gain": (df["Projected_5Y_Capital_Gain_Pct"] - expected_gain).abs(),
    }
    tolerances = {"Price per square foot": 1.0, "Net rental yield": 0.011, "Projected five-year value": 0.011, "Projected five-year capital gain": 0.011}

    sns.set_theme(style="whitegrid")
    figures: list[str] = []
    plt.figure(figsize=(9, 5)); sns.histplot(df["Price_INR_Lakhs"], bins=40, color="#2563eb"); plt.title("Synthetic Property Price Distribution"); plt.xlabel("Price (INR lakhs)"); plt.ylabel("Properties"); figures.append(save_figure("property_price_distribution.png"))
    city_price = df.groupby("City", as_index=False)["Price_INR_Lakhs"].mean().sort_values("Price_INR_Lakhs")
    plt.figure(figsize=(10, 6)); sns.barplot(data=city_price, x="Price_INR_Lakhs", y="City", color="#2563eb"); plt.title("Average Synthetic Property Price by City"); plt.xlabel("Mean price (INR lakhs)"); figures.append(save_figure("property_prices_by_city_2.png"))
    plt.figure(figsize=(9, 6)); sns.boxplot(data=df, x="Property_Type", y="Price_INR_Lakhs", hue="Property_Type", legend=False); plt.title("Synthetic Property Price by Property Type"); plt.xlabel("Property type"); plt.ylabel("Price (INR lakhs)"); figures.append(save_figure("property_prices_by_property_type_2.png"))
    sample = df.sample(min(5000, len(df)), random_state=42)
    plt.figure(figsize=(9, 6)); sns.scatterplot(data=sample, x="Area_SqFt", y="Price_INR_Lakhs", hue="Property_Type", alpha=0.5, s=24); plt.title("Synthetic Property Area versus Price"); plt.xlabel("Area (sq ft)"); plt.ylabel("Price (INR lakhs)"); figures.append(save_figure("price_vs_area_2.png"))
    city_yield = df.groupby("City", as_index=False)["Net_Rental_Yield_Pct"].mean().sort_values("Net_Rental_Yield_Pct")
    plt.figure(figsize=(10, 6)); sns.barplot(data=city_yield, x="Net_Rental_Yield_Pct", y="City", color="#0f766e"); plt.title("Average Synthetic Net Rental Yield by City"); plt.xlabel("Mean net rental yield (%)"); figures.append(save_figure("rental_yield_by_city_2.png"))
    city_app = df.groupby("City", as_index=False)["Assumed_Annual_Appreciation_Rate_Pct"].mean().sort_values("Assumed_Annual_Appreciation_Rate_Pct")
    plt.figure(figsize=(10, 6)); sns.barplot(data=city_app, x="Assumed_Annual_Appreciation_Rate_Pct", y="City", color="#b45309"); plt.title("Assumed Synthetic Appreciation Rate by City"); plt.xlabel("Mean assumed rate (%)"); figures.append(save_figure("assumed_appreciation_by_city_2.png"))
    labels = df["Good_Investment_Label"].value_counts().sort_index()
    plt.figure(figsize=(7, 5)); sns.barplot(x=labels.index.astype(str), y=labels.values, color="#7c3aed"); plt.title("Synthetic Investment-Label Distribution"); plt.xlabel("Good_Investment_Label"); plt.ylabel("Properties"); figures.append(save_figure("good_investment_label_distribution_2.png"))
    correlation_columns = [column for column in numeric if df[column].nunique() > 1]
    plt.figure(figsize=(14, 11)); sns.heatmap(df[correlation_columns].corr(), cmap="vlag", center=0); plt.title("Synthetic Numerical-Feature Correlation Heatmap"); figures.append(save_figure("numerical_correlation_heatmap_2.png"))

    after_hash = sha256(RAW_PATH)
    dtype_lines = "\n".join(f"- `{column}`: `{dtype}`" for column, dtype in df.dtypes.items())
    range_lines = "\n".join(f"- `{name}`: {int(mask.sum()):,} flagged rows" for name, mask in flags.items())
    formula_lines = []
    for name, residual in checks.items():
        consistent = residual <= tolerances[name]
        formula_lines.append(f"| {name} | {int(consistent.sum()):,} | {int((~consistent).sum()):,} | {residual.max():,.4f} | {tolerances[name]:,.3f} |")
    formula_table = "\n".join(formula_lines)
    report = f"""# Phase 3: Dataset Validation and Exploratory Data Analysis

## Scope and provenance

This is a validation and EDA report for the **synthetic** file `data/raw/india_real_estate_investment_synthetic_dataset.csv`. Prices, rents, appreciation rates, projected values, and labels are generated or assumption-based; they are not verified market facts, investment advice, or observed outcomes. The source CSV was not changed and no machine-learning model was trained.

## Dataset structure

- Rows: **{len(df):,}**
- Columns: **{len(df.columns):,}**
- Source size before/after: **{before_size:,} / {RAW_PATH.stat().st_size:,} bytes**
- Source SHA-256 unchanged: **{before_hash == after_hash}**
- Exact duplicate records: **{duplicate_rows:,}**
- Duplicate `Property_ID` values: **{duplicate_ids:,}**
- Total missing cells: **{int(missing.sum()):,}**

### Column names and data types

{dtype_lines}

### Missing values by column

{bullets(missing)}

## Numerical range and consistency checks

No values were corrected or removed. These are audit flags against basic domain expectations:

{range_lines}

`Floor_No > Total_Floors` is reported as a structural inconsistency. Zero floor and zero total floors were not automatically treated as invalid because they may represent ground-level/non-high-rise records.

## Calculated-column validation

Expected formulas: price per square foot = `Price_INR_Lakhs * 100000 / Area_SqFt`; net rental yield = `(rent - maintenance - tax) / price * 100`; projected five-year value = `price * (1 + appreciation / 100) ** 5`; projected capital gain = `(projected value / price - 1) * 100`. Tolerances allow rounded source values.

| Check | Consistent rows | Inconsistent rows | Maximum absolute residual | Tolerance |
|---|---:|---:|---:|---:|
{formula_table}

### `Good_Investment_Label` against its documented rule

No documented label rule was found in the existing scripts, reports, requirements, or other project text. Therefore, the label cannot be validated against a documented specification without inventing one. Observed values are `{sorted(df['Good_Investment_Label'].unique().tolist())}` with counts:

{bullets(df['Good_Investment_Label'].value_counts().sort_index())}

The label is treated as synthetic and assumption-based; no replacement rule was inferred or applied.

## EDA charts

The eight requested charts were generated in `reports/figures/phase3/` using non-overwriting filenames:

{chr(10).join(f"- `{name}`" for name in figures)}

## Conclusion

This file contains {len(df):,} synthetic records. Validation results above preserve all source issues for review. Projected values depend on assumed appreciation rates, and the investment label has no documented rule in this project. Findings describe this synthetic file only. Phase 3 did not build Streamlit, configure MLflow, or train models.
"""
    report_path.write_text(report, encoding="utf-8")
    print(f"Report: {report_path}")
    print(f"Figures written: {len(figures)}")
    print(f"Rows/columns: {len(df):,}/{len(df.columns):,}")
    print(f"Missing cells: {int(missing.sum()):,}; duplicates: {duplicate_rows:,}")
    print(f"Source SHA unchanged: {before_hash == after_hash}")
    print(f"Formula inconsistent rows: {', '.join(f'{name}={int((residual > tolerances[name]).sum()):,}' for name, residual in checks.items())}")


if __name__ == "__main__":
    main()
