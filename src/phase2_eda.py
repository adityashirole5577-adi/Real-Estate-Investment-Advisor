"""Reproducible Phase 2 cleaning and exploratory analysis.

The raw CSV is read-only. This script writes a separately named processed CSV,
figures, and a plain-language report.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "india_housing_prices.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "india_housing_prices_cleaned.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORT_PATH = ROOT / "reports" / "eda_report.md"
REFERENCE_YEAR = 2025

EXPECTED_COLUMNS = [
    "ID",
    "State",
    "City",
    "Locality",
    "Property_Type",
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Year_Built",
    "Furnished_Status",
    "Floor_No",
    "Total_Floors",
    "Age_of_Property",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Public_Transport_Accessibility",
    "Parking_Space",
    "Security",
    "Amenities",
    "Facing",
    "Owner_Type",
    "Availability_Status",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()


def pct(value: int, total: int) -> str:
    return f"{value:,} ({value / total:.2%})"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_hash_before = sha256(RAW_PATH)
    df = pd.read_csv(RAW_PATH)
    row_count_before = len(df)

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError("The raw CSV columns do not match the verified schema.")

    numeric_columns = [
        "ID",
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Year_Built",
        "Floor_No",
        "Total_Floors",
        "Age_of_Property",
        "Nearby_Schools",
        "Nearby_Hospitals",
    ]

    missing = df.isna().sum()
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df["ID"].duplicated().sum())

    suspicious_rules = {
        "non_positive_price": df["Price_in_Lakhs"] <= 0,
        "non_positive_size": df["Size_in_SqFt"] <= 0,
        "non_positive_bhk": df["BHK"] <= 0,
        "negative_price_per_sqft": df["Price_per_SqFt"] < 0,
        "year_before_1800": df["Year_Built"] < 1800,
        "year_after_reference": df["Year_Built"] > REFERENCE_YEAR,
        "negative_age": df["Age_of_Property"] < 0,
        "age_over_150": df["Age_of_Property"] > 150,
        "negative_floor": df["Floor_No"] < 0,
        "non_positive_total_floors": df["Total_Floors"] <= 0,
        "negative_nearby_schools": df["Nearby_Schools"] < 0,
        "negative_nearby_hospitals": df["Nearby_Hospitals"] < 0,
    }

    expected_price_per_sqft = df["Price_in_Lakhs"] / df["Size_in_SqFt"]
    price_consistent = (df["Price_per_SqFt"] - expected_price_per_sqft).abs() <= 0.011
    age_consistent = df["Age_of_Property"] == REFERENCE_YEAR - df["Year_Built"]
    floor_consistent = df["Floor_No"] <= df["Total_Floors"]
    generated_locality = df["Locality"].str.fullmatch(r"Locality_\d+").fillna(False)
    floor_inconsistent_count = int((~floor_consistent).sum())
    price_inconsistent_count = int((~price_consistent).sum())
    age_inconsistent_count = int((~age_consistent).sum())
    generated_locality_count = int(generated_locality.sum())

    # Cleaning policy: remove only exact duplicate rows and rows violating
    # unambiguous physical/domain constraints. Keep source inconsistencies as
    # flags when fixing them would require inventing values or deleting many rows.
    impossible_mask = pd.Series(False, index=df.index)
    for rule_mask in suspicious_rules.values():
        impossible_mask |= rule_mask
    exact_duplicate_mask = df.duplicated(keep="first")
    removal_mask = exact_duplicate_mask | impossible_mask
    cleaned = df.loc[~removal_mask].copy()

    cleaned["Price_per_SqFt_Check"] = price_consistent.loc[cleaned.index].map(
        {True: "consistent_after_rounding", False: "inconsistent"}
    )
    cleaned["Age_Reference_Year"] = REFERENCE_YEAR
    cleaned["Age_Consistent_With_2025"] = age_consistent.loc[cleaned.index]
    cleaned["Floor_Consistency_Check"] = floor_consistent.loc[cleaned.index].map(
        {True: "consistent", False: "Floor_No_exceeds_Total_Floors"}
    )
    cleaned["Locality_Name_Looks_Generated"] = generated_locality.loc[cleaned.index]
    cleaned.to_csv(PROCESSED_PATH, index=False)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(9, 5))
    sns.histplot(df["Price_in_Lakhs"], bins=40, color="#2563eb")
    plt.title("Distribution of Property Prices")
    plt.xlabel("Price (lakhs)")
    plt.ylabel("Number of properties")
    savefig("price_distribution.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Property_Type", y="Price_in_Lakhs", hue="Property_Type", legend=False)
    plt.title("Property Prices by Property Type")
    plt.xlabel("Property type")
    plt.ylabel("Price (lakhs)")
    plt.xticks(rotation=15)
    savefig("price_by_property_type.png")

    state_price = df.groupby("State", as_index=False)["Price_per_SqFt"].mean().sort_values("Price_per_SqFt")
    plt.figure(figsize=(10, 7))
    sns.barplot(data=state_price, x="Price_per_SqFt", y="State", color="#0f766e")
    plt.title("Average Price per Square Foot by State")
    plt.xlabel("Average price per square foot (lakhs/sq ft)")
    plt.ylabel("State")
    savefig("average_price_per_sqft_by_state.png")

    sample = df.sample(n=min(20_000, len(df)), random_state=42)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=sample, x="Size_in_SqFt", y="Price_in_Lakhs", hue="Property_Type", alpha=0.35, s=18)
    plt.title("Relationship Between Property Size and Price")
    plt.xlabel("Size (sq ft)")
    plt.ylabel("Price (lakhs)")
    savefig("size_vs_price.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="BHK", y="Price_in_Lakhs", hue="BHK", legend=False)
    plt.title("Price Distribution by BHK")
    plt.xlabel("BHK")
    plt.ylabel("Price (lakhs)")
    savefig("price_by_bhk.png")

    correlation_columns = [
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Year_Built",
        "Age_of_Property",
        "Floor_No",
        "Total_Floors",
        "Nearby_Schools",
        "Nearby_Hospitals",
    ]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[correlation_columns].corr(), cmap="vlag", center=0, annot=True, fmt=".2f")
    plt.title("Correlations Among Numerical Variables")
    savefig("numeric_correlations.png")

    age_price = df.groupby("Age_of_Property", as_index=False)["Price_in_Lakhs"].mean()
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=age_price, x="Age_of_Property", y="Price_in_Lakhs", marker="o", color="#b45309")
    plt.title("Average Price by Property Age")
    plt.xlabel("Property age (dataset reference year 2025)")
    plt.ylabel("Average price (lakhs)")
    savefig("age_vs_price.png")

    feature_columns = [
        "Furnished_Status",
        "Public_Transport_Accessibility",
        "Parking_Space",
        "Security",
        "Availability_Status",
    ]
    feature_rows = []
    for column in feature_columns:
        counts = df[column].value_counts(dropna=False)
        for value, count in counts.items():
            feature_rows.append({"Feature": column, "Value": str(value), "Count": count})
    feature_summary = pd.DataFrame(feature_rows)
    plt.figure(figsize=(11, 7))
    sns.barplot(data=feature_summary, x="Count", y="Value", hue="Feature", dodge=False)
    plt.title("Availability of Selected Property Features")
    plt.xlabel("Number of properties")
    plt.ylabel("Feature value")
    plt.legend(title="Feature", bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig("feature_availability.png")

    raw_hash_after = sha256(RAW_PATH)
    size_price_corr = df["Size_in_SqFt"].corr(df["Price_in_Lakhs"])
    age_price_corr = df["Age_of_Property"].corr(df["Price_in_Lakhs"])
    median_price_by_type = df.groupby("Property_Type")["Price_in_Lakhs"].median().sort_values(ascending=False)
    median_price_by_bhk = df.groupby("BHK")["Price_in_Lakhs"].median()
    top_state = state_price.iloc[-1]
    bottom_state = state_price.iloc[0]

    dtype_lines = "\n".join(f"- `{column}`: `{dtype}`" for column, dtype in df.dtypes.items())
    missing_lines = "\n".join(f"- `{column}`: {int(value):,}" for column, value in missing.items())
    suspicious_lines = "\n".join(
        f"- {name}: {int(mask.sum()):,} rows" for name, mask in suspicious_rules.items()
    )
    category_lines = "\n".join(
        f"- `{column}`: {df[column].nunique(dropna=False):,} unique values"
        for column in df.select_dtypes(include=["str"]).columns
    )
    cleaning_lines = "\n".join(
        [
            f"- Exact duplicate rows removed: {int(exact_duplicate_mask.sum()):,}.",
            f"- Rows violating unambiguous numeric/domain rules removed: {int(impossible_mask.sum()):,}.",
            f"- Total rows removed: {int(removal_mask.sum()):,}.",
            "- Missing values were not imputed because there were none.",
            "- Duplicate IDs were not altered because there were none.",
            "- Floor inconsistencies were retained and flagged because removing {0:,} rows would be a large, potentially biased deletion.".format(floor_inconsistent_count),
            "- Age was retained and flagged using 2025 as the dataset reference year because it matches every row; the file does not document why 2025 is the reference year.",
            "- Price-per-square-foot values were retained because they agree with price divided by size after two-decimal rounding for {0:,} of {1:,} rows.".format(int(price_consistent.sum()), len(df)),
        ]
    )

    report = f"""# Phase 2: Data Cleaning and Exploratory Data Analysis

## Scope and provenance caution

This report was generated by `src/phase2_eda.py` from the downloaded raw CSV. The raw file was never overwritten. The dataset's repository does not document a collection method, so the findings below describe the file's observed values; they should not be treated as verified Indian market facts.

## Dataset overview

- Raw records: **{row_count_before:,}**
- Processed records: **{len(cleaned):,}**
- Raw file SHA-256 before and after processing: `{raw_hash_before}` / `{raw_hash_after}`
- Raw file changed: **{raw_hash_before != raw_hash_after}**
- Numeric columns: {len(numeric_columns)}

## Data types

{dtype_lines}

## Missing values and duplicates

Missing values by column:

{missing_lines}

- Exact duplicate rows: **{duplicate_rows:,}**
- Duplicate IDs: **{duplicate_ids:,}**

## Numeric checks

The following checks found no negative values or other unambiguous domain violations in the source data:

{suspicious_lines}

The main structural inconsistency is `Floor_No > Total_Floors` in **{floor_inconsistent_count:,}** rows. This is too large a group to remove without risking substantial selection bias, so the processed file keeps these records and adds `Floor_Consistency_Check`.

`Price_per_SqFt` is expressed in lakhs per square foot in this file. Comparing it with `Price_in_Lakhs / Size_in_SqFt` shows **{int(price_consistent.sum()):,}** consistent rows under a two-decimal rounding tolerance and **{price_inconsistent_count:,}** rows outside that tolerance. No price values were overwritten; the processed file adds `Price_per_SqFt_Check`.

`Age_of_Property` equals `2025 - Year_Built` in **{int(age_consistent.sum()):,}** rows and disagrees in **{age_inconsistent_count:,}** rows. This supports using 2025 as an internal reference year, but it is an observed relationship, not documented metadata. The processed file adds `Age_Reference_Year` and `Age_Consistent_With_2025`.

## Categorical inspection

{category_lines}

All **{generated_locality_count:,}** locality values match the generic pattern `Locality_<number>`. This is evidence consistent with generated or anonymized locality names, but it is not proof of how the dataset was created. Localities were not replaced.

## Cleaning decisions and affected records

{cleaning_lines}

The cleaned CSV is saved at `{PROCESSED_PATH.relative_to(ROOT)}`. It includes the original columns plus diagnostic flag columns; the original source values remain available for audit.

## EDA findings

These are observations from this file, not claims about the real housing market:

- Prices range from {df['Price_in_Lakhs'].min():.2f} to {df['Price_in_Lakhs'].max():.2f} lakhs. The distribution is bounded and should not automatically be interpreted as a natural market distribution.
- The median prices by property type, from highest to lowest, are: {', '.join(f'{k} ({v:.2f} lakhs)' for k, v in median_price_by_type.items())}.
- The state with the highest observed average `Price_per_SqFt` is **{top_state['State']}** ({top_state['Price_per_SqFt']:.3f} lakhs/sq ft); the lowest is **{bottom_state['State']}** ({bottom_state['Price_per_SqFt']:.3f} lakhs/sq ft). These are file-level comparisons, not validated market rankings.
- Size and price have a Pearson correlation of **{size_price_corr:.3f}** in this file. Correlation does not establish causation.
- Median prices by BHK are: {', '.join(f'{int(k)} BHK ({v:.2f} lakhs)' for k, v in median_price_by_bhk.items())}.
- Property age and price have a Pearson correlation of **{age_price_corr:.3f}**. Because age is derived from the same year relationship in every row, this should be interpreted cautiously.
- The feature-availability chart summarizes categorical counts for furnishing, transport access, parking, security, and availability status. These are availability patterns in the file, not measured service quality.

## Generated figures

The following PNG files are in `reports/figures/`:

- `price_distribution.png`
- `price_by_property_type.png`
- `average_price_per_sqft_by_state.png`
- `size_vs_price.png`
- `price_by_bhk.png`
- `numeric_correlations.png`
- `age_vs_price.png`
- `feature_availability.png`

No classification or regression model, MLflow configuration, or Streamlit application was created in Phase 2.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Raw rows: {row_count_before:,}")
    print(f"Processed rows: {len(cleaned):,}")
    print(f"Rows removed: {int(removal_mask.sum()):,}")
    print(f"Raw SHA unchanged: {raw_hash_before == raw_hash_after}")
    print(f"Processed file: {PROCESSED_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(f"Figures: {len(list(FIGURES_DIR.glob('*.png')))}")


if __name__ == "__main__":
    main()
