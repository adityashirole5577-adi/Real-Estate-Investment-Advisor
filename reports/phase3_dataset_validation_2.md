# Phase 3: Dataset Validation and Exploratory Data Analysis

## Scope and provenance

This is a validation and EDA report for the **synthetic** file `data/raw/india_real_estate_investment_synthetic_dataset.csv`. Prices, rents, appreciation rates, projected values, and labels are generated or assumption-based; they are not verified market facts, investment advice, or observed outcomes. The source CSV was not changed and no machine-learning model was trained.

## Dataset structure

- Rows: **5,000**
- Columns: **28**
- Source size before/after: **780,640 / 780,640 bytes**
- Source SHA-256 unchanged: **True**
- Exact duplicate records: **0**
- Duplicate `Property_ID` values: **0**
- Total missing cells: **0**

### Column names and data types

- `Property_ID`: `str`
- `City`: `str`
- `Locality_Tier`: `str`
- `Property_Type`: `str`
- `BHK`: `int64`
- `Area_SqFt`: `int64`
- `Property_Age_Years`: `int64`
- `Floor_No`: `int64`
- `Total_Floors`: `int64`
- `Distance_to_CBD_km`: `float64`
- `Distance_to_Metro_km`: `float64`
- `Distance_to_School_km`: `float64`
- `Distance_to_Hospital_km`: `float64`
- `Amenities_Score_1_to_10`: `int64`
- `Parking_Available`: `str`
- `Gated_Community`: `str`
- `Furnishing_Status`: `str`
- `Occupancy_Status`: `str`
- `Price_Per_SqFt_INR`: `int64`
- `Price_INR_Lakhs`: `float64`
- `Estimated_Annual_Rent_INR_Lakhs`: `float64`
- `Annual_Maintenance_INR_Lakhs`: `float64`
- `Annual_Property_Tax_INR_Lakhs`: `float64`
- `Net_Rental_Yield_Pct`: `float64`
- `Assumed_Annual_Appreciation_Rate_Pct`: `float64`
- `Projected_Value_After_5Y_INR_Lakhs`: `float64`
- `Projected_5Y_Capital_Gain_Pct`: `float64`
- `Good_Investment_Label`: `int64`

### Missing values by column

- `Property_ID`: 0
- `City`: 0
- `Locality_Tier`: 0
- `Property_Type`: 0
- `BHK`: 0
- `Area_SqFt`: 0
- `Property_Age_Years`: 0
- `Floor_No`: 0
- `Total_Floors`: 0
- `Distance_to_CBD_km`: 0
- `Distance_to_Metro_km`: 0
- `Distance_to_School_km`: 0
- `Distance_to_Hospital_km`: 0
- `Amenities_Score_1_to_10`: 0
- `Parking_Available`: 0
- `Gated_Community`: 0
- `Furnishing_Status`: 0
- `Occupancy_Status`: 0
- `Price_Per_SqFt_INR`: 0
- `Price_INR_Lakhs`: 0
- `Estimated_Annual_Rent_INR_Lakhs`: 0
- `Annual_Maintenance_INR_Lakhs`: 0
- `Annual_Property_Tax_INR_Lakhs`: 0
- `Net_Rental_Yield_Pct`: 0
- `Assumed_Annual_Appreciation_Rate_Pct`: 0
- `Projected_Value_After_5Y_INR_Lakhs`: 0
- `Projected_5Y_Capital_Gain_Pct`: 0
- `Good_Investment_Label`: 0

## Numerical range and consistency checks

No values were corrected or removed. These are audit flags against basic domain expectations:

- `area_non_positive`: 0 flagged rows
- `property_age_negative`: 0 flagged rows
- `floor_above_total_floors`: 0 flagged rows
- `distance_negative`: 0 flagged rows
- `amenities_outside_1_to_10`: 0 flagged rows
- `price_or_rent_non_positive`: 0 flagged rows
- `maintenance_or_tax_negative`: 0 flagged rows
- `net_yield_outside_0_to_100_pct`: 0 flagged rows
- `appreciation_outside_minus_100_to_100_pct`: 0 flagged rows
- `projected_value_non_positive`: 0 flagged rows
- `capital_gain_below_minus_100_pct`: 0 flagged rows
- `label_not_binary`: 0 flagged rows

`Floor_No > Total_Floors` is reported as a structural inconsistency. Zero floor and zero total floors were not automatically treated as invalid because they may represent ground-level/non-high-rise records.

## Calculated-column validation

Expected formulas: price per square foot = `Price_INR_Lakhs * 100000 / Area_SqFt`; net rental yield = `(rent - maintenance - tax) / price * 100`; projected five-year value = `price * (1 + appreciation / 100) ** 5`; projected capital gain = `(projected value / price - 1) * 100`. Tolerances allow rounded source values.

| Check | Consistent rows | Inconsistent rows | Maximum absolute residual | Tolerance |
|---|---:|---:|---:|---:|
| Price per square foot | 4,984 | 16 | 1.6829 | 1.000 |
| Net rental yield | 4,444 | 556 | 0.0593 | 0.011 |
| Projected five-year value | 1,878 | 3,122 | 0.2268 | 0.011 |
| Projected five-year capital gain | 5,000 | 0 | 0.0050 | 0.011 |

### `Good_Investment_Label` against its documented rule

No documented label rule was found in the existing scripts, reports, requirements, or other project text. Therefore, the label cannot be validated against a documented specification without inventing one. Observed values are `[0, 1]` with counts:

- `0`: 3,834
- `1`: 1,166

The label is treated as synthetic and assumption-based; no replacement rule was inferred or applied.

## EDA charts

The eight requested charts were generated in `reports/figures/phase3/` using non-overwriting filenames:

- `property_price_distribution.png`
- `property_prices_by_city_2.png`
- `property_prices_by_property_type_2.png`
- `price_vs_area_2.png`
- `rental_yield_by_city_2.png`
- `assumed_appreciation_by_city_2.png`
- `good_investment_label_distribution_2.png`
- `numerical_correlation_heatmap_2.png`

## Conclusion

This file contains 5,000 synthetic records. Validation results above preserve all source issues for review. Projected values depend on assumed appreciation rates, and the investment label has no documented rule in this project. Findings describe this synthetic file only. Phase 3 did not build Streamlit, configure MLflow, or train models.
