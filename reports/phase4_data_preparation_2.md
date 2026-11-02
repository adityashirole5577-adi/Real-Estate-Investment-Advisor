# Phase 4: Feature Engineering and Data Preparation

## Scope and provenance

This preparation uses `data/raw/india_real_estate_investment_synthetic_dataset.csv`, a **synthetic** dataset. Its prices, rents, appreciation rates, projected values, and labels are generated or assumption-based. They are not verified market estimates, actual investment outcomes, or financial advice. The original CSV was read-only and remains unchanged.

## Source audit before correction

- Source rows / columns: **5,000 / 28**
- Source size before / after: **780,640 / 780,640 bytes**
- Source SHA-256 unchanged: **True**
- Source missing cells: **0**
- Source exact duplicate rows: **0**
- Source duplicate property IDs: **0**
- Stored numeric precision observed: `Price_INR_Lakhs`: 1-2 decimal places; `Estimated_Annual_Rent_INR_Lakhs`: 1-2 decimal places; `Annual_Maintenance_INR_Lakhs`: 1-2 decimal places; `Annual_Property_Tax_INR_Lakhs`: 1-2 decimal places; `Net_Rental_Yield_Pct`: 1-2 decimal places; `Assumed_Annual_Appreciation_Rate_Pct`: 1-2 decimal places; `Projected_Value_After_5Y_INR_Lakhs`: 1-2 decimal places

The source stores money and percentage fields at one or two decimal places, while price per square foot is stored as an integer. Recomputing from those displayed values therefore cannot recover any hidden higher-precision values used during synthetic generation. The residuals below were measured before correction.

| Formula audit | Within Phase 3 tolerance | Outside tolerance | Maximum residual | Tolerance |
|---|---:|---:|---:|---:|
| price per square foot | 4,984 | 16 | 1.682927 | 1.000 |
| net rental yield | 4,444 | 556 | 0.059294 | 0.011 |
| five-year projected value | 1,878 | 3,122 | 0.226771 | 0.011 |
| five-year capital gain | 5,000 | 0 | 0.005000 | 0.011 |

Interpretation: price-per-square-foot differences are compatible with integer output rounding plus rounded source price (only the largest 16 exceeded 1 INR). Yield differences are small percentage-point deviations consistent with rounded rent, costs, and price (median residual 0.004119). Five-year value differences compound the displayed price and appreciation precision (median residual 0.015450 lakhs). The capital-gain relationship matches within 0.005 percentage points. The units and scale match the intended formulas; the stored values do not preserve enough precision to reproduce every generated value exactly.

## Corrected formulas and targets

Corrected values are calculated from the stored source inputs, without overwriting the source:

- `Price_Per_SqFt_INR_Corrected` = `Price_INR_Lakhs * 100000 / Area_SqFt`.
- `Net_Rental_Yield_Pct_Corrected` = `(annual rent - annual maintenance - annual property tax) / Price_INR_Lakhs * 100`.
- `Target_Projected_Value_After_5Y_INR_Lakhs` = `Price_INR_Lakhs * (1 + Assumed_Annual_Appreciation_Rate_Pct / 100) ** 5`. The stored appreciation percentage is explicitly converted to a decimal.
- `Target_Good_Investment_Label` = 1 only when yield is at least 3%, assumed appreciation is at least 7.5%, distance to city centre is at most 15 km, and amenities score is at least 4; otherwise it is 0.

The corrected values retain six decimal places to avoid introducing another display-rounding discrepancy. The original derived fields were not silently replaced.

## Label verification

Applying the supplied educational synthetic rule to the stored source columns produces **1,171** positive labels versus **1,166** recorded positives. There are **5** mismatches: `PROP01319, PROP02795, PROP03506, PROP03625, PROP04837`. All five are recorded as 0 but satisfy the supplied rule using the stored values; each has assumed appreciation exactly 7.5%.

This label is a transparent educational rule, not a proven measure of actual investment quality. It must not be interpreted as a real-world investment recommendation.

## Leakage-safe feature set

The prepared file contains **5,000 rows and 42 columns**. It includes `Property_ID`, numeric current-property features, one-hot encoded categorical features, and corrected targets.

Numeric predictors:

- `BHK`
- `Area_SqFt`
- `Property_Age_Years`
- `Floor_No`
- `Total_Floors`
- `Distance_to_CBD_km`
- `Distance_to_Metro_km`
- `Distance_to_School_km`
- `Distance_to_Hospital_km`
- `Amenities_Score_1_to_10`
- `Price_INR_Lakhs`

Categorical predictors encoded as one-hot columns:

- `City`
- `Locality_Tier`
- `Property_Type`
- `Parking_Available`
- `Gated_Community`
- `Furnishing_Status`
- `Occupancy_Status`

The categorical variables are retained through their generated `column__category` indicator columns. No imputation was needed, and no scaling was applied because scaling can be chosen inside a later model-specific pipeline.

### Target-specific leakage decisions

- Future-value task: use the current-property predictors above and `Price_INR_Lakhs`; predict `Target_Projected_Value_After_5Y_INR_Lakhs`. Exclude the original projected value, projected capital gain, corrected target, investment labels, and `Assumed_Annual_Appreciation_Rate_Pct` from a leakage-safe general model because appreciation is a direct input to this deterministic target formula. A scenario model may use appreciation deliberately, but that is not an independent prediction task.
- Label task: use `Target_Good_Investment_Label` only as the target. The four rule inputs (`Net_Rental_Yield_Pct`, `Assumed_Annual_Appreciation_Rate_Pct`, `Distance_to_CBD_km`, and `Amenities_Score_1_to_10`) are retained for transparent rule validation, but using them to train a label model would reproduce the rule and constitute target leakage. The corrected label is not a predictor.
- Price-per-square-foot or rental-yield tasks: exclude the corresponding corrected target and the source field it replaces; also exclude any direct formula inputs when evaluating a model intended to learn rather than recompute the formula.

Excluded from predictors: `Projected_Value_After_5Y_INR_Lakhs`, `Projected_5Y_Capital_Gain_Pct`, `Good_Investment_Label`, and the corrected target columns. `Net_Rental_Yield_Pct` and `Assumed_Annual_Appreciation_Rate_Pct` are not included in the default feature matrix because they are derived/assumption fields and directly participate in the supplied label or future-value formulas.

## Output validation

- Prepared output: `data\processed\real_estate_phase4_prepared_2.csv`
- Prepared rows / columns: **5,000 / 42**
- Prepared exact duplicate rows: **0**
- Prepared duplicate property IDs: **0**
- Prepared missing cells: **0**
- Corrected ppsf, yield, and future-value fields have no missing values: **True**

No machine-learning model was trained, no MLflow configuration was added, and no Streamlit application was built.
