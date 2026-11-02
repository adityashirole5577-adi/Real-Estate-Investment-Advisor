# Phase 2.5: Data Suitability Check

## Executive conclusion

The cleaned file is technically usable for an educational current-price experiment, but it does not provide strong evidence for a trustworthy investment classifier or a genuine five-year property-value forecasting model.

The strongest measured warning is that the data has almost no relationship between size and price: Pearson correlation is **-0.0025** and Spearman correlation is **-0.0025**. This is not a calculation mistake. The price-per-square-foot calculation is internally consistent, but it is derived from price and size. The pattern appears to be a property of the supplied data, which is consistent with the earlier provenance concerns.

## Dataset integrity and preservation

- Raw rows: **250,000**
- Cleaned rows: **250,000**
- Raw SHA-256 unchanged during this audit: **True**
- Cleaned SHA-256 unchanged during this audit: **True**
- Raw and cleaned row counts match: **True**

No dataset file was written by this script.

## 1. Variation and data-quality observations

### Numerical variation

The least-variable numerical features by number of distinct values are:

- `BHK`: 5 unique values; range 1.00 to 5.00
- `Nearby_Hospitals`: 10 unique values; range 1.00 to 10.00
- `Nearby_Schools`: 10 unique values; range 1.00 to 10.00
- `Total_Floors`: 30 unique values; range 1.00 to 30.00
- `Floor_No`: 31 unique values; range 0.00 to 30.00
- `Year_Built`: 34 unique values; range 1,990.00 to 2,023.00
- `Age_of_Property`: 34 unique values; range 2.00 to 35.00
- `Size_in_SqFt`: 4,501 unique values; range 500.00 to 5,000.00

Repeated-value inspection shows these largest single-value shares among the numerical features:

- `BHK`: value `1` appears 50,196 times (20.08%)
- `Nearby_Schools`: value `1` appears 25,360 times (10.14%)
- `Nearby_Hospitals`: value `4` appears 25,168 times (10.07%)
- `Total_Floors`: value `24` appears 8,520 times (3.41%)
- `Floor_No`: value `20` appears 8,283 times (3.31%)
- `Year_Built`: value `2023` appears 7,568 times (3.03%)
- `Age_of_Property`: value `2` appears 7,568 times (3.03%)
- `Size_in_SqFt`: value `2098` appears 86 times (0.03%)

There are no constant numerical columns, but the ranges are unusually neat: BHK is 1–5, size is 500–5,000 sq ft, price is 10–500 lakhs, and price per sq ft is reported to two decimal places. These patterns are not automatically wrong, but they support treating the file as possibly simulated or heavily constrained.

### Categorical variation

- `Parking_Space`: 2 unique values; most common value `No` represents 50.18%
- `Security`: 2 unique values; most common value `Yes` represents 50.09%
- `Availability_Status`: 2 unique values; most common value `Under_Construction` represents 50.01%
- `Property_Type`: 3 unique values; most common value `Villa` represents 33.50%
- `Furnished_Status`: 3 unique values; most common value `Unfurnished` represents 33.36%
- `Public_Transport_Accessibility`: 3 unique values; most common value `High` represents 33.48%
- `Owner_Type`: 3 unique values; most common value `Broker` represents 33.39%
- `Facing`: 4 unique values; most common value `West` represents 25.10%
- `State`: 20 unique values; most common value `Odisha` represents 5.07%
- `City`: 42 unique values; most common value `Coimbatore` represents 2.58%
- `Amenities`: 325 unique values; most common value `Pool` represents 4.09%
- `Locality`: 500 unique values; most common value `Locality_296` represents 0.23%

All locality names match `Locality_<number>`. This is artificial-looking and removes meaningful locality identity. It is evidence of generated or anonymized labels, not proof of synthetic generation.

### Relationships and consistency checks

- Size versus price: Pearson **-0.0025**; Spearman **-0.0025**.
- Size-price Pearson correlations by property type: Apartment: -0.0044, Independent House: 0.0049, Villa: -0.0080.
- Across cities, the minimum and maximum size-price correlations are **-0.0145** and **0.0119**.
- `Price_per_SqFt` agrees with `Price_in_Lakhs / Size_in_SqFt` within a two-decimal rounding tolerance for **250,000 / 250,000** rows.
- `Age_of_Property` equals `2025 - Year_Built` for **250,000 / 250,000** rows.
- Existing floor flags mark **116,304 / 250,000** rows where `Floor_No > Total_Floors`.
- Repeated property-description combinations (State, City, Locality, type, BHK, size, and year) occur in **1** rows after the first occurrence. This is not proof of repeated observations because there is no property identifier beyond the unique ID.

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
| Naive mean baseline | 122.3178 | 141.1932 | -0.0000 |
| Leakage-safe Ridge pipeline | 122.4366 | 141.4584 | -0.0038 |

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
