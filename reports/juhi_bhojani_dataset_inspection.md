# Juhi Bhojani Kaggle Dataset Inspection

Inspection date: 2026-10-25

## Source and license

- Original Kaggle URL: <https://www.kaggle.com/datasets/juhibhojani/house-price>
- Dataset title: **House Price**
- Author: **Juhi Bhojani**
- Kaggle page status: publicly viewable; download requires Kaggle access
- Kaggle-reported update: approximately 3 years ago
- Kaggle-reported license: **Community Data License Agreement - Sharing - Version 1.0**
- Kaggle-reported file: `house_prices.csv`
- Kaggle-reported file size: approximately **106.15 MB**
- Kaggle-reported schema size: **21 columns**
- Kaggle-reported expected update frequency: Never
- Reported source script: <https://github.com/Juhibhojani/Web-Scraping>

The Kaggle description says the data contains property listings and asking-price-related fields. Listing prices are not necessarily completed transaction prices, so this dataset must not be described as verified transaction data.

License note: the Community Data License Agreement - Sharing - Version 1.0 is the license displayed by Kaggle. Its sharing and attribution conditions should be read before redistributing the file or publishing derived data.

## Download status

The project folder was inspected before attempting the download.

Existing relevant files:

- `data/raw/india_housing_prices.csv` exists and was not changed.
- `data/raw/juhi_bhojani/` did not exist and was not created because the authorized download could not begin.

The download was **not completed**. No archive or extracted files from the Juhi Bhojani dataset exist in this project.

Reason: the workspace has neither:

- an available `kaggle` command-line tool, nor
- a configured Kaggle authentication file such as `%USERPROFILE%\\.kaggle\\kaggle.json`.

No unofficial mirror was used. No Kaggle password, API token, or other secret was requested or pasted into chat.

## Actual files and schema

Because the original Kaggle file was not downloaded, there are no local Juhi Bhojani files to inspect. Consequently, the following values are intentionally **not claimed**:

- Local archive filename and size
- Extracted filename and size
- Actual row count
- Actual column count beyond Kaggle's page summary
- Actual pandas data types
- Actual missing-value counts
- Actual duplicate-row count
- Sample rows
- Confirmed price and area column names or units

The Kaggle page describes fields including carpet area, property status, floor, transaction type, furnishing, facing, overlooking, society name, bathrooms, balconies, parking, ownership type, super area, dimensions, plot area, property title, total amount, price per square foot, and location description. These are published-description hints only; they are not a substitute for reading the downloaded CSV header.

## Simple instructions to enable the authorized download later

1. Sign in to Kaggle in a browser using your own account.
2. Configure the official Kaggle CLI authentication on the computer. Do not paste credentials or tokens into chat. Kaggle's current authentication instructions are available at <https://www.kaggle.com/docs/api>.
3. Verify that the CLI can authenticate locally.
4. Ask the project assistant to retry, or run the official download command locally:

```powershell
New-Item -ItemType Directory -Path data\raw\juhi_bhojani -Force
kaggle datasets download -d juhibhojani/house-price -p data\raw\juhi_bhojani
```

After the archive is downloaded, preserve it and extract into a separate subfolder without deleting the archive:

```powershell
Expand-Archive -LiteralPath data\raw\juhi_bhojani\house-price.zip -DestinationPath data\raw\juhi_bhojani\extracted
```

The actual archive name may differ if Kaggle changes its naming convention. Use the filename shown by the download command rather than assuming `house-price.zip`.

## Recommendation

This dataset is worth inspecting in a later step because it appears to contain richer listing attributes than the current 250,000-row dataset. However, it should be treated as scraped/listing data with asking prices, not as verified sale transactions.

Do not clean, merge, or model it until the original archive and extracted CSV have been downloaded and their actual schema, units, missing values, duplicates, and license details have been checked locally.

No existing EDA report, dataset, model, MLflow configuration, or Streamlit application was modified.
