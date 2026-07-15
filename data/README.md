# Dataset Access

The aspect-level ABSA dataset (51,098 instances from 37,790 cleaned Gojek reviews)
is not committed to this repository due to size. To obtain it:

1. Re-run `notebooks/01_scraping_eda_cleaning.ipynb` (scrapes 50,000 reviews from
   Google Play Store for `com.gojek.app`, then cleans them), or
2. Request access via the Google Drive link listed in the E-Repository form.

Columns: `review_id`, `review_text`, `aspect`, `sentiment` (positive/negative/neutral).
