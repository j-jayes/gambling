# Observable Dashboard

This folder contains the data contract and starter artifacts for an Observable notebook dashboard.

## Recommended dashboard views

1. Bankroll and cumulative PnL over time.
2. Signal quality: edge distribution and realized return by edge bucket.
3. Market attribution: 1X2 vs outrights vs goals markets.
4. Team and stage attribution.
5. Calibration panel: predicted vs realized bins.

## Data inputs

Export these parquet/csv files from the pipeline:

- `data/gold/backtest_results.parquet`
- `data/silver/features.parquet`
- `data/bronze/quotes.parquet`

Then convert to CSV or Arrow for Observable ingestion.

## Holdout Outcomes Notebook

This repository now includes an Observable-style notebook page at:

- holdout-outcomes.md

And a data attachment expected at:

- data/backtest_results.csv

To refresh data before opening the notebook:

1. Run the pipeline and backtest in the project root.
2. Export data/gold/backtest_results.parquet to observable/data/backtest_results.csv.

The notebook includes:

- Model probability distribution.
- Reliability (calibration) chart.
- Cutoff-based confusion outcomes.
- Interactive edge-threshold PnL sensitivity.
- Row-level inspection table.
