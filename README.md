# World Cup 2026 Prediction Markets

Research-first data science project for aggregating prediction market odds, bookmaker odds, and football statistics to model probabilities and evaluate expected-value betting signals.

## Scope

- Primary objective: reproducible research and backtesting.
- Current implementation: data ingestion, canonical schema, feature build scaffold, model training scaffold, and backtest scaffold.
- Deferred: live automated trade execution.

## Stack

- Python managed with `uv`
- Package code in `src/worldcup_markets`
- Data layers in `data/bronze`, `data/silver`, `data/gold`
- Quarto RevealJS presentation in `docs/`
- Observable dashboard contract in `observable/`

## Project Structure

```text
conf/
	project.yaml
	providers.example.yaml
data/
	raw/ bronze/ silver/ gold/ external/
docs/
	_quarto.yml
	presentation.qmd
observable/
	README.md
scripts/
	bootstrap.py
	run_pipeline.py
	train_model.py
	backtest.py
src/worldcup_markets/
	data_sources/
	data_engineering/
	features/
	models/
	backtest/
	portfolio/
	reporting/
tests/
```

## Quickstart

1. Install dependencies:

```bash
uv sync
```

2. Prepare environment variables as needed:

```bash
export THE_ODDS_API_KEY="..."      # optional
export API_FOOTBALL_KEY="..."      # optional
export SPORTMONKS_TOKEN="..."      # optional
```

3. Bootstrap storage directories:

```bash
uv run python scripts/bootstrap.py
```

4. Run ingestion and feature build:

```bash
uv run python scripts/run_pipeline.py
```

5. Train starter model:

```bash
uv run python scripts/train_model.py
```

6. Run backtest scaffold:

```bash
uv run python scripts/backtest.py
```

7. Run tests:

```bash
uv run pytest
```

## Data Sources Included in Code

- Polymarket (Gamma endpoint public market fetch)
- football-data.co.uk World Cup historical odds CSVs (open, no API key)
- The Odds API (optional via API key)
- soccerdata (FBref schedule pull for bootstrap stats)

## Open-Data Backtest Path (No Keys Required)

You can run a full ingestion + holdout backtest using open sources only:

```bash
uv run python scripts/run_pipeline.py
uv run python scripts/backtest.py
```

This uses public Polymarket data and historical 1X2 odds/results from football-data.co.uk.

## Notes

- This implementation is an MVP scaffold for research and rapid iteration.
- Replace placeholder target generation with real labels before strategy evaluation.
- Expand feature engineering and calibration before relying on results for decision-making.