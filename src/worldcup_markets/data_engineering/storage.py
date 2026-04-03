from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from worldcup_markets.config.settings import StorageConfig


def ensure_storage_dirs(cfg: StorageConfig) -> None:
    for directory in [cfg.bronze_path, cfg.silver_path, cfg.gold_path, cfg.db_path.parent]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def upsert_to_duckdb(df: pd.DataFrame, db_path: str | Path, table_name: str) -> None:
    with duckdb.connect(str(db_path)) as con:
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
