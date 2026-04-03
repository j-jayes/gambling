from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageConfig(BaseModel):
    bronze_path: Path
    silver_path: Path
    gold_path: Path
    db_path: Path


class ModelingConfig(BaseModel):
    random_seed: int = 42
    train_start_date: str
    validation_start_date: str
    test_start_date: str


class BacktestConfig(BaseModel):
    min_edge: float = 0.02
    max_position_fraction: float = 0.03
    kelly_fraction: float = 0.5


class AppConfig(BaseModel):
    project: dict[str, object]
    pipeline: dict[str, object]
    storage: StorageConfig
    modeling: ModelingConfig
    backtest: BacktestConfig


class Secrets(BaseSettings):
    the_odds_api_key: str | None = Field(default=None, alias="THE_ODDS_API_KEY")
    api_football_key: str | None = Field(default=None, alias="API_FOOTBALL_KEY")
    sportmonks_token: str | None = Field(default=None, alias="SPORTMONKS_TOKEN")
    polymarket_private_key: str | None = Field(default=None, alias="POLYMARKET_PRIVATE_KEY")

    model_config = SettingsConfigDict(extra="ignore")


def load_app_config(path: str | Path = "conf/project.yaml") -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return AppConfig.model_validate(payload)
