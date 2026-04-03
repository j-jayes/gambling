from __future__ import annotations

from worldcup_markets.config.settings import load_app_config
from worldcup_markets.data_engineering.storage import ensure_storage_dirs
from worldcup_markets.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    cfg = load_app_config()
    ensure_storage_dirs(cfg.storage)
    print("Bootstrap complete. Storage directories are ready.")


if __name__ == "__main__":
    main()
