"""
Helpers available to custom backtesting strategies.

Custom strategy scripts are executed with:
    BACKTESTING_DIR
    BACKTESTING_DATA_DIR
    BACKTEST_RESULT_DIR
    BACKTEST_CONFIG_JSON

The recommended contract is:
1. Read config with `get_config`.
2. Read market data from `DATA_DIR` or via the shared engine helpers.
3. Save standardized outputs with `save_results`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


BACKTESTING_DIR = Path(
    os.environ.get("BACKTESTING_DIR", Path(__file__).resolve().parent)
).resolve()
DATA_DIR = Path(
    os.environ.get("BACKTESTING_DATA_DIR", BACKTESTING_DIR / "data")
).resolve()
RESULT_DIR = Path(
    os.environ.get("BACKTEST_RESULT_DIR", BACKTESTING_DIR / "results" / "adhoc")
).resolve()


def get_config(default_config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(default_config or {})
    raw = os.environ.get("BACKTEST_CONFIG_JSON", "{}")
    try:
        config.update(json.loads(raw))
    except json.JSONDecodeError:
        pass
    return config


def ensure_result_dir() -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR


def save_json(name: str, payload: Any) -> Path:
    target = ensure_result_dir() / name
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return target


def save_dataframe(name: str, df: pd.DataFrame) -> Path:
    target = ensure_result_dir() / name
    df.to_csv(target, index=False)
    return target


def save_results(
    trades_df: pd.DataFrame | None = None,
    stats: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, str]:
    trades_df = trades_df if trades_df is not None else pd.DataFrame()
    stats = stats or {}
    config = config or get_config()

    return {
        "config": str(save_json("config.json", config)),
        "stats": str(save_json("stats.json", stats)),
        "trades": str(save_dataframe("trades.csv", trades_df)),
    }
