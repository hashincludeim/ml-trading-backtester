from __future__ import annotations

import datetime as dt

from stockml.config import (
    DataConfig,
    config_hash,
    currency_symbol,
    price_unit,
    ticker_label,
    today_inclusive_end,
)


def test_default_universe_is_sp500_and_big_tech() -> None:
    assert DataConfig().tickers == ("^GSPC", "AMZN", "MSFT", "GOOGL", "ORCL")


def test_default_end_includes_today() -> None:
    # yfinance's end is exclusive, so tomorrow's date means "up to and including today".
    assert DataConfig().end == today_inclusive_end()
    assert dt.date.fromisoformat(today_inclusive_end()) == dt.date.today() + dt.timedelta(days=1)


def test_labels_units_and_currency() -> None:
    assert ticker_label("^GSPC") == "S&P 500"
    assert ticker_label("GOOGL") == "Alphabet (Google)"
    assert ticker_label("XYZ") == "XYZ"
    assert price_unit("^GSPC") == "points"
    assert price_unit("MSFT") == "USD"
    assert price_unit("BARC.L") == "GBX"
    assert currency_symbol("MSFT") == "$"
    assert currency_symbol("BARC.L") == "£"


def test_config_hash_is_stable() -> None:
    cfg = DataConfig(end="2025-01-01")
    assert config_hash(cfg) == config_hash(DataConfig(end="2025-01-01"))
