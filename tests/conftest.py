"""Shared fixtures: small synthetic price frames (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_prices(n: int = 300, seed: int = 0, start: str = "2020-01-01") -> pd.DataFrame:
    """Random-walk OHLCV frame on business days."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n, name="Date")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.integers(1_000, 10_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx
    )


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices()
