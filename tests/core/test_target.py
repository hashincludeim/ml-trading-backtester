from __future__ import annotations

import pandas as pd

from stockml.features.target import make_price_rise_target


def test_target_compares_next_close() -> None:
    close = pd.Series([1.0, 2.0, 1.5, 1.5, 3.0])
    y = make_price_rise_target(close)
    assert y.iloc[:4].tolist() == [1, 0, 0, 1]
    assert pd.isna(y.iloc[-1])
    assert y.name == "price_rise"
