from __future__ import annotations

import pandas as pd
import pytest

from stockml.features.pipeline import build_feature_frame
from stockml.models.registry import build_pipeline
from stockml.models.walk_forward import walk_forward_predict
from tests.conftest import make_prices


@pytest.fixture
def xy() -> tuple[pd.DataFrame, pd.Series]:
    return build_feature_frame(make_prices(400, seed=3))


def test_covers_every_test_row_with_one_refit_per_block(
    xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = xy
    start = 250
    wf = walk_forward_predict(build_pipeline("logistic_regression"), X, y, start, 40)
    assert wf.predictions.index.equals(X.index[start:])
    assert wf.scores.index.equals(X.index[start:])
    expected_blocks = -(-(len(X) - start) // 40)  # ceiling division
    assert len(wf.refit_dates) == expected_blocks
    assert wf.refit_dates[0] == X.index[start]
    assert wf.refit_dates[1] == X.index[start + 40]
    assert set(wf.predictions.unique()) <= {0, 1}


def test_single_block_matches_a_model_fitted_once(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = xy
    start = 250
    pipe = build_pipeline("logistic_regression")
    wf = walk_forward_predict(pipe, X, y, start, retrain_every=len(X))
    once = build_pipeline("logistic_regression").fit(X.iloc[:start], y.iloc[:start])
    assert len(wf.refit_dates) == 1
    assert (wf.predictions.to_numpy() == once.predict(X.iloc[start:])).all()


def test_no_leakage_future_rows_do_not_change_earlier_blocks(
    xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = xy
    start, every = 250, 30
    pipe = build_pipeline("logistic_regression")
    base = walk_forward_predict(pipe, X, y, start, every)
    cut = start + every  # rows from here on belong to the second block or later
    X_alt, y_alt = X.copy(), y.copy()
    X_alt.iloc[cut:] = X_alt.iloc[cut:] * 10 + 3
    y_alt.iloc[cut:] = 1 - y_alt.iloc[cut:]
    altered = walk_forward_predict(pipe, X_alt, y_alt, start, every)
    first_block = X.index[start:cut]
    pd.testing.assert_series_equal(base.scores[first_block], altered.scores[first_block])
    assert not base.scores.equals(altered.scores)  # later blocks did see the change


def test_rejects_bad_arguments(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = xy
    pipe = build_pipeline("logistic_regression")
    with pytest.raises(ValueError, match="test_start"):
        walk_forward_predict(pipe, X, y, 0, 10)
    with pytest.raises(ValueError, match="test_start"):
        walk_forward_predict(pipe, X, y, len(X), 10)
    with pytest.raises(ValueError, match="retrain_every"):
        walk_forward_predict(pipe, X, y, 100, 0)
    with pytest.raises(ValueError, match="index"):
        walk_forward_predict(pipe, X, y.iloc[1:], 100, 10)
