"""Walk-forward evaluation: refit on an expanding window as the test period unfolds.

A single train/test split fits a model once and then asks it to predict years ahead. Walk-forward
evaluation is closer to how a model would really be used: every ``retrain_every`` rows it is
refitted on all rows seen so far and predicts only the next block.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from stockml.evaluation.metrics import decision_scores

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardResult:
    """Out-of-sample predictions stitched together from successive refits.

    Attributes:
        predictions: Predicted class for every row from ``test_start`` onwards.
        scores: Continuous score for the same rows (P(up) or a decision function).
        refit_dates: First date predicted by each refitted model.
    """

    predictions: pd.Series
    scores: pd.Series
    refit_dates: tuple[pd.Timestamp, ...]


def block_starts(n_rows: int, test_start: int, retrain_every: int) -> range:
    """Row positions where walk-forward refits: the first row of each prediction block.

    Args:
        n_rows: Rows in the whole history.
        test_start: Position of the first row to predict.
        retrain_every: Rows predicted by each fitted model before it is refitted.
    """
    return range(test_start, n_rows, retrain_every)


def walk_forward_predict(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    test_start: int,
    retrain_every: int,
) -> WalkForwardResult:
    """Predict rows ``test_start:`` in blocks, refitting a clone before each block.

    The model for the block starting at row ``s`` is fitted on rows ``0..s-1`` only. The last of
    those rows is labelled with the close of day ``s``, which is known at the close of day ``s``
    when the prediction for ``s`` is made, so no future information is used. Hyper-parameters
    come from ``pipeline`` and are not re-tuned.

    For models without ``predict_proba`` the scores are decision-function values, whose scale
    can shift slightly between refits.

    Args:
        pipeline: Pipeline with the chosen hyper-parameters (fitted or not; it is cloned).
        X: Features for the whole history, chronological.
        y: Target aligned with ``X``.
        test_start: Position of the first row to predict.
        retrain_every: Rows predicted by each fitted model before it is refitted.

    Returns:
        A :class:`WalkForwardResult` covering rows ``test_start`` to the end.

    Raises:
        ValueError: If the indexes differ, ``test_start`` leaves no training or test rows, or
            ``retrain_every`` is below 1.
    """
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same index")
    if not 0 < test_start < len(X):
        raise ValueError("test_start must leave at least one training and one test row")
    if retrain_every < 1:
        raise ValueError("retrain_every must be at least 1")
    predictions: list[pd.Series] = []
    scores: list[pd.Series] = []
    refit_dates: list[pd.Timestamp] = []
    for start in block_starts(len(X), test_start, retrain_every):
        block = X.iloc[start : start + retrain_every]
        model = clone(pipeline).fit(X.iloc[:start], y.iloc[:start])
        predictions.append(pd.Series(model.predict(block).astype(int), index=block.index))
        scores.append(decision_scores(model, block))
        refit_dates.append(pd.Timestamp(block.index[0]))
    logger.debug("walk_forward_predict: %d refits", len(refit_dates))
    return WalkForwardResult(
        predictions=pd.concat(predictions).rename("prediction"),
        scores=pd.concat(scores).rename("score"),
        refit_dates=tuple(refit_dates),
    )
