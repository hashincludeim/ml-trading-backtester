"""Chronological splitting, time-series cross-validation, tuning, and fitting."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline

from stockml.config import ModelConfig
from stockml.models.registry import build_pipeline, get_spec

logger = logging.getLogger(__name__)

CV_SCORING: tuple[str, ...] = ("accuracy", "roc_auc", "f1")


@dataclass(frozen=True)
class Split:
    """A chronological train/test split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class CVResult:
    """Per-fold time-series cross-validation scores keyed by metric name."""

    scores: dict[str, list[float]]

    def mean(self, metric: str) -> float:
        """Mean score across folds."""
        return float(np.mean(self.scores[metric]))

    def std(self, metric: str) -> float:
        """Population standard deviation across folds."""
        return float(np.std(self.scores[metric]))


@dataclass(frozen=True)
class TrainedModel:
    """A fitted pipeline plus how it was selected."""

    name: str
    pipeline: Pipeline
    cv: CVResult
    best_params: dict[str, object]


def chronological_split(X: pd.DataFrame, y: pd.Series, test_size: float) -> Split:
    """Split without shuffling: the last ``test_size`` fraction of rows is the test set.

    Args:
        X: Features in chronological order.
        y: Target aligned with ``X``.
        test_size: Fraction of rows in ``(0, 1)`` to hold out.

    Returns:
        A :class:`Split` where every training date precedes every test date.

    Raises:
        ValueError: If ``test_size`` is out of range or either side would be empty.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same index")
    n_test = int(np.ceil(len(X) * test_size))
    n_train = len(X) - n_test
    if n_train < 1 or n_test < 1:
        raise ValueError("Not enough rows for a train/test split")
    return Split(X.iloc[:n_train], X.iloc[n_train:], y.iloc[:n_train], y.iloc[n_train:])


def time_series_cv(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> CVResult:
    """Cross-validate with expanding-window ``TimeSeriesSplit`` (never shuffled KFold).

    Args:
        pipeline: Unfitted pipeline; it is cloned per fold so preprocessing sees training folds
            only.
        X: Training features, chronological.
        y: Training target.
        n_splits: Number of folds.

    Returns:
        Fold scores for accuracy, ROC AUC and F1.
    """
    res = cross_validate(
        pipeline, X, y, cv=TimeSeriesSplit(n_splits=n_splits), scoring=list(CV_SCORING)
    )
    return CVResult({m: [float(v) for v in res[f"test_{m}"]] for m in CV_SCORING})


def train_model(
    name: str, X_train: pd.DataFrame, y_train: pd.Series, config: ModelConfig
) -> TrainedModel:
    """Tune (optionally), cross-validate, and fit one registered model on the training set.

    Tuning uses ``GridSearchCV`` with ``TimeSeriesSplit`` on the training data only. The CV
    scores reported are for the selected configuration, and the returned pipeline is that same
    configuration refitted on all training rows.

    Args:
        name: Registry key.
        X_train: Training features.
        y_train: Training target.
        config: Model settings.

    Returns:
        A :class:`TrainedModel` with the fitted pipeline.
    """
    spec = get_spec(name)
    pipeline = build_pipeline(name, config)
    best_params: dict[str, object] = {}
    if config.tune and spec.param_grid:
        search = GridSearchCV(
            pipeline,
            spec.param_grid,
            cv=TimeSeriesSplit(n_splits=config.cv_splits),
            scoring="roc_auc",
            n_jobs=config.n_jobs,
        )
        search.fit(X_train, y_train)
        best_params = dict(search.best_params_)
        pipeline = clone(pipeline).set_params(**best_params)
        logger.info("%s best params: %s", name, best_params)
    cv = time_series_cv(pipeline, X_train, y_train, config.cv_splits)
    fitted = clone(pipeline).fit(X_train, y_train)
    logger.info("%s CV accuracy %.3f ± %.3f", name, cv.mean("accuracy"), cv.std("accuracy"))
    return TrainedModel(name=name, pipeline=fitted, cv=cv, best_params=best_params)


def validation_importance(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelConfig,
) -> pd.DataFrame:
    """Permutation importance on a held-out *validation* fold of the training data.

    A clone of ``pipeline`` is fitted on the earlier training rows and importance is measured on
    the last ``TimeSeriesSplit`` fold, so the test set is never used for feature analysis.
    Importance is computed on the original (named) features even when the pipeline contains PCA.

    Args:
        pipeline: Pipeline with the chosen hyper-parameters (fitted or not).
        X_train: Training features.
        y_train: Training target.
        config: Supplies ``cv_splits``, ``importance_repeats`` and ``random_state``.

    Returns:
        Frame indexed by feature name with ``importance_mean`` and ``importance_std`` (drop in
        ROC AUC when the feature is shuffled), sorted descending.
    """
    fit_idx, val_idx = list(TimeSeriesSplit(n_splits=config.cv_splits).split(X_train))[-1]
    model = clone(pipeline).fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
    result = permutation_importance(
        model,
        X_train.iloc[val_idx],
        y_train.iloc[val_idx],
        scoring="roc_auc",
        n_repeats=config.importance_repeats,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )
    return pd.DataFrame(
        {"importance_mean": result.importances_mean, "importance_std": result.importances_std},
        index=pd.Index(X_train.columns, name="feature"),
    ).sort_values("importance_mean", ascending=False)
