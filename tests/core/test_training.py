from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from stockml.config import ModelConfig
from stockml.features.pipeline import build_feature_frame
from stockml.models.persistence import load_model, save_model
from stockml.models.registry import build_pipeline
from stockml.models.training import (
    chronological_split,
    time_series_cv,
    train_model,
    validation_importance,
)
from tests.conftest import make_prices

FAST = ModelConfig(cv_splits=3, tune=True, importance_repeats=2)


@pytest.fixture
def xy() -> tuple[pd.DataFrame, pd.Series]:
    return build_feature_frame(make_prices(400, seed=2))


def test_chronological_split_has_no_overlap(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = xy
    split = chronological_split(X, y, 0.25)
    assert split.X_train.index.max() < split.X_test.index.min()
    assert len(split.X_train) + len(split.X_test) == len(X)
    assert len(split.X_test) == pytest.approx(0.25 * len(X), abs=1)


def test_chronological_split_rejects_bad_size(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    with pytest.raises(ValueError):
        chronological_split(*xy, test_size=1.5)


def test_time_series_cv_scores(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = xy
    cv = time_series_cv(build_pipeline("logistic_regression"), X, y, n_splits=3)
    assert set(cv.scores) == {"accuracy", "roc_auc", "f1"}
    assert len(cv.scores["accuracy"]) == 3
    assert 0 <= cv.mean("accuracy") <= 1


def test_train_model_tunes_and_fits(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = xy
    trained = train_model("logistic_regression", X, y, FAST)
    assert "model__C" in trained.best_params
    assert trained.best_params["model__C"] == trained.pipeline.named_steps["model"].C
    assert len(trained.pipeline.predict(X)) == len(X)


def test_validation_importance_uses_named_features(xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = xy
    imp = validation_importance(build_pipeline("bagging_knn"), X, y, FAST)
    assert set(imp.index) == set(X.columns)  # mapped to original names despite PCA
    assert list(imp.columns) == ["importance_mean", "importance_std"]


def test_persistence_roundtrip(tmp_path: Path, xy: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = xy
    fitted = build_pipeline("logistic_regression").fit(X, y)
    path = save_model(fitted, tmp_path / "m" / "model.joblib")
    loaded = load_model(path)
    assert isinstance(loaded, Pipeline)
    assert (loaded.predict(X) == fitted.predict(X)).all()
