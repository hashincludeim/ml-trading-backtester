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
    FINAL_FIT_STAGE,
    WALK_FORWARD_STAGE,
    chronological_split,
    split_timeline,
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


def test_split_timeline_mirrors_split_folds_and_walk_forward() -> None:
    index = pd.bdate_range("2020-01-01", periods=100)
    timeline = split_timeline(index, n_train=80, n_splits=4, retrain_every=8)
    folds = timeline[timeline["stage"].str.startswith("CV fold")]
    assert list(dict.fromkeys(folds["stage"])) == [f"CV fold {i}" for i in range(1, 5)]
    for _, fold in folds.groupby("stage"):
        train, val = fold.iloc[0], fold.iloc[1]
        assert (train["role"], val["role"]) == ("train", "validation")
        assert train["start"] == index[0]  # expanding window
        assert index.get_loc(val["start"]) == index.get_loc(train["end"]) + 1
        assert val["end"] < index[80]  # CV never touches the test period
    final = timeline[timeline["stage"] == FINAL_FIT_STAGE].set_index("role")
    assert final.loc["train", "end"] == index[79]
    assert final.loc["test", "start"] == index[80]
    assert final.loc["test", "end"] == index[-1]
    assert final["n_rows"].sum() == len(index)
    blocks = timeline[(timeline["stage"] == WALK_FORWARD_STAGE) & (timeline["role"] == "test")]
    assert list(blocks["start"]) == list(index[80::8])
    assert blocks["n_rows"].tolist() == [8, 8, 4]  # the last block is cut at the final day


def test_split_timeline_without_walk_forward_and_bad_sizes() -> None:
    index = pd.bdate_range("2020-01-01", periods=50)
    assert WALK_FORWARD_STAGE not in set(split_timeline(index, 40, 3)["stage"])
    with pytest.raises(ValueError):
        split_timeline(index, n_train=50, n_splits=3)
