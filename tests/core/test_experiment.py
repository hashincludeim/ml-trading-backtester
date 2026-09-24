from __future__ import annotations

from stockml.config import ExperimentConfig, ModelConfig, TargetConfig
from stockml.experiment import run_experiment
from tests.conftest import make_prices


def test_run_experiment_small() -> None:
    cfg = ExperimentConfig(
        model=ModelConfig(
            models=("logistic_regression", "linear_svc"),
            cv_splits=3,
            tune=False,
            importance_repeats=2,
            walk_forward=False,
        )
    )
    result = run_experiment(make_prices(400, seed=5), cfg)
    assert set(result.outcomes) == {"logistic_regression", "linear_svc"}
    frame = result.predictions_frame()
    assert list(frame.columns) == [
        "y_true",
        "logistic_regression_pred",
        "logistic_regression_score",
        "linear_svc_pred",
        "linear_svc_score",
    ]
    assert frame.index.equals(result.split.X_test.index)
    assert 0 <= result.baseline["accuracy"] <= 1
    assert result.outcomes["linear_svc"].walk_forward is None
    assert result.refit_dates == ()


def test_run_experiment_walk_forward_scores_the_same_test_rows() -> None:
    cfg = ExperimentConfig(
        model=ModelConfig(
            models=("logistic_regression",),
            cv_splits=3,
            tune=False,
            importance_repeats=2,
            retrain_every=20,
        )
    )
    result = run_experiment(make_prices(400, seed=5), cfg)
    wf = result.outcomes["logistic_regression"].walk_forward
    assert wf is not None
    assert wf.predictions.index.equals(result.split.X_test.index)
    assert len(result.refit_dates) == -(-len(result.split.X_test) // 20)
    assert result.refit_dates[0] == result.split.X_test.index[0]
    frame = result.predictions_frame()
    assert {"logistic_regression_wf_pred", "logistic_regression_wf_score"} <= set(frame.columns)


def test_run_experiment_volatility_target_reports_the_no_model_rule() -> None:
    cfg = ExperimentConfig(
        model=ModelConfig(
            models=("logistic_regression",),
            cv_splits=3,
            tune=False,
            importance_repeats=2,
            walk_forward=False,
        ),
        target=TargetConfig(kind="volatility", volatility_window=60),
    )
    result = run_experiment(make_prices(400, seed=5), cfg)
    assert set(result.heuristic) >= {"accuracy", "roc_auc"}
    assert 0 <= result.heuristic["roc_auc"] <= 1
    assert result.split.y_test.name == "big_move"


def test_direction_target_has_no_heuristic() -> None:
    cfg = ExperimentConfig(
        model=ModelConfig(
            models=("logistic_regression",), cv_splits=3, tune=False, walk_forward=False
        )
    )
    assert run_experiment(make_prices(300, seed=5), cfg).heuristic == {}
