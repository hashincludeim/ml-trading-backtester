"""End-to-end experiment: features -> split -> train every model -> evaluate.

This is the single loop over ``MODEL_REGISTRY`` that replaces the copy-pasted per-model blocks
of the original notebook. Optionally each model is also scored walk-forward over the same test
period, refitted on an expanding window with the hyper-parameters chosen on the training set.
It is framework-agnostic: the CLI, notebooks and the Django management command all call
:func:`run_experiment`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from stockml.config import ExperimentConfig
from stockml.evaluation.metrics import (
    ModelEvaluation,
    always_up_metrics,
    evaluate_model,
    evaluate_predictions,
)
from stockml.features.pipeline import build_feature_frame
from stockml.features.target import volatility_persistence_score
from stockml.models.training import (
    Split,
    TrainedModel,
    chronological_split,
    train_model,
    validation_importance,
)
from stockml.models.walk_forward import walk_forward_predict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelOutcome:
    """Everything produced for one model: the fitted pipeline, test evaluation, importance.

    ``walk_forward`` scores the same test rows with periodic refits; it is ``None`` when
    walk-forward evaluation is switched off. The backtest uses ``evaluation`` (the saved model).
    """

    trained: TrainedModel
    evaluation: ModelEvaluation
    importance: pd.DataFrame
    walk_forward: ModelEvaluation | None = None


@dataclass(frozen=True)
class ExperimentResult:
    """Outputs of :func:`run_experiment`.

    ``baseline`` is the always-positive classifier. ``heuristic`` holds test metrics for a
    no-model rule when the target has one (the volatility target's recent-volatility rule).
    """

    split: Split
    outcomes: dict[str, ModelOutcome] = field(default_factory=dict)
    baseline: dict[str, float] = field(default_factory=dict)
    refit_dates: tuple[pd.Timestamp, ...] = ()
    heuristic: dict[str, float] = field(default_factory=dict)

    def predictions_frame(self) -> pd.DataFrame:
        """Test-set frame with ``y_true`` plus ``{model}_pred`` and ``{model}_score`` columns.

        Walk-forward predictions, when present, add ``{model}_wf_pred`` and ``{model}_wf_score``.
        """
        frame = pd.DataFrame({"y_true": self.split.y_test.astype(int)})
        for name, outcome in self.outcomes.items():
            frame[f"{name}_pred"] = outcome.evaluation.predictions
            frame[f"{name}_score"] = outcome.evaluation.scores
            if outcome.walk_forward is not None:
                frame[f"{name}_wf_pred"] = outcome.walk_forward.predictions
                frame[f"{name}_wf_score"] = outcome.walk_forward.scores
        return frame


def run_experiment(
    prices: pd.DataFrame, config: ExperimentConfig | None = None
) -> ExperimentResult:
    """Train and evaluate every configured model on one ticker's cleaned prices.

    Args:
        prices: Cleaned OHLCV frame.
        config: Feature, model and target settings.

    Returns:
        An :class:`ExperimentResult` whose outcomes share one chronological split.
    """
    cfg = config or ExperimentConfig()
    X, y = build_feature_frame(prices, cfg.features, cfg.target)
    split = chronological_split(X, y, cfg.model.test_size)
    logger.info(
        "Split: train %s..%s (%d), test %s..%s (%d)",
        split.X_train.index[0].date(),
        split.X_train.index[-1].date(),
        len(split.X_train),
        split.X_test.index[0].date(),
        split.X_test.index[-1].date(),
        len(split.X_test),
    )
    outcomes: dict[str, ModelOutcome] = {}
    refit_dates: tuple[pd.Timestamp, ...] = ()
    for name in cfg.model.models:
        trained = train_model(name, split.X_train, split.y_train, cfg.model)
        evaluation = evaluate_model(trained.pipeline, split.X_test, split.y_test)
        importance = validation_importance(
            trained.pipeline, split.X_train, split.y_train, cfg.model
        )
        walk_forward = None
        if cfg.model.walk_forward:
            wf = walk_forward_predict(
                trained.pipeline, X, y, len(split.X_train), cfg.model.retrain_every
            )
            walk_forward = evaluate_predictions(split.y_test, wf.predictions, wf.scores)
            refit_dates = wf.refit_dates
            logger.info(
                "%s walk-forward ROC AUC %.3f (%d refits)",
                name,
                walk_forward.metrics["roc_auc"],
                len(wf.refit_dates),
            )
        outcomes[name] = ModelOutcome(trained, evaluation, importance, walk_forward)
        logger.info("%s test accuracy %.3f", name, evaluation.metrics["accuracy"])
    heuristic: dict[str, float] = {}
    if cfg.target.kind == "volatility":
        score = volatility_persistence_score(prices["Close"], cfg.target).loc[split.X_test.index]
        rule = (score > 1.0).astype(int).rename("prediction")
        heuristic = evaluate_predictions(split.y_test, rule, score).metrics
        logger.info("Recent-volatility rule ROC AUC %.3f", heuristic["roc_auc"])
    return ExperimentResult(
        split=split,
        outcomes=outcomes,
        baseline=always_up_metrics(split.y_test),
        refit_dates=refit_dates,
        heuristic=heuristic,
    )
