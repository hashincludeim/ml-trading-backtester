"""End-to-end experiment: features -> split -> train every model -> evaluate.

This is the single loop over ``MODEL_REGISTRY`` that replaces the copy-pasted per-model blocks
of the original notebook. It is framework-agnostic: the CLI, notebooks and the Django
management command all call :func:`run_experiment`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from stockml.config import ExperimentConfig
from stockml.evaluation.metrics import ModelEvaluation, always_up_metrics, evaluate_model
from stockml.features.pipeline import build_feature_frame
from stockml.models.training import (
    Split,
    TrainedModel,
    chronological_split,
    train_model,
    validation_importance,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelOutcome:
    """Everything produced for one model: the fitted pipeline, test evaluation, importance."""

    trained: TrainedModel
    evaluation: ModelEvaluation
    importance: pd.DataFrame


@dataclass(frozen=True)
class ExperimentResult:
    """Outputs of :func:`run_experiment`."""

    split: Split
    outcomes: dict[str, ModelOutcome] = field(default_factory=dict)
    baseline: dict[str, float] = field(default_factory=dict)

    def predictions_frame(self) -> pd.DataFrame:
        """Test-set frame with ``y_true`` plus ``{model}_pred`` and ``{model}_score`` columns."""
        frame = pd.DataFrame({"y_true": self.split.y_test.astype(int)})
        for name, outcome in self.outcomes.items():
            frame[f"{name}_pred"] = outcome.evaluation.predictions
            frame[f"{name}_score"] = outcome.evaluation.scores
        return frame


def run_experiment(
    prices: pd.DataFrame, config: ExperimentConfig | None = None
) -> ExperimentResult:
    """Train and evaluate every configured model on one ticker's cleaned prices.

    Args:
        prices: Cleaned OHLCV frame.
        config: Feature and model settings.

    Returns:
        An :class:`ExperimentResult` whose outcomes share one chronological split.
    """
    cfg = config or ExperimentConfig()
    X, y = build_feature_frame(prices, cfg.features)
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
    for name in cfg.model.models:
        trained = train_model(name, split.X_train, split.y_train, cfg.model)
        evaluation = evaluate_model(trained.pipeline, split.X_test, split.y_test)
        importance = validation_importance(
            trained.pipeline, split.X_train, split.y_train, cfg.model
        )
        outcomes[name] = ModelOutcome(trained, evaluation, importance)
        logger.info("%s test accuracy %.3f", name, evaluation.metrics["accuracy"])
    return ExperimentResult(
        split=split, outcomes=outcomes, baseline=always_up_metrics(split.y_test)
    )
