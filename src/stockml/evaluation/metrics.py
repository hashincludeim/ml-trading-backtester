"""Classification and risk/return metrics.

Risk metrics expect *daily log returns*. Annualisation uses ``periods_per_year`` (252 trading
days by default): Sharpe and Sortino scale by ``sqrt(252)``; annualised return is
``exp(mean * 252) - 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from stockml.config import TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class RocCurve:
    """False/true positive rates and the area under the curve."""

    fpr: list[float]
    tpr: list[float]
    auc: float


@dataclass(frozen=True)
class ModelEvaluation:
    """Test-set predictions and metrics for one fitted pipeline."""

    predictions: pd.Series
    scores: pd.Series
    metrics: dict[str, float]
    confusion: list[list[int]]
    roc: RocCurve


def classification_metrics(
    y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series | None = None
) -> dict[str, float]:
    """Accuracy, precision, recall, F1 and (if scores are given) ROC AUC for a binary target.

    ROC AUC is NaN when ``y_score`` is missing or ``y_true`` has a single class.
    """
    auc = float("nan")
    if y_score is not None and pd.Series(y_true).nunique() == 2:
        auc = float(roc_auc_score(y_true, y_score))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
    }


def decision_scores(pipeline: Pipeline, X: pd.DataFrame) -> pd.Series:
    """Continuous scores for ROC analysis: P(up) if available, else the decision function."""
    if hasattr(pipeline, "predict_proba"):
        try:
            values = pipeline.predict_proba(X)[:, 1]
        except AttributeError:
            values = pipeline.decision_function(X)
    else:
        values = pipeline.decision_function(X)
    return pd.Series(np.asarray(values, dtype=float), index=X.index, name="score")


def roc_points(y_true: pd.Series, y_score: pd.Series) -> RocCurve:
    """ROC curve points and AUC (requires both classes in ``y_true``)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return RocCurve(
        fpr=[float(v) for v in fpr],
        tpr=[float(v) for v in tpr],
        auc=float(roc_auc_score(y_true, y_score)),
    )


def evaluate_predictions(
    y_true: pd.Series, predictions: pd.Series, scores: pd.Series
) -> ModelEvaluation:
    """Metrics, confusion matrix and ROC curve for out-of-sample predictions.

    Args:
        y_true: Actual classes (0/1).
        predictions: Predicted classes, aligned with ``y_true``.
        scores: Continuous scores for ranking (P(up) or a decision function).
    """
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    return ModelEvaluation(
        predictions=predictions,
        scores=scores,
        metrics=classification_metrics(y_true, predictions, scores),
        confusion=[[int(v) for v in row] for row in cm],
        roc=roc_points(y_true, scores),
    )


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> ModelEvaluation:
    """Evaluate one *already fitted* pipeline on the test set.

    The same pipeline object produces the predictions, the metrics, the confusion matrix and
    the ROC curve, so every artefact describes one model.
    """
    preds = pd.Series(pipeline.predict(X_test).astype(int), index=X_test.index, name="prediction")
    return evaluate_predictions(y_test, preds, decision_scores(pipeline, X_test))


def auc_confidence_interval(
    auc: float, n_pos: int, n_neg: int, z: float = 1.96
) -> tuple[float, float]:
    """Normal-approximation confidence interval for ROC AUC (Hanley & McNeil, 1982).

    The standard error depends only on the AUC and the class counts, so it can be computed from
    stored results. It assumes independent observations; daily returns are close to that, but
    volatility clustering means the true interval is somewhat wider.

    Args:
        auc: Observed area under the ROC curve.
        n_pos: Number of positive (up) days.
        n_neg: Number of negative (down) days.
        z: Normal quantile; 1.96 gives a 95% interval.

    Returns:
        ``(low, high)`` clipped to ``[0, 1]``; ``(nan, nan)`` if either class is empty.
    """
    if n_pos < 1 or n_neg < 1 or not np.isfinite(auc):
        return float("nan"), float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc**2 / (1.0 + auc)
    variance = (auc * (1.0 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (
        n_pos * n_neg
    )
    se = float(np.sqrt(max(variance, 0.0)))
    return max(0.0, auc - z * se), min(1.0, auc + z * se)


def always_up_metrics(y_true: pd.Series) -> dict[str, float]:
    """Metrics for the naive baseline that predicts "up" every day (ROC AUC is 0.5)."""
    ones = pd.Series(1, index=y_true.index)
    metrics = classification_metrics(y_true, ones)
    metrics["roc_auc"] = 0.5
    return metrics


# --- risk / return ---------------------------------------------------------------------------


def equity_curve(log_returns: pd.Series) -> pd.Series:
    """Growth of 1 unit: ``exp(cumsum(log_returns))``."""
    return pd.Series(np.exp(log_returns.fillna(0.0).cumsum()), index=log_returns.index)


def drawdown(equity: pd.Series) -> pd.Series:
    """Drawdown from the running peak, as a non-positive fraction (``-0.2`` = 20% below peak)."""
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough fall of an equity curve, as a positive fraction."""
    if equity.empty:
        return 0.0
    return float(-drawdown(equity).min())


def annualised_return(
    log_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Compound annual growth rate implied by mean daily log return."""
    if log_returns.empty:
        return 0.0
    return float(np.expm1(log_returns.mean() * periods_per_year))


def annualised_volatility(
    log_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Standard deviation of daily returns scaled by ``sqrt(periods_per_year)``."""
    if len(log_returns) < 2:
        return 0.0
    return float(log_returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(
    log_returns: pd.Series,
    risk_free_daily: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio: ``mean(excess) / std(excess) * sqrt(periods_per_year)``.

    Returns 0.0 when volatility is zero.
    """
    excess = log_returns - risk_free_daily
    sd = excess.std(ddof=1)
    if len(excess) < 2 or not np.isfinite(sd) or sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino(
    log_returns: pd.Series,
    target_daily: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sortino ratio using downside deviation ``sqrt(mean(min(r - target, 0)^2))``.

    Returns 0.0 when there are no returns below the target.
    """
    excess = log_returns - target_daily
    downside = np.sqrt((excess.clip(upper=0.0) ** 2).mean())
    if excess.empty or not np.isfinite(downside) or downside == 0:
        return 0.0
    return float(excess.mean() / downside * np.sqrt(periods_per_year))


def calmar(log_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised return divided by max drawdown of the equity curve (0.0 if no drawdown)."""
    mdd = max_drawdown(equity_curve(log_returns))
    if mdd == 0:
        return 0.0
    return annualised_return(log_returns, periods_per_year) / mdd


def rolling_sharpe(
    log_returns: pd.Series, window: int, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> pd.Series:
    """Trailing-window annualised Sharpe ratio (NaN until the window fills)."""
    roll = log_returns.rolling(window, min_periods=window)
    result: pd.Series = roll.mean() / roll.std(ddof=1) * np.sqrt(periods_per_year)
    return result


def risk_metrics(
    log_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> dict[str, float]:
    """Summary performance statistics for a series of daily log returns."""
    return {
        "total_return": float(np.expm1(log_returns.sum())),
        "annual_return": annualised_return(log_returns, periods_per_year),
        "annual_volatility": annualised_volatility(log_returns, periods_per_year),
        "sharpe": sharpe(log_returns, periods_per_year=periods_per_year),
        "sortino": sortino(log_returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity_curve(log_returns)),
        "calmar": calmar(log_returns, periods_per_year),
        "hit_rate": float((log_returns > 0).mean()) if len(log_returns) else 0.0,
    }
