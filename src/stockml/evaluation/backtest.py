"""Vectorised daily backtest of a direction-prediction strategy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockml.config import BacktestConfig
from stockml.evaluation.metrics import drawdown, equity_curve, risk_metrics

STRATEGY = "strategy"
BUY_AND_HOLD = "buy_and_hold"


@dataclass(frozen=True)
class BacktestResult:
    """Daily series and summary metrics for a strategy and its buy-and-hold benchmark.

    Attributes:
        returns: Net daily log returns, columns ``strategy`` and ``buy_and_hold``.
        positions: Strategy position held from close ``t`` to close ``t+1`` (-1, 0 or 1).
        equity: Growth of 1 unit for each column of ``returns``.
        drawdown: Drawdown of each equity curve (non-positive fractions).
        metrics: ``{column: risk_metrics}`` plus ``n_trades`` and ``cost_paid`` per column.
    """

    returns: pd.DataFrame
    positions: pd.Series
    equity: pd.DataFrame
    drawdown: pd.DataFrame
    metrics: dict[str, dict[str, float]]


def next_day_log_returns(close: pd.Series) -> pd.Series:
    """Log return earned by holding from close ``t`` to close ``t+1``, stamped at ``t``."""
    return pd.Series(np.log(close.shift(-1) / close), index=close.index, name="next_return")


def positions_from_predictions(predictions: pd.Series, mode: str) -> pd.Series:
    """Map 1/0 predictions to positions: long on 1; short (long_short) or flat (long_flat) on 0."""
    down_position = -1.0 if mode == "long_short" else 0.0
    return pd.Series(
        np.where(predictions.astype(int) == 1, 1.0, down_position),
        index=predictions.index,
        name="position",
    )


def _net_returns(position: pd.Series, next_ret: pd.Series, cost_rate: float) -> pd.Series:
    """Gross ``position * next_ret`` minus ``cost_rate * |position change|`` (entry from flat)."""
    turnover = position.diff().fillna(position).abs()
    return position * next_ret - turnover * cost_rate


def run_backtest(
    close: pd.Series, predictions: pd.Series, config: BacktestConfig | None = None
) -> BacktestResult:
    """Backtest daily direction predictions against buy-and-hold.

    Timing: the prediction made at the close of day ``t`` sets the position held until the close
    of ``t+1`` and therefore earns ``ln(C_{t+1} / C_t)``. A cost of ``cost_bps / 10_000`` is
    deducted (in log-return space) per unit of position change, including the initial entry.
    Rows whose next close is unknown are dropped.

    Args:
        close: Full close-price history (must include the day after the last prediction).
        predictions: 1/0 predictions indexed by date (a subset of ``close.index``).
        config: Strategy mode and costs.

    Returns:
        A :class:`BacktestResult`.
    """
    cfg = config or BacktestConfig()
    next_ret = next_day_log_returns(close).reindex(predictions.index).dropna()
    preds = predictions.reindex(next_ret.index)
    cost_rate = cfg.cost_bps / 10_000.0
    position = positions_from_predictions(preds, cfg.mode)
    hold = pd.Series(1.0, index=next_ret.index)
    returns = pd.DataFrame(
        {
            STRATEGY: _net_returns(position, next_ret, cost_rate),
            BUY_AND_HOLD: _net_returns(hold, next_ret, cost_rate),
        }
    )
    equity = returns.apply(equity_curve)
    metrics: dict[str, dict[str, float]] = {}
    for col, pos in ((STRATEGY, position), (BUY_AND_HOLD, hold)):
        turnover = pos.diff().fillna(pos).abs()
        metrics[col] = {
            **risk_metrics(returns[col], cfg.periods_per_year),
            "n_trades": float((turnover > 0).sum()),
            "cost_paid": float(turnover.sum() * cost_rate),
        }
    return BacktestResult(
        returns=returns,
        positions=position,
        equity=equity,
        drawdown=equity.apply(drawdown),
        metrics=metrics,
    )
