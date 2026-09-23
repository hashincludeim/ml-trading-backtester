"""The only module in the web app that calls into ``stockml``.

Views pass validated primitives in and get template-ready dictionaries back (numbers, strings,
and Plotly figures serialised to JSON). Expensive results are cached with Django's cache
framework, keyed by ticker, the latest training run, and a hash of the relevant config.

Two functions (``fetch_prices`` and ``train_ticker``) do network/CPU-heavy work and are only
called from management commands, never from a request.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from django.conf import settings
from django.core.cache import cache
from django.db import transaction

from dashboard.models import ModelResult, Ticker, TrainingRun
from stockml import analysis
from stockml.config import (
    AnalysisConfig,
    BacktestConfig,
    DataConfig,
    ExperimentConfig,
    config_hash,
    currency_symbol,
    price_unit,
    ticker_label,
)
from stockml.data.cleaning import clean_prices
from stockml.data.loader import cache_path, list_cached_tickers, load_prices
from stockml.evaluation.backtest import (
    BUY_AND_HOLD,
    STRATEGY,
    BacktestResult,
    cost_sensitivity,
    next_day_log_returns,
    run_backtest,
)
from stockml.evaluation.metrics import (
    RocCurve,
    annualised_return,
    annualised_volatility,
    max_drawdown,
    rolling_sharpe,
    sharpe,
)
from stockml.experiment import run_experiment
from stockml.features.pipeline import build_feature_frame, compute_indicators
from stockml.features.technical import log_returns, sma
from stockml.models.persistence import save_model
from stockml.models.registry import MODEL_REGISTRY, model_label
from stockml.viz import charts
from stockml.viz.theme import CATEGORICAL, model_color

logger = logging.getLogger(__name__)

T = TypeVar("T")

BENCHMARK = charts.BENCHMARK_LABEL
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "roc_auc": "ROC AUC",
    "sharpe": "Sharpe ratio",
}


class NoDataError(LookupError):
    """Raised when a ticker has no cached prices or no training run yet."""


# --- helpers ---------------------------------------------------------------------------------


def data_config() -> DataConfig:
    """Data settings rooted at ``settings.DATA_DIR``."""
    return DataConfig(data_dir=Path(settings.DATA_DIR))


_MIDNIGHT = re.compile(r'T00:00:00(?:\.0+)?"')


def figure_json(fig: go.Figure) -> str:
    """Serialise a figure for embedding inside ``<script type="application/json">``.

    Daily timestamps are shortened to plain dates (smaller pages), and ``</`` is escaped so the
    JSON can never close its script tag.
    """
    return _MIDNIGHT.sub('"', str(fig.to_json())).replace("</", "<\\/")


def _cached(key: str, builder: Callable[[], T]) -> T:
    value = cache.get(key)
    if value is None:
        value = builder()
        cache.set(key, value)
    return value


def _prices_version(ticker: str) -> str:
    path = cache_path(ticker, data_config().data_dir)
    return str(int(path.stat().st_mtime)) if path.exists() else "missing"


def _clip_dates(frame: pd.DataFrame, start: dt.date | None, end: dt.date | None) -> pd.DataFrame:
    return frame.loc[pd.Timestamp(start) if start else None : pd.Timestamp(end) if end else None]


# --- tickers & prices ------------------------------------------------------------------------


def display_name(ticker: str) -> str:
    """Human-readable name for a ticker (e.g. ``^GSPC`` -> ``S&P 500``)."""
    return ticker_label(ticker)


def ticker_choices(tickers: list[str]) -> list[tuple[str, str]]:
    """``(symbol, label)`` pairs for select boxes, e.g. ``("MSFT", "Microsoft (MSFT)")``."""
    return [(t, t if ticker_label(t) == t else f"{ticker_label(t)} ({t})") for t in tickers]


def _universe_order(symbols: list[str]) -> list[str]:
    """Sort symbols in the configured universe order (index first), others alphabetically after."""
    order = {s: i for i, s in enumerate(DataConfig().tickers)}
    return sorted(symbols, key=lambda s: (order.get(s, len(order)), s))


def available_tickers() -> list[str]:
    """Tickers with cached prices (database first, then the Parquet cache directory)."""
    symbols = list(Ticker.objects.filter(n_rows__gt=0).values_list("symbol", flat=True))
    return _universe_order(symbols or list_cached_tickers(data_config().data_dir))


def trained_tickers() -> list[str]:
    """Tickers with at least one training run."""
    return _universe_order(
        list(Ticker.objects.filter(runs__isnull=False).distinct().values_list("symbol", flat=True))
    )


def get_prices(ticker: str) -> pd.DataFrame:
    """Cleaned prices from the Parquet cache (never downloads).

    Raises:
        NoDataError: If the ticker has not been fetched.
    """

    def build() -> pd.DataFrame:
        try:
            return clean_prices(load_prices(ticker, data_config()), data_config())
        except FileNotFoundError as exc:
            raise NoDataError(
                f"No price data for {ticker}. Run: python web/manage.py fetch_prices {ticker}"
            ) from exc

    return _cached(f"prices:{ticker}:{_prices_version(ticker)}", build)


def date_bounds(ticker: str) -> tuple[dt.date, dt.date]:
    """First and last available dates for a ticker."""
    idx = get_prices(ticker).index
    return idx[0].date(), idx[-1].date()


def _format_price(value: float, unit: str) -> str:
    """Compact price for a stat tile: ``$245.10``, ``158.30p`` or ``7,706.03`` (index points)."""
    if unit == "USD":
        return f"${value:,.2f}"
    if unit == "GBX":
        return f"{value:,.2f}p"
    return f"{value:,.2f}"


def _price_stats(prices: pd.DataFrame, unit: str) -> list[dict[str, str]]:
    close = prices["Close"]
    returns = log_returns(close).dropna()
    total = float(close.iloc[-1] / close.iloc[0] - 1.0)
    cagr = annualised_return(returns)
    return [
        {
            "label": f"Close · {close.index[-1]:%d %b %Y}",
            "value": _format_price(close.iloc[-1], unit),
        },
        {"label": "Period return", "value": f"{total:+.1%}", "tone": _tone(total)},
        {"label": "Annualised return", "value": f"{cagr:+.1%}", "tone": _tone(cagr)},
        {"label": "Annualised volatility", "value": f"{annualised_volatility(returns):.1%}"},
        {"label": "Max drawdown", "value": f"{-max_drawdown(close):.1%}", "tone": "neg"},
        {"label": "Trading days", "value": f"{len(close):,}"},
    ]


def _count(n: int) -> str:
    return "none" if n == 0 else str(n)


def _tone(value: float) -> str:
    return "pos" if value > 0 else "neg" if value < 0 else ""


# --- pages -----------------------------------------------------------------------------------


def _fmt_month(ts: pd.Timestamp | None) -> str:
    return "not yet" if ts is None else f"{ts:%b %Y}"


def overview_context(ticker: str, start: dt.date | None, end: dt.date | None) -> dict[str, Any]:
    """Price history, drawdowns, calendar returns and volatility for a date range."""
    acfg = AnalysisConfig()

    name, unit = ticker_label(ticker), price_unit(ticker)

    def build() -> dict[str, Any]:
        full = get_prices(ticker)
        prices = _clip_dates(full, start, end)
        if len(prices) < 2:
            raise NoDataError("Not enough data in the selected date range.")
        close = prices["Close"]
        overlays = pd.DataFrame({f"SMA {w}": sma(full["Close"], w) for w in acfg.trend_sma_windows})
        rets = log_returns(close)
        annual = analysis.annual_returns(rets)
        months = analysis.calendar_returns(rets)
        vol = _clip_dates(
            analysis.rolling_annualised_volatility(
                full["Close"], acfg.volatility_window
            ).to_frame(),
            start,
            end,
        )["volatility"]
        episode = analysis.worst_drawdown(close)
        month_avg = months.mean()
        vol_peak = vol.idxmax() if vol.notna().any() else None
        return {
            "stats": _price_stats(prices, unit),
            "range": (prices.index[0].date(), prices.index[-1].date()),
            "charts": {
                "price": figure_json(
                    charts.price_volume_chart(prices, name, overlays, acfg.events, unit=unit)
                ),
                "underwater": figure_json(charts.underwater_chart(close, name, episode)),
                "annual": figure_json(
                    charts.annual_returns_chart(annual, f"{name} return by calendar year")
                ),
                "monthly": figure_json(
                    charts.monthly_returns_heatmap(months, f"{name} monthly returns")
                ),
                "volatility": figure_json(
                    charts.volatility_chart(vol, name, acfg.volatility_window, acfg.events)
                ),
            },
            "insights": {
                "underwater": (
                    f"Worst fall: {episode.depth:.0%} from {_fmt_month(episode.peak_date)} to "
                    f"{_fmt_month(episode.trough_date)}; back at the old high: "
                    f"{_fmt_month(episode.recovery_date)}."
                ),
                "annual": (
                    f"{(annual > 0).sum()} of {len(annual)} years were positive. Best "
                    f"{annual.idxmax()} ({annual.max():+.0%}), worst {annual.idxmin()} "
                    f"({annual.min():+.0%})."
                ),
                "monthly": (
                    f"Best average month: {month_avg.idxmax()} ({month_avg.max():+.1%}); worst: "
                    f"{month_avg.idxmin()} ({month_avg.min():+.1%}). With about "
                    f"{months.notna().sum().min()} years per month, these gaps are mostly noise."
                ),
                "volatility": (
                    f"Volatility peaked at {vol.max():.0%} in {_fmt_month(vol_peak)}, against a "
                    f"typical {vol.median():.0%}."
                    if vol_peak is not None
                    else "Not enough data for rolling volatility."
                ),
            },
        }

    return _cached(f"overview:{ticker}:{start}:{end}:{_prices_version(ticker)}", build)


def indicators_context(ticker: str, start: dt.date | None, end: dt.date | None) -> dict[str, Any]:
    """Indicator overlays computed on the full history, then clipped (so warm-up is correct)."""
    cfg = ExperimentConfig().features

    def build() -> dict[str, Any]:
        full = compute_indicators(get_prices(ticker), cfg)
        indicators = _clip_dates(full, start, end)
        if len(indicators) < 2:
            raise NoDataError("Not enough data in the selected date range.")
        n_bull = int(indicators["ema_cross_bullish"].sum())
        n_bear = int(indicators["ema_cross_bearish"].sum())
        last = indicators.iloc[-1]
        signals = analysis.indicator_signal_table(indicators, cfg)
        nxt = next_day_log_returns(indicators["Close"]).dropna()
        base = float((nxt > 0).mean())
        distinct = signals[(signals["up_rate"] - base).abs() > 1.96 * signals["se"]]
        extreme = signals.loc[(signals["up_rate"] - base).abs().idxmax()] if len(signals) else None
        return {
            "charts": {
                "indicators": figure_json(
                    charts.indicator_chart(
                        indicators, ticker_label(ticker), cfg, unit=price_unit(ticker)
                    )
                ),
                "signals": figure_json(charts.indicator_signal_chart(signals, base)),
            },
            "stats": [
                {"label": f"RSI ({cfg.rsi_window}) latest", "value": f"{last['rsi']:.1f}"},
                {
                    "label": "MACD histogram",
                    "value": f"{last['macd_hist']:+.2f}",
                    "tone": _tone(float(last["macd_hist"])),
                },
                {"label": "Bullish EMA crosses", "value": str(n_bull)},
                {"label": "Bearish EMA crosses", "value": str(n_bear)},
            ],
            "insights": {
                "signals": (
                    f"Over {len(nxt):,} days, {len(distinct)} of {len(signals)} signal states "
                    f"differ from the {base:.1%} average by more than their 95% range. The most "
                    f"extreme, {extreme['indicator']} {extreme['state'].lower()} "
                    f"({extreme['up_rate']:.0%} up), rests on just {int(extreme['n'])} days."
                    if extreme is not None
                    else "Not enough data to evaluate signals."
                ),
            },
            "config": cfg,
        }

    return _cached(
        f"indicators:{ticker}:{start}:{end}:{config_hash(cfg)}:{_prices_version(ticker)}", build
    )


def exploration_context(ticker: str) -> dict[str, Any]:
    """Feature distributions and predictive power, correlations, and return behaviour."""
    cfg = ExperimentConfig().features
    acfg = AnalysisConfig()

    def build() -> dict[str, Any]:
        prices = get_prices(ticker)
        X, y = build_feature_frame(prices, cfg)
        daily = X["return_1d"]
        forward = next_day_log_returns(prices["Close"])
        buckets = {f: analysis.feature_bucket_stats(X[f], y, acfg.signal_bins) for f in X.columns}
        corr = analysis.feature_return_correlation(X, forward)
        band = analysis.noise_band(len(X))
        acf = analysis.autocorrelation(daily, acfg.acf_max_lag)
        n_sig = int((corr.abs() > band).sum())
        n_acf = int((acf.abs() > band).sum())
        kurt = float(daily.kurt())
        tail = float((daily.abs() > 3 * daily.std()).mean())
        return {
            "charts": {
                "distribution": figure_json(charts.feature_distribution_chart(X, y)),
                "signal": figure_json(
                    charts.feature_signal_chart(buckets, float(y.mean()), acfg.signal_bins)
                ),
                "feature_corr": figure_json(charts.feature_correlation_chart(corr, band)),
                "correlation": figure_json(charts.correlation_heatmap(X)),
                "balance": figure_json(charts.target_balance_chart(y)),
                "acf": figure_json(charts.autocorrelation_chart(acf, band, ticker_label(ticker))),
                "returns": figure_json(
                    charts.returns_histogram(
                        pd.DataFrame({ticker_label(ticker): daily}),
                        f"{ticker_label(ticker)} daily returns vs a normal distribution",
                        fit_normal=True,
                    )
                ),
            },
            "insights": {
                "feature_corr": (
                    f"{n_sig} of {len(corr)} features clear the noise band; the strongest "
                    f"({corr.index[0]}) has ρ = {corr.iloc[0]:+.3f}, so it explains well under "
                    f"1% of the variation in next-day returns."
                ),
                "acf": (
                    f"{n_acf} of {len(acf)} lags fall outside the band, but the largest is only "
                    f"|ρ| = {acf.abs().max():.3f} (lag {int(acf.abs().idxmax())}), which explains "
                    f"{acf.abs().max() ** 2:.2%} of the variation. Volatility clustering also "
                    "makes the simple band too narrow, so even these are weaker than they look."
                    if n_acf
                    else f"No lag falls outside the noise band (lag-1 ρ = {acf.iloc[0]:+.3f}): "
                    "past returns tell you almost nothing about the next one."
                ),
                "returns": (
                    f"Excess kurtosis is {kurt:.1f} (a normal distribution has 0): moves bigger "
                    f"than 3σ happen on {tail:.1%} of days against 0.3% for a normal curve."
                ),
            },
            "n_rows": len(X),
            "n_features": X.shape[1],
            "up_share": float(y.mean()),
            "features": list(X.columns),
        }

    return _cached(f"explore:{ticker}:{config_hash(cfg)}:{_prices_version(ticker)}", build)


def latest_run(ticker: str) -> TrainingRun:
    """Most recent training run for a ticker.

    Raises:
        NoDataError: If the ticker has never been trained.
    """
    run = (
        TrainingRun.objects.filter(ticker__symbol=ticker)
        .prefetch_related("results")
        .order_by("-created_at")
        .first()
    )
    if run is None:
        raise NoDataError(
            f"No trained models for {ticker}. Run: python web/manage.py train_models "
            f"--ticker {ticker}"
        )
    return run


def naive_accuracy(run: TrainingRun) -> float:
    """Best constant guess on the test set: max(always-up, always-down) accuracy."""
    up = float(run.baseline.get("accuracy", 0.5))
    return max(up, 1.0 - up)


def _model_rows(run: TrainingRun) -> list[dict[str, Any]]:
    baseline_acc = naive_accuracy(run)
    rows = []
    for r in run.results.all():
        rows.append(
            {
                "name": r.model_name,
                "label": model_label(r.model_name),
                "color": model_color(r.model_name),
                "cv_mean": r.cv_accuracy_mean,
                "cv_std": r.cv_accuracy_std,
                "cv_auc": r.cv_roc_auc_mean,
                "accuracy": r.accuracy,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "roc_auc": r.roc_auc,
                "beats_baseline": r.accuracy > baseline_acc,
                "params": ", ".join(
                    f"{k.removeprefix('model__')}={v}" for k, v in r.best_params.items()
                ),
            }
        )
    return sorted(rows, key=lambda row: row["roc_auc"], reverse=True)


def models_context(ticker: str, selected: str | None) -> dict[str, Any]:
    """Model comparison table plus charts on accuracy, stability, separation and importance."""
    run = latest_run(ticker)
    acfg = AnalysisConfig()

    def build() -> dict[str, Any]:
        results = {r.model_name: r for r in run.results.all()}
        preds = _run_predictions(run)
        window = acfg.rolling_accuracy_window
        name = selected if selected in results else next(iter(results))
        chosen: ModelResult = results[name]
        rows = _model_rows(run)
        importance = pd.DataFrame(chosen.importance, index=["importance_mean", "importance_std"]).T
        rocs = {
            n: RocCurve(fpr=r.roc["fpr"], tpr=r.roc["tpr"], auc=r.roc["auc"])
            for n, r in results.items()
        }
        cv = {n: (r.cv_accuracy_mean, r.cv_accuracy_std) for n, r in results.items()}
        n_beat = sum(row["beats_baseline"] for row in rows)
        hits = pd.DataFrame(
            {
                n: analysis.rolling_hit_rate(preds["y_true"], preds[f"{n}_pred"], window)
                for n in results
            }
        ).dropna(how="all")
        above = (hits > 0.5).mean()
        folds = {n: r.cv_scores.get("accuracy", []) for n, r in results.items()}
        unstable = sum(1 for v in folds.values() if v and min(v) < 0.5 < max(v))
        return {
            "run": run,
            "rows": rows,
            "baseline": run.baseline,
            "selected": name,
            "selected_label": model_label(name),
            "model_choices": [(n, model_label(n)) for n in results],
            "n_beat": n_beat,
            "naive_accuracy": naive_accuracy(run),
            "naive_direction": "up" if run.baseline["accuracy"] >= 0.5 else "down",
            "n_models": len(rows),
            "best_auc": rows[0] if rows else None,
            "charts": {
                "cv": figure_json(charts.cv_scores_chart(cv, 0.5, baseline_label="Coin flip")),
                "roc": figure_json(charts.roc_curves_chart(rocs)),
                "confusion": figure_json(charts.confusion_matrix_chart(chosen.confusion, name)),
                "importance": figure_json(charts.feature_importance_chart(importance, name)),
                "rolling": figure_json(charts.rolling_accuracy_chart(hits, window)),
                "folds": figure_json(charts.cv_folds_chart(folds)),
                "scores": figure_json(
                    charts.score_distribution_chart(preds[f"{name}_score"], preds["y_true"], name)
                ),
            },
            "insights": {
                "rolling": (
                    f"Share of the test period each model spent above 50% (rolling {window} days): "
                    + ", ".join(f"{model_label(n)} {v:.0%}" for n, v in above.items())
                    + "."
                ),
                "folds": (
                    f"{unstable} of {len(folds)} models swing between beating and losing to a "
                    "coin flip across folds, so their average CV score hides a lot of instability."
                ),
                "scores": (
                    f"{model_label(name)} has a test ROC AUC of {chosen.roc_auc:.3f}: pick a "
                    f"random up day and a random down day, and it ranks the up day higher only "
                    f"{chosen.roc_auc:.0%} of the time (50% = guessing)."
                ),
            },
        }

    return _cached(f"models:{ticker}:{run.pk}:{selected}", build)


def _run_predictions(run: TrainingRun) -> pd.DataFrame:
    path = Path(run.predictions_path)
    if not path.exists():
        raise NoDataError(f"Predictions file missing for {run}. Re-run train_models.")
    return pd.read_parquet(path)


def _backtests(ticker: str, run: TrainingRun, config: BacktestConfig) -> dict[str, BacktestResult]:
    close = get_prices(ticker)["Close"]
    preds = _run_predictions(run)
    return {
        r.model_name: run_backtest(close, preds[f"{r.model_name}_pred"], config)
        for r in run.results.all()
    }


def _risk_row(name: str, m: dict[str, float], is_benchmark: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "label": model_label(name),
        "color": model_color(name),
        "benchmark": is_benchmark,
        **m,
    }


def backtest_context(ticker: str, cost_bps: float, mode: str, focus: str | None) -> dict[str, Any]:
    """Equity curves, drawdowns, rolling Sharpe, return distribution, and a risk table."""
    run = latest_run(ticker)
    config = BacktestConfig(mode=mode, cost_bps=cost_bps)  # type: ignore[arg-type]
    acfg = AnalysisConfig()

    def build() -> dict[str, Any]:
        results = _backtests(ticker, run, config)
        if not results:
            raise NoDataError("This training run has no model results.")
        bench = next(iter(results.values()))
        equity = pd.DataFrame({n: r.equity[STRATEGY] for n, r in results.items()})
        equity[BENCHMARK] = bench.equity[BUY_AND_HOLD]
        dd = pd.DataFrame({n: r.drawdown[STRATEGY] for n, r in results.items()})
        dd[BENCHMARK] = bench.drawdown[BUY_AND_HOLD]
        returns = pd.DataFrame({n: r.returns[STRATEGY] for n, r in results.items()})
        returns[BENCHMARK] = bench.returns[BUY_AND_HOLD]
        rolling = returns.apply(
            lambda s: rolling_sharpe(s, config.rolling_window, config.periods_per_year)
        )
        table = [_risk_row(n, r.metrics[STRATEGY]) for n, r in results.items()]
        table.sort(key=lambda row: row["sharpe"], reverse=True)
        bench_row = _risk_row(BENCHMARK, bench.metrics[BUY_AND_HOLD], is_benchmark=True)
        best = table[0]
        focus_name = focus if focus in results else best["name"]
        preds = _run_predictions(run)
        sensitivity = cost_sensitivity(
            get_prices(ticker)["Close"],
            {n: preds[f"{n}_pred"] for n in results},
            acfg.cost_grid_bps,
            config,
        )
        beating = (
            sensitivity.drop(columns=BUY_AND_HOLD).gt(sensitivity[BUY_AND_HOLD], axis=0)
        ).sum(axis=1)
        points = pd.DataFrame(
            [
                {
                    "label": row["label"],
                    "annual_volatility": row["annual_volatility"],
                    "annual_return": row["annual_return"],
                    "sharpe": row["sharpe"],
                    "color": row["color"],
                    "benchmark": row["benchmark"],
                }
                for row in [*table, bench_row]
            ]
        ).set_index("label")
        focus_months = analysis.calendar_returns(returns[focus_name])
        vol_spread = float(points["annual_volatility"].max() - points["annual_volatility"].min())
        focus_trades = results[focus_name].metrics[STRATEGY]["n_trades"]
        return {
            "run": run,
            "table": [*table, bench_row],
            "benchmark": bench_row,
            "best": best,
            "n_beat": sum(row["sharpe"] > bench_row["sharpe"] for row in table),
            "n_models": len(table),
            "focus": focus_name,
            "model_choices": [(n, model_label(n)) for n in results],
            "charts": {
                "equity": figure_json(
                    charts.equity_curves_chart(equity, cost_bps, currency_symbol(ticker))
                ),
                "drawdown": figure_json(charts.drawdown_chart(dd)),
                "rolling": figure_json(charts.rolling_sharpe_chart(rolling, config.rolling_window)),
                "returns": figure_json(
                    charts.returns_histogram(
                        returns[[focus_name, BENCHMARK]],
                        f"Daily returns: {model_label(focus_name)} vs buy & hold",
                    )
                ),
                "costs": figure_json(charts.cost_sensitivity_chart(sensitivity, cost_bps)),
                "risk_return": figure_json(
                    charts.risk_return_scatter(
                        points,
                        "Risk vs return on the test period",
                        "Up and to the left is better: more return for less volatility.",
                    )
                ),
                "monthly": figure_json(
                    charts.monthly_returns_heatmap(
                        focus_months, f"Monthly returns: {model_label(focus_name)}"
                    )
                ),
            },
            "insights": {
                "risk_return": (
                    "In long/short mode every strategy is always fully invested (long or short), "
                    "so they all carry the stock's volatility. Only the return differs."
                    if vol_spread < 0.005
                    else "Long/flat strategies sit in cash on predicted down days, so they take "
                    "less risk than holding the stock. Compare returns per unit of volatility."
                ),
                "costs": (
                    f"With free trading, {_count(int(beating.iloc[0]))} of {len(results)} "
                    f"strategies beat buy & hold on Sharpe; at {sensitivity.index[-1]:g} bp, "
                    f"{_count(int(beating.iloc[-1]))} do. {model_label(focus_name)} changed "
                    f"position {focus_trades:,.0f} times in {len(returns):,} trading days."
                ),
            },
        }

    return _cached(f"backtest:{ticker}:{run.pk}:{config_hash(config)}:{focus}", build)


def multi_ticker_context(metric: str) -> dict[str, Any]:
    """Compare every trained ticker: metric heatmap, best-model vs benchmark, relative prices."""
    tickers = trained_tickers()
    if not tickers:
        raise NoDataError("No tickers trained yet. Run: python web/manage.py train_models --all")
    runs = {t: latest_run(t) for t in tickers}
    key = "multi:" + metric + ":" + ",".join(f"{t}={r.pk}" for t, r in runs.items())

    def build() -> dict[str, Any]:
        config = BacktestConfig()
        matrix: dict[str, dict[str, float]] = {}
        best_rows = []
        summary = []
        for ticker, run in runs.items():
            bts = _backtests(ticker, run, config)
            if metric == "sharpe":
                matrix[ticker] = {n: bt.metrics[STRATEGY]["sharpe"] for n, bt in bts.items()}
            else:
                matrix[ticker] = {r.model_name: getattr(r, metric) for r in run.results.all()}
            # Select on training-period CV only, so the test result is out-of-sample.
            chosen = max(run.results.all(), key=lambda r: r.cv_roc_auc_mean)
            best_name = chosen.model_name
            best_bt = bts[best_name]
            best_rows.append(
                {
                    "ticker": ticker_label(ticker),
                    "model": best_name,
                    "strategy": best_bt.metrics[STRATEGY]["sharpe"],
                    "buy_and_hold": best_bt.metrics[BUY_AND_HOLD]["sharpe"],
                }
            )
            summary.append(
                {
                    "ticker": ticker,
                    "name": ticker_label(ticker),
                    "test_period": f"{run.test_start:%b %Y} – {run.test_end:%b %Y}",
                    "n_test": run.n_test,
                    "baseline_acc": naive_accuracy(run),
                    "best_acc": chosen.accuracy,
                    "best_model": model_label(best_name),
                    "best_sharpe": best_bt.metrics[STRATEGY]["sharpe"],
                    "bh_sharpe": best_bt.metrics[BUY_AND_HOLD]["sharpe"],
                }
            )
        mat = pd.DataFrame(matrix).T
        mat = mat[[c for c in MODEL_REGISTRY if c in mat.columns]]
        mat.index = [ticker_label(t) for t in mat.index]
        center = 0.0 if metric == "sharpe" else 0.5
        closes = pd.DataFrame({ticker_label(t): get_prices(t)["Close"] for t in tickers})
        daily = closes.apply(log_returns)
        ticker_points = pd.DataFrame(
            {
                t: {
                    "annual_volatility": annualised_volatility(daily[t].dropna()),
                    "annual_return": annualised_return(daily[t].dropna()),
                    "sharpe": sharpe(daily[t].dropna()),
                    "color": CATEGORICAL[i % len(CATEGORICAL)],
                }
                for i, t in enumerate(closes.columns)
            }
        ).T
        ret_corr = daily.corr()
        off_diag = ret_corr.where(~np.eye(len(ret_corr), dtype=bool)).stack()
        best_frame = pd.DataFrame(best_rows).set_index("ticker")
        return {
            "summary": summary,
            "metric": metric,
            "metric_label": METRIC_LABELS[metric],
            "total_rows": sum(len(get_prices(t)) for t in tickers),
            "charts": {
                "heatmap": figure_json(charts.metric_heatmap(mat, METRIC_LABELS[metric], center)),
                "best": figure_json(charts.strategy_vs_benchmark_chart(best_frame, "Sharpe ratio")),
                "prices": figure_json(
                    charts.normalised_prices_chart(closes, AnalysisConfig().events)
                ),
                "ticker_corr": figure_json(
                    charts.correlation_heatmap(
                        daily,
                        "Daily return correlation between tickers",
                        "High values mean the stocks mostly move together (one shared risk).",
                    )
                ),
                "ticker_risk": figure_json(
                    charts.risk_return_scatter(
                        ticker_points,
                        "Buy & hold risk vs return, full history",
                        "Each dot is a stock held for the whole period.",
                    )
                ),
            },
            "insights": {
                "ticker_corr": (
                    f"Pairwise correlations range from {off_diag.min():.2f} to "
                    f"{off_diag.max():.2f}: big tech moves largely with the market, so these "
                    "names diversify each other less than it might seem."
                    if len(off_diag)
                    else ""
                ),
            },
        }

    return _cached(key, build)


# --- offline jobs (management commands only) --------------------------------------------------


def fetch_prices(
    tickers: list[str], start: str | None = None, end: str | None = None, refresh: bool = False
) -> list[Ticker]:
    """Download (or refresh) prices into the Parquet cache and record ticker metadata."""
    base = data_config()
    cfg = DataConfig(
        tickers=tuple(tickers),
        start=start or base.start,
        end=end or base.end,
        data_dir=base.data_dir,
    )
    out = []
    for symbol in tickers:
        prices = load_prices(symbol, cfg, allow_download=True, refresh=refresh)
        ticker, _ = Ticker.objects.update_or_create(
            symbol=symbol,
            defaults={
                "first_date": prices.index[0].date(),
                "last_date": prices.index[-1].date(),
                "n_rows": len(prices),
                "prices_path": str(cache_path(symbol, cfg.data_dir)),
            },
        )
        out.append(ticker)
    cache.clear()
    return out


def remove_tickers(symbols: list[str], delete_files: bool = False) -> list[str]:
    """Remove tickers (and, via cascade, their training runs) from the database.

    Args:
        symbols: Ticker symbols to remove.
        delete_files: Also delete their cached prices and saved run artefacts under DATA_DIR.

    Returns:
        The symbols that were found and removed.
    """
    data_dir = data_config().data_dir
    removed = []
    for symbol in symbols:
        deleted, _ = Ticker.objects.filter(symbol=symbol).delete()
        if delete_files:
            cache_path(symbol, data_dir).unlink(missing_ok=True)
            run_dir = data_dir / "runs" / symbol
            if run_dir.is_dir():
                shutil.rmtree(run_dir)
        if deleted:
            removed.append(symbol)
    cache.clear()
    return removed


def train_ticker(symbol: str, config: ExperimentConfig | None = None) -> TrainingRun:
    """Train every configured model for one ticker and persist models, predictions, metrics."""
    cfg = config or ExperimentConfig()
    prices = get_prices(symbol)
    ticker, _ = Ticker.objects.get_or_create(
        symbol=symbol,
        defaults={
            "first_date": prices.index[0].date(),
            "last_date": prices.index[-1].date(),
            "n_rows": len(prices),
            "prices_path": str(cache_path(symbol, data_config().data_dir)),
        },
    )
    result = run_experiment(prices, cfg)
    digest = config_hash(cfg)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")
    run_dir = data_config().data_dir / "runs" / symbol / f"{stamp}_{digest}"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.parquet"
    result.predictions_frame().to_parquet(predictions_path)
    split = result.split
    with transaction.atomic():
        run = TrainingRun.objects.create(
            ticker=ticker,
            config_hash=digest,
            config=_jsonable(asdict(cfg)),
            train_start=split.X_train.index[0].date(),
            train_end=split.X_train.index[-1].date(),
            test_start=split.X_test.index[0].date(),
            test_end=split.X_test.index[-1].date(),
            n_train=len(split.X_train),
            n_test=len(split.X_test),
            baseline=result.baseline,
            predictions_path=str(predictions_path),
        )
        for name, outcome in result.outcomes.items():
            model_path = save_model(outcome.trained.pipeline, run_dir / f"{name}.joblib")
            ev, cv = outcome.evaluation, outcome.trained.cv
            ModelResult.objects.create(
                run=run,
                model_name=name,
                best_params=_jsonable(outcome.trained.best_params),
                cv_scores=cv.scores,
                cv_accuracy_mean=cv.mean("accuracy"),
                cv_accuracy_std=cv.std("accuracy"),
                cv_roc_auc_mean=cv.mean("roc_auc"),
                confusion=ev.confusion,
                roc=asdict(ev.roc),
                importance={
                    feat: [float(row.importance_mean), float(row.importance_std)]
                    for feat, row in outcome.importance.iterrows()
                },
                model_path=str(model_path),
                **ev.metrics,
            )
    cache.clear()
    return run


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)
