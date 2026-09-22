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
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
import plotly.graph_objects as go
from django.conf import settings
from django.core.cache import cache
from django.db import transaction

from dashboard.models import ModelResult, Ticker, TrainingRun
from stockml.config import BacktestConfig, DataConfig, ExperimentConfig, config_hash
from stockml.data.cleaning import clean_prices
from stockml.data.loader import cache_path, list_cached_tickers, load_prices
from stockml.evaluation.backtest import BUY_AND_HOLD, STRATEGY, BacktestResult, run_backtest
from stockml.evaluation.metrics import (
    RocCurve,
    annualised_return,
    annualised_volatility,
    max_drawdown,
    rolling_sharpe,
)
from stockml.experiment import run_experiment
from stockml.features.pipeline import build_feature_frame, compute_indicators
from stockml.features.technical import log_returns
from stockml.models.persistence import save_model
from stockml.models.registry import MODEL_REGISTRY, model_label
from stockml.viz import charts
from stockml.viz.theme import model_color

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


def figure_json(fig: go.Figure) -> str:
    """Serialise a figure for embedding inside ``<script type="application/json">``."""
    return str(fig.to_json()).replace("</", "<\\/")


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


def available_tickers() -> list[str]:
    """Tickers with cached prices (database first, then the Parquet cache directory)."""
    symbols = list(Ticker.objects.filter(n_rows__gt=0).values_list("symbol", flat=True))
    return symbols or list_cached_tickers(data_config().data_dir)


def trained_tickers() -> list[str]:
    """Tickers with at least one training run."""
    return list(
        Ticker.objects.filter(runs__isnull=False).distinct().values_list("symbol", flat=True)
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


def _price_stats(prices: pd.DataFrame) -> list[dict[str, str]]:
    close = prices["Close"]
    returns = log_returns(close).dropna()
    total = float(close.iloc[-1] / close.iloc[0] - 1.0)
    cagr = annualised_return(returns)
    return [
        {"label": "Last close", "value": f"{close.iloc[-1]:,.2f} GBX"},
        {"label": "Period return", "value": f"{total:+.1%}", "tone": _tone(total)},
        {"label": "Annualised return", "value": f"{cagr:+.1%}", "tone": _tone(cagr)},
        {"label": "Annualised volatility", "value": f"{annualised_volatility(returns):.1%}"},
        {"label": "Max drawdown", "value": f"{-max_drawdown(close):.1%}", "tone": "neg"},
        {"label": "Trading days", "value": f"{len(close):,}"},
    ]


def _tone(value: float) -> str:
    return "pos" if value > 0 else "neg" if value < 0 else ""


# --- pages -----------------------------------------------------------------------------------


def overview_context(ticker: str, start: dt.date | None, end: dt.date | None) -> dict[str, Any]:
    """Candlestick + volume chart and headline statistics for a date range."""

    def build() -> dict[str, Any]:
        prices = _clip_dates(get_prices(ticker), start, end)
        if len(prices) < 2:
            raise NoDataError("Not enough data in the selected date range.")
        return {
            "stats": _price_stats(prices),
            "charts": {"price": figure_json(charts.price_volume_chart(prices, ticker))},
            "range": (prices.index[0].date(), prices.index[-1].date()),
        }

    return _cached(f"overview:{ticker}:{start}:{end}:{_prices_version(ticker)}", build)


def indicators_context(ticker: str, start: dt.date | None, end: dt.date | None) -> dict[str, Any]:
    """Indicator overlays computed on the full history, then clipped (so warm-up is correct)."""
    cfg = ExperimentConfig().features

    def build() -> dict[str, Any]:
        indicators = _clip_dates(compute_indicators(get_prices(ticker), cfg), start, end)
        if len(indicators) < 2:
            raise NoDataError("Not enough data in the selected date range.")
        n_bull = int(indicators["ema_cross_bullish"].sum())
        n_bear = int(indicators["ema_cross_bearish"].sum())
        last = indicators.iloc[-1]
        return {
            "charts": {"indicators": figure_json(charts.indicator_chart(indicators, ticker, cfg))},
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
            "config": cfg,
        }

    return _cached(
        f"indicators:{ticker}:{start}:{end}:{config_hash(cfg)}:{_prices_version(ticker)}", build
    )


def exploration_context(ticker: str) -> dict[str, Any]:
    """Feature distributions, correlations, target balance, and daily-return distribution."""
    cfg = ExperimentConfig().features

    def build() -> dict[str, Any]:
        prices = get_prices(ticker)
        X, y = build_feature_frame(prices, cfg)
        daily = pd.DataFrame({BENCHMARK: X["return_1d"]})
        return {
            "charts": {
                "distribution": figure_json(charts.feature_distribution_chart(X, y)),
                "correlation": figure_json(charts.correlation_heatmap(X)),
                "balance": figure_json(charts.target_balance_chart(y)),
                "returns": figure_json(
                    charts.returns_histogram(daily, f"{ticker} daily log returns")
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
    """Model comparison table, CV chart, ROC curves, confusion matrix, feature importance."""
    run = latest_run(ticker)

    def build() -> dict[str, Any]:
        results = {r.model_name: r for r in run.results.all()}
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
                "equity": figure_json(charts.equity_curves_chart(equity, cost_bps)),
                "drawdown": figure_json(charts.drawdown_chart(dd)),
                "rolling": figure_json(charts.rolling_sharpe_chart(rolling, config.rolling_window)),
                "returns": figure_json(
                    charts.returns_histogram(
                        returns[[focus_name, BENCHMARK]],
                        f"Daily returns: {model_label(focus_name)} vs buy & hold",
                    )
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
                    "ticker": ticker,
                    "model": best_name,
                    "strategy": best_bt.metrics[STRATEGY]["sharpe"],
                    "buy_and_hold": best_bt.metrics[BUY_AND_HOLD]["sharpe"],
                }
            )
            summary.append(
                {
                    "ticker": ticker,
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
        center = 0.0 if metric == "sharpe" else 0.5
        closes = pd.DataFrame({t: get_prices(t)["Close"] for t in tickers})
        best_frame = pd.DataFrame(best_rows).set_index("ticker")
        return {
            "summary": summary,
            "metric": metric,
            "metric_label": METRIC_LABELS[metric],
            "total_rows": sum(len(get_prices(t)) for t in tickers),
            "charts": {
                "heatmap": figure_json(charts.metric_heatmap(mat, METRIC_LABELS[metric], center)),
                "best": figure_json(charts.strategy_vs_benchmark_chart(best_frame, "Sharpe ratio")),
                "prices": figure_json(charts.normalised_prices_chart(closes)),
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
