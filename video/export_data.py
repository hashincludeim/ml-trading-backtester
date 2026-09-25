"""Export the real numbers the explainer video animates to ``video/build/data.js``.

Every figure in the video (prices, feature values, split dates, model scores, equity curves)
comes from the same cached prices, training runs and core-library functions as the dashboard.
Run it after ``fetch_prices`` and ``train_models``::

    python video/export_data.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "video" / "build"

# The dashboard's services need Django configured before they can be imported.
sys.path.insert(0, str(ROOT / "web"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.base")
import django  # noqa: E402

django.setup()

import pandas as pd  # noqa: E402

from dashboard import services  # noqa: E402
from dashboard.context_processors import SOURCE_URL  # noqa: E402
from stockml.config import AnalysisConfig, BacktestConfig, ExperimentConfig  # noqa: E402
from stockml.evaluation.backtest import BUY_AND_HOLD, STRATEGY, run_backtest  # noqa: E402
from stockml.features.pipeline import build_feature_frame  # noqa: E402
from stockml.models.registry import MODEL_REGISTRY, model_label  # noqa: E402
from stockml.models.training import split_timeline  # noqa: E402
from stockml.viz.theme import CATEGORICAL, model_color  # noqa: E402

FEATURE_WINDOW_DAYS = 130  # trading days shown in the "features" scene
EQUITY_STEP = 5  # plot every 5th day (weekly) of the equity curves: smoother, lighter lines
# (feature column, on-screen label, display format) for the feature card in the video.
FEATURE_CARD = (
    ("rsi_14", "RSI, 14 days", "num1"),
    ("macd_hist_pct", "MACD histogram", "pct3"),
    ("sma_30_ratio", "Close vs 30-day average", "pct1"),
    ("volatility_21", "Volatility, 21 days", "pct2day"),
    ("bb_position", "Bollinger position", "num2"),
    ("volume_ratio", "Volume vs 20-day average", "times"),
)


def year_fraction(index: pd.DatetimeIndex) -> list[float]:
    """Dates as decimal years (2021.42...), a compact x coordinate for the animation."""
    return [round(d.year + (d.dayofyear - 1) / 365.25, 4) for d in index]


def rounded(values: Any, digits: int = 5) -> list[float]:
    return [round(float(v), digits) for v in values]


def price_scene(tickers: list[str]) -> list[dict[str, Any]]:
    """Weekly closes for each ticker, rebased to 1 on its own first day."""
    out = []
    for i, ticker in enumerate(tickers):
        close = services.get_prices(ticker)["Close"]
        weekly = close.resample("W-FRI").last().dropna()
        rebased = weekly / close.iloc[0]
        out.append(
            {
                "symbol": ticker,
                "name": services.display_name(ticker),
                "color": CATEGORICAL[i % len(CATEGORICAL)],
                "x": year_fraction(pd.DatetimeIndex(rebased.index)),
                "y": rounded(rebased, 4),
                "multiple": round(float(close.iloc[-1] / close.iloc[0]), 1),
                "first_year": close.index[0].year,
            }
        )
    return out


def feature_scene(ticker: str) -> dict[str, Any]:
    """The last few months of one ticker's closes, feature values and next-day labels."""
    prices = services.get_prices(ticker)
    X, y = build_feature_frame(prices, ExperimentConfig().features)
    X, y = X.iloc[-FEATURE_WINDOW_DAYS:], y.iloc[-FEATURE_WINDOW_DAYS:]
    # Closes for the labelled days plus the next day, so the scene can reveal "tomorrow".
    closes = prices["Close"].loc[X.index[0] :].iloc[: len(X) + 1]
    return {
        "name": services.display_name(ticker),
        "dates": [f"{d.day} {d:%b %Y}" for d in closes.index],
        "close": rounded(closes, 2),
        "target": [int(v) for v in y],
        "features": [
            {"label": label, "format": fmt, "values": rounded(X[col], 6)}
            for col, label, fmt in FEATURE_CARD
        ],
        "n_features": X.shape[1],
    }


def model_scene(runs: dict[str, Any]) -> dict[str, Any]:
    """Split timeline of the first ticker, plus every model's test ROC AUC on every ticker."""
    first, run = next(iter(runs.items()))
    dates = services.get_prices(first).index
    index = dates[(dates >= pd.Timestamp(run.train_start)) & (dates <= pd.Timestamp(run.test_end))]
    cfg = run.config.get("model", {})
    timeline = split_timeline(
        pd.DatetimeIndex(index), run.n_train, int(cfg["cv_splits"]), int(cfg["retrain_every"])
    )
    blocks = [
        {
            "stage": row.stage,
            "role": row.role,
            "start": year_fraction(pd.DatetimeIndex([row.start]))[0],
            "end": year_fraction(pd.DatetimeIndex([row.end]))[0],
        }
        for row in timeline.itertuples()
        if row.stage != "Walk-forward"
    ]
    scores: dict[str, dict[str, float]] = {name: {} for name in MODEL_REGISTRY}  # registry order
    for ticker, ticker_run in runs.items():
        for result in ticker_run.results.all():
            scores.setdefault(result.model_name, {})[ticker] = result.roc_auc
    chosen = {t: max(run.results.all(), key=lambda r: r.cv_roc_auc_mean) for t, run in runs.items()}
    aucs = [v for per in scores.values() for v in per.values()]
    return {
        "split_name": services.display_name(first),
        "blocks": blocks,
        "train_share": round(run.n_train / (run.n_train + run.n_test), 3),
        "test_start": f"{run.test_start:%b %Y}",
        "test_end": f"{run.test_end:%b %Y}",
        "n_test": run.n_test,
        "cv_splits": int(cfg["cv_splits"]),
        "models": [
            {
                "label": model_label(name),
                "color": model_color(name),
                "aucs": [
                    {"ticker": services.display_name(t), "auc": round(v, 4)} for t, v in per.items()
                ],
            }
            for name, per in scores.items()
            if per
        ],
        "auc_min": round(min(aucs), 2),
        "auc_max": round(max(aucs), 2),
        "n_beat_naive": sum(
            chosen[t].accuracy > services.naive_accuracy(run) for t, run in runs.items()
        ),
        "n_tickers": len(runs),
    }


def backtest_scene(ticker: str, run: Any) -> dict[str, Any]:
    """Equity of the CV-selected model at each cost level, against buy and hold."""
    chosen = max(run.results.all(), key=lambda r: r.cv_roc_auc_mean)  # never by test score
    close = services.get_prices(ticker)["Close"]
    preds = pd.read_parquet(services.data_path(run.predictions_path))[f"{chosen.model_name}_pred"]
    costs = AnalysisConfig().cost_grid_bps
    results = {c: run_backtest(close, preds, BacktestConfig(cost_bps=c)) for c in costs}
    base = results[costs[0]]
    keep = list(range(0, len(base.equity), EQUITY_STEP))
    if keep[-1] != len(base.equity) - 1:
        keep.append(len(base.equity) - 1)  # always end on the final day

    def thin(series: pd.Series) -> list[float]:
        return rounded(series.iloc[keep], 4)

    return {
        "name": services.display_name(ticker),
        "model": model_label(chosen.model_name),
        "color": model_color(chosen.model_name),
        "x": year_fraction(pd.DatetimeIndex(base.equity.index[keep])),
        "buy_and_hold": thin(base.equity[BUY_AND_HOLD]),
        "bh_sharpe": round(base.metrics[BUY_AND_HOLD]["sharpe"], 2),
        "costs": [
            {
                "bps": c,
                "equity": thin(r.equity[STRATEGY]),
                "sharpe": round(r.metrics[STRATEGY]["sharpe"], 2),
            }
            for c, r in results.items()
        ],
        "default_cost": BacktestConfig().cost_bps,
        "start": f"{base.equity.index[0]:%b %Y}",
        "end": f"{base.equity.index[-1]:%b %Y}",
    }


def main() -> None:
    tickers = services.available_tickers()
    runs = {t: services.latest_run(t) for t in services.trained_tickers()}
    if not tickers or not runs:
        sys.exit("No data yet: run fetch_prices and train_models first.")
    prices = price_scene(tickers)
    first_trained = next(iter(runs))
    last = max(services.get_prices(t).index[-1] for t in tickers)
    first = min(services.get_prices(t).index[0] for t in tickers)
    data = {
        "as_of": f"{last.day} {last:%b %Y}",
        "n_bars": sum(len(services.get_prices(t)) for t in tickers),
        "n_markets": len(tickers),
        "first_year": first.year,
        "last_year": last.year,
        "n_models": len(ExperimentConfig().model.models),
        "source_url": SOURCE_URL.removeprefix("https://"),
        "events": [list(event) for event in AnalysisConfig().events],
        "prices": prices,
        "features": feature_scene(first_trained),
        "models": model_scene(runs),
        "backtest": backtest_scene(first_trained, runs[first_trained]),
    }
    BUILD.mkdir(parents=True, exist_ok=True)
    out = BUILD / "data.js"
    out.write_text("window.STOCKML_VIDEO = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"Wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
