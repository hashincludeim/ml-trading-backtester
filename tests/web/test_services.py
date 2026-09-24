from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from django.test import override_settings

from dashboard import services
from dashboard.models import ModelResult, Ticker, TrainingRun
from stockml.config import ExperimentConfig, ModelConfig
from stockml.data.loader import save_prices
from tests.conftest import make_prices

FAST = ExperimentConfig(
    model=ModelConfig(
        models=("logistic_regression", "linear_svc"), cv_splits=3, tune=False, importance_repeats=2
    )
)


@pytest.fixture
def data_dir(tmp_path: Path):  # type: ignore[no-untyped-def]
    with override_settings(DATA_DIR=tmp_path):
        save_prices(make_prices(500, seed=11), "TEST.L", tmp_path)
        yield tmp_path


def test_figure_json_escapes_script_close() -> None:
    import pandas as pd
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Scatter(x=pd.to_datetime(["2024-01-02"]), y=[1]),
        layout={"title": {"text": "</script>"}},
    )
    out = services.figure_json(fig)
    assert "</script>" not in out
    parsed = json.loads(out)
    assert parsed["layout"]["title"]["text"] == "</script>"
    assert parsed["data"][0]["x"] == ["2024-01-02"]  # midnight timestamps compacted


def test_available_tickers_from_cache(data_dir: Path, db: None) -> None:
    assert services.available_tickers() == ["TEST.L"]


def test_missing_prices_raise_no_data(data_dir: Path, db: None) -> None:
    with pytest.raises(services.NoDataError, match="fetch_prices"):
        services.get_prices("NOPE.L")


def test_overview_and_indicators(data_dir: Path, db: None) -> None:
    ctx = services.overview_context("TEST.L", None, None)
    assert set(ctx["charts"]) == {"price", "underwater", "annual", "monthly", "volatility"}
    assert "Worst fall" in ctx["insights"]["underwater"]
    assert ctx["stats"][0]["label"].startswith("Close · ")
    ind = services.indicators_context("TEST.L", None, None)
    assert json.loads(ind["charts"]["indicators"])["layout"]["title"]["text"]
    assert json.loads(ind["charts"]["signals"])["data"]


def test_exploration(data_dir: Path, db: None) -> None:
    ctx = services.exploration_context("TEST.L")
    assert set(ctx["charts"]) == {
        "distribution",
        "signal",
        "feature_corr",
        "correlation",
        "balance",
        "acf",
        "returns",
    }
    assert set(ctx["insights"]) == {"feature_corr", "acf", "returns"}
    assert 0 < ctx["up_share"] < 1


def test_models_page_requires_training(data_dir: Path, db: None) -> None:
    with pytest.raises(services.NoDataError, match="train_models"):
        services.models_context("TEST.L", None)


def test_train_then_models_backtest_multi(data_dir: Path, db: None) -> None:
    run = services.train_ticker("TEST.L", FAST)
    assert Ticker.objects.get(symbol="TEST.L").runs.count() == 1
    assert TrainingRun.objects.count() == 1
    assert ModelResult.objects.filter(run=run).count() == 2
    # Stored relative to DATA_DIR, so the data folder can be built elsewhere and moved.
    assert not Path(run.predictions_path).is_absolute()
    assert (data_dir / run.predictions_path).exists()
    assert all((data_dir / r.model_path).exists() for r in run.results.all())

    models = services.models_context("TEST.L", "linear_svc")
    assert models["selected"] == "linear_svc"
    stored = ModelResult.objects.get(run=run, model_name="linear_svc").walk_forward
    assert set(stored) == {"metrics", "confusion"}
    assert sum(map(sum, stored["confusion"])) == run.n_test
    wf = models["walk_forward"]
    assert wf["retrain_every"] == FAST.model.retrain_every
    assert wf["n_refits"] == -(-run.n_test // FAST.model.retrain_every)
    assert '"Walk-forward"' in wf["chart"]
    assert "Refitting" in wf["insight"]
    assert {"rolling", "folds", "scores"} <= set(models["charts"])
    assert {r["name"] for r in models["rows"]} == {"logistic_regression", "linear_svc"}

    bt = services.backtest_context("TEST.L", cost_bps=0, mode="long_flat", focus=None)
    labels = [row["label"] for row in bt["table"]]
    assert labels[-1] == "Buy & hold"
    assert set(bt["charts"]) == {
        "equity",
        "drawdown",
        "rolling",
        "returns",
        "costs",
        "risk_return",
        "monthly",
    }
    assert "costs" in bt["insights"]
    costly = services.backtest_context("TEST.L", cost_bps=50, mode="long_flat", focus=None)
    assert costly["benchmark"]["total_return"] < bt["benchmark"]["total_return"]

    multi = services.multi_ticker_context("sharpe")
    assert multi["summary"][0]["ticker"] == "TEST.L"
    assert set(multi["charts"]) == {"heatmap", "best", "prices", "ticker_corr", "ticker_risk"}


def test_data_path_resolves_relative_and_keeps_absolute(data_dir: Path) -> None:
    assert services.stored_path(data_dir / "runs" / "x.parquet") == str(Path("runs/x.parquet"))
    assert services.data_path("runs/x.parquet") == data_dir / "runs" / "x.parquet"
    assert services.data_path("/elsewhere/x.parquet") == Path("/elsewhere/x.parquet")


def test_remove_tickers_cascades_and_deletes_files(data_dir: Path, db: None) -> None:
    run = services.train_ticker("TEST.L", FAST)
    run_dir = services.data_path(run.predictions_path).parent
    assert services.remove_tickers(["TEST.L", "NOPE"], delete_files=True) == ["TEST.L"]
    assert not Ticker.objects.filter(symbol="TEST.L").exists()
    assert TrainingRun.objects.count() == 0
    assert not run_dir.exists()
    assert services.available_tickers() == []


def test_ticker_choices_and_universe_order(db: None) -> None:
    assert services.ticker_choices(["^GSPC", "XYZ"]) == [
        ("^GSPC", "S&P 500 (^GSPC)"),
        ("XYZ", "XYZ"),
    ]
    assert services._universe_order(["ZZZ", "MSFT", "^GSPC"]) == ["^GSPC", "MSFT", "ZZZ"]


def test_format_price_units() -> None:
    assert services._format_price(1234.5, "USD") == "$1,234.50"
    assert services._format_price(158.3, "GBX") == "158.30p"
    assert services._format_price(7706.03, "points") == "7,706.03"


def test_models_page_without_walk_forward(data_dir: Path, db: None) -> None:
    config = ExperimentConfig(model=replace(FAST.model, walk_forward=False))
    run = services.train_ticker("TEST.L", config)
    assert all(r.walk_forward == {} for r in run.results.all())
    assert services.models_context("TEST.L", None)["walk_forward"] is None
