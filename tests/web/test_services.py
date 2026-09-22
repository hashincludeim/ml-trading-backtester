from __future__ import annotations

import json
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
    import plotly.graph_objects as go

    out = services.figure_json(go.Figure(layout={"title": {"text": "</script>"}}))
    assert "</script>" not in out
    assert json.loads(out)["layout"]["title"]["text"] == "</script>"


def test_available_tickers_from_cache(data_dir: Path, db: None) -> None:
    assert services.available_tickers() == ["TEST.L"]


def test_missing_prices_raise_no_data(data_dir: Path, db: None) -> None:
    with pytest.raises(services.NoDataError, match="fetch_prices"):
        services.get_prices("NOPE.L")


def test_overview_and_indicators(data_dir: Path, db: None) -> None:
    ctx = services.overview_context("TEST.L", None, None)
    assert "price" in ctx["charts"]
    assert ctx["stats"][0]["label"] == "Last close"
    ind = services.indicators_context("TEST.L", None, None)
    assert json.loads(ind["charts"]["indicators"])["layout"]["title"]["text"]


def test_exploration(data_dir: Path, db: None) -> None:
    ctx = services.exploration_context("TEST.L")
    assert set(ctx["charts"]) == {"distribution", "correlation", "balance", "returns"}
    assert 0 < ctx["up_share"] < 1


def test_models_page_requires_training(data_dir: Path, db: None) -> None:
    with pytest.raises(services.NoDataError, match="train_models"):
        services.models_context("TEST.L", None)


def test_train_then_models_backtest_multi(data_dir: Path, db: None) -> None:
    run = services.train_ticker("TEST.L", FAST)
    assert Ticker.objects.get(symbol="TEST.L").runs.count() == 1
    assert TrainingRun.objects.count() == 1
    assert ModelResult.objects.filter(run=run).count() == 2
    assert Path(run.predictions_path).exists()
    assert all(Path(r.model_path).exists() for r in run.results.all())

    models = services.models_context("TEST.L", "linear_svc")
    assert models["selected"] == "linear_svc"
    assert {r["name"] for r in models["rows"]} == {"logistic_regression", "linear_svc"}

    bt = services.backtest_context("TEST.L", cost_bps=0, mode="long_flat", focus=None)
    labels = [row["label"] for row in bt["table"]]
    assert labels[-1] == "Buy & hold"
    assert set(bt["charts"]) == {"equity", "drawdown", "rolling", "returns"}
    costly = services.backtest_context("TEST.L", cost_bps=50, mode="long_flat", focus=None)
    assert costly["benchmark"]["total_return"] < bt["benchmark"]["total_return"]

    multi = services.multi_ticker_context("sharpe")
    assert multi["summary"][0]["ticker"] == "TEST.L"
    assert set(multi["charts"]) == {"heatmap", "best", "prices"}
