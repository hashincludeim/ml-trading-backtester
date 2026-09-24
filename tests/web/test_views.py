from __future__ import annotations

import datetime as dt
from typing import Any
from unittest.mock import patch

import pytest
from django.test import Client
from django.urls import reverse

from dashboard import services

FAKE_CHART = '{"data": [], "layout": {"title": {"text": "x"}}}'


@pytest.fixture
def client() -> Client:
    return Client()


def _ctx(*keys: str, **extra: Any) -> dict[str, Any]:
    return {"charts": dict.fromkeys(keys, FAKE_CHART), **extra}


@pytest.fixture(autouse=True)
def tickers():  # type: ignore[no-untyped-def]
    with (
        patch.object(services, "available_tickers", return_value=["AMZN", "MSFT"]),
        patch.object(services, "trained_tickers", return_value=["AMZN"]),
    ):
        yield


def test_overview_calls_service_with_parsed_dates(client: Client) -> None:
    with patch.object(services, "overview_context", return_value=_ctx("price", stats=[])) as svc:
        resp = client.get(
            reverse("dashboard:overview"),
            {"ticker": "MSFT", "start": "2020-01-01", "end": "2021-01-01"},
        )
    assert resp.status_code == 200
    svc.assert_called_once_with("MSFT", dt.date(2020, 1, 1), dt.date(2021, 1, 1))
    assert b"chart-price" in resp.content


def test_overview_defaults_to_first_ticker(client: Client) -> None:
    with patch.object(services, "overview_context", return_value=_ctx("price", stats=[])) as svc:
        assert client.get(reverse("dashboard:overview")).status_code == 200
    svc.assert_called_once_with("AMZN", None, None)


def test_invalid_ticker_does_not_call_service(client: Client) -> None:
    with patch.object(services, "overview_context") as svc:
        resp = client.get(reverse("dashboard:overview"), {"ticker": "EVIL"})
    assert resp.status_code == 200
    svc.assert_not_called()


def test_bad_date_order_rejected(client: Client) -> None:
    with patch.object(services, "indicators_context") as svc:
        resp = client.get(
            reverse("dashboard:indicators"), {"start": "2021-01-01", "end": "2020-01-01"}
        )
    assert resp.status_code == 200
    svc.assert_not_called()


def test_indicators(client: Client) -> None:
    ctx = _ctx("indicators", stats=[], config=services.ExperimentConfig().features)
    with patch.object(services, "indicators_context", return_value=ctx) as svc:
        assert client.get(reverse("dashboard:indicators")).status_code == 200
    svc.assert_called_once_with("AMZN", None, None)


def test_exploration(client: Client) -> None:
    ctx = _ctx("distribution", "correlation", "balance", "returns", up_share=0.5, n_rows=10)
    with patch.object(services, "exploration_context", return_value=ctx) as svc:
        assert client.get(reverse("dashboard:exploration")).status_code == 200
    svc.assert_called_once_with("AMZN")


def test_models(client: Client) -> None:
    ctx = _ctx(
        "cv",
        "roc",
        "confusion",
        "importance",
        rows=[],
        baseline={"accuracy": 0.5},
        model_choices=[],
        n_beat=0,
        n_models=0,
    )
    with patch.object(services, "models_context", return_value=ctx) as svc:
        resp = client.get(reverse("dashboard:models"), {"model": "svm_rbf"})
    assert resp.status_code == 200
    svc.assert_called_once_with("AMZN", "svm_rbf")


def test_models_walk_forward_section(client: Client) -> None:
    section = {"chart": FAKE_CHART, "insight": "Refitting 3 times", "retrain_every": 63}
    ctx = _ctx("roc", rows=[], baseline={}, insights={}, walk_forward=section)
    with patch.object(services, "models_context", return_value=ctx):
        resp = client.get(reverse("dashboard:models"))
    assert b"chart-walk-forward" in resp.content
    assert b"Refitting 3 times" in resp.content
    ctx["walk_forward"] = None
    with patch.object(services, "models_context", return_value=ctx):
        resp = client.get(reverse("dashboard:models"))
    assert b"chart-walk-forward" not in resp.content
    assert b"trained before walk-forward evaluation" in resp.content


def test_backtest_parses_cost_and_mode(client: Client) -> None:
    ctx = _ctx(
        "equity",
        "drawdown",
        "rolling",
        "returns",
        table=[],
        benchmark={},
        best={},
        model_choices=[],
        n_beat=0,
        n_models=0,
    )
    with patch.object(services, "backtest_context", return_value=ctx) as svc:
        resp = client.get(reverse("dashboard:backtest"), {"cost_bps": "12", "mode": "long_flat"})
    assert resp.status_code == 200
    svc.assert_called_once_with("AMZN", cost_bps=12.0, mode="long_flat", focus=None)


def test_backtest_rejects_out_of_range_cost(client: Client) -> None:
    with patch.object(services, "backtest_context") as svc:
        client.get(reverse("dashboard:backtest"), {"cost_bps": "500"})
    svc.assert_not_called()


def test_multi_ticker(client: Client) -> None:
    ctx = _ctx("heatmap", "best", "prices", summary=[], metric="sharpe")
    with patch.object(services, "multi_ticker_context", return_value=ctx) as svc:
        assert (
            client.get(reverse("dashboard:multi_ticker"), {"metric": "sharpe"}).status_code == 200
        )
    svc.assert_called_once_with("sharpe")


def test_no_data_error_is_shown(client: Client) -> None:
    with patch.object(services, "models_context", side_effect=services.NoDataError("train it")):
        resp = client.get(reverse("dashboard:models"))
    assert resp.status_code == 200
    assert b"train it" in resp.content


def test_empty_state_when_nothing_fetched(client: Client) -> None:
    with patch.object(services, "available_tickers", return_value=[]):
        resp = client.get(reverse("dashboard:overview"))
    assert b"No price data yet" in resp.content


def test_pages_are_gzipped_for_clients_that_accept_it(client: Client) -> None:
    with patch.object(services, "overview_context", return_value=_ctx("price", stats=[])):
        resp = client.get(reverse("dashboard:overview"), HTTP_ACCEPT_ENCODING="gzip")
    assert resp.status_code == 200
    assert resp["Content-Encoding"] == "gzip"


def test_tables_mark_their_key_columns_for_phones(client: Client) -> None:
    ctx = _ctx("roc", rows=[], baseline={}, selected="linear_svc", insights={})
    with patch.object(services, "models_context", return_value=ctx):
        resp = client.get(reverse("dashboard:models"), {"ticker": "AMZN"})
    assert b'data-key-cols="1,4,8,9"' in resp.content
