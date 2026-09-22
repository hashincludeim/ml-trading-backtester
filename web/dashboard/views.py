"""Thin views: validate the query string, call one service, render."""

from __future__ import annotations

from typing import Any, ClassVar

from django import forms as django_forms
from django.views.generic import TemplateView

from dashboard import services
from dashboard.forms import (
    BacktestForm,
    DateRangeForm,
    ModelSelectForm,
    MultiTickerForm,
    TickerForm,
)


class TickerPageView(TemplateView):
    """Base view: binds a ticker form from GET, then asks :meth:`get_page` for content."""

    form_class: ClassVar[type[TickerForm]] = TickerForm
    page: ClassVar[str] = ""
    requires_training: ClassVar[bool] = False

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        tickers = (
            services.trained_tickers() if self.requires_training else services.available_tickers()
        )
        context.update(page=self.page, tickers=tickers)
        if not tickers:
            context["empty"] = True
            return context
        data = self.request.GET.copy()
        data.setdefault("ticker", tickers[0])
        form = self.form_class(data, tickers=tickers)
        context["form"] = form
        if not form.is_valid():
            return context
        try:
            context.update(self.get_page(form.cleaned_data))
        except services.NoDataError as exc:
            context["error"] = str(exc)
        return context

    def get_page(self, data: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class OverviewView(TickerPageView):
    template_name = "dashboard/overview.html"
    form_class = DateRangeForm
    page = "overview"

    def get_page(self, data: dict[str, Any]) -> dict[str, Any]:
        return services.overview_context(data["ticker"], data.get("start"), data.get("end"))


class IndicatorsView(TickerPageView):
    template_name = "dashboard/indicators.html"
    form_class = DateRangeForm
    page = "indicators"

    def get_page(self, data: dict[str, Any]) -> dict[str, Any]:
        return services.indicators_context(data["ticker"], data.get("start"), data.get("end"))


class ExplorationView(TickerPageView):
    template_name = "dashboard/exploration.html"
    page = "exploration"

    def get_page(self, data: dict[str, Any]) -> dict[str, Any]:
        return services.exploration_context(data["ticker"])


class ModelsView(TickerPageView):
    template_name = "dashboard/models.html"
    form_class = ModelSelectForm
    page = "models"
    requires_training = True

    def get_page(self, data: dict[str, Any]) -> dict[str, Any]:
        return services.models_context(data["ticker"], data.get("model") or None)


class BacktestView(TickerPageView):
    template_name = "dashboard/backtest.html"
    form_class = BacktestForm
    page = "backtest"
    requires_training = True

    def get_page(self, data: dict[str, Any]) -> dict[str, Any]:
        cost = data.get("cost_bps")
        return services.backtest_context(
            data["ticker"],
            cost_bps=5.0 if cost is None else float(cost),
            mode=data.get("mode") or "long_short",
            focus=data.get("model") or None,
        )


class MultiTickerView(TemplateView):
    template_name = "dashboard/multi_ticker.html"

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        form: django_forms.Form = MultiTickerForm(self.request.GET or None)
        metric = form.cleaned_data.get("metric") if form.is_valid() else None
        context.update(page="multi", form=form)
        try:
            context.update(services.multi_ticker_context(metric or "accuracy"))
        except services.NoDataError as exc:
            context["error"] = str(exc)
        return context
