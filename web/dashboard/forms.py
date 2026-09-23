"""Query-string forms. Every page is a GET form, so views are bookmarkable."""

from __future__ import annotations

from typing import Any

from django import forms

from stockml.models.registry import MODEL_REGISTRY

DATE_WIDGET = forms.DateInput(attrs={"type": "date"})
MODEL_CHOICES = [(name, spec.label) for name, spec in MODEL_REGISTRY.items()]


class TickerForm(forms.Form):
    ticker = forms.ChoiceField(choices=())

    def __init__(self, *args: Any, tickers: list[tuple[str, str]], **kwargs: Any) -> None:
        """``tickers`` are ``(symbol, label)`` pairs."""
        super().__init__(*args, **kwargs)
        self.fields["ticker"].choices = tickers  # type: ignore[attr-defined]


class DateRangeForm(TickerForm):
    start = forms.DateField(required=False, widget=DATE_WIDGET)
    end = forms.DateField(required=False, widget=DATE_WIDGET)

    def clean(self) -> dict[str, Any]:
        data = super().clean() or {}
        start, end = data.get("start"), data.get("end")
        if start and end and start >= end:
            raise forms.ValidationError("Start date must be before end date.")
        return data


class ModelSelectForm(TickerForm):
    model = forms.ChoiceField(choices=MODEL_CHOICES, required=False)


class BacktestForm(TickerForm):
    cost_bps = forms.FloatField(
        min_value=0,
        max_value=50,
        initial=5,
        required=False,
        label="Cost (bp per trade)",
        widget=forms.NumberInput(attrs={"type": "range", "min": 0, "max": 50, "step": 1}),
    )
    mode = forms.ChoiceField(
        choices=[("long_short", "Long / short"), ("long_flat", "Long / flat")],
        initial="long_short",
        required=False,
    )
    model = forms.ChoiceField(choices=MODEL_CHOICES, required=False, label="Returns histogram")


class MultiTickerForm(forms.Form):
    metric = forms.ChoiceField(
        choices=[("accuracy", "Accuracy"), ("roc_auc", "ROC AUC"), ("sharpe", "Sharpe ratio")],
        initial="accuracy",
        required=False,
    )
