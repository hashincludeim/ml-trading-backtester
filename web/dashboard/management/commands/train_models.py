from __future__ import annotations

from dataclasses import replace
from typing import Any

from django.core.management.base import BaseCommand, CommandError, CommandParser

from dashboard import services
from stockml.config import ExperimentConfig, ModelConfig
from stockml.models.registry import MODEL_REGISTRY


class Command(BaseCommand):
    help = "Train every registered model for one or more tickers and store the results."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--ticker", action="append", default=[], help="Repeatable")
        parser.add_argument("--all", action="store_true", help="Train every cached ticker")
        parser.add_argument("--models", nargs="+", choices=sorted(MODEL_REGISTRY))
        parser.add_argument("--no-tune", action="store_true", help="Skip grid search")
        parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs inside sklearn")

    def handle(self, *args: Any, **options: Any) -> None:
        tickers = services.available_tickers() if options["all"] else options["ticker"]
        if not tickers:
            raise CommandError("Pass --ticker SYMBOL (repeatable) or --all.")
        model_cfg = ModelConfig(tune=not options["no_tune"], n_jobs=options["jobs"])
        if options["models"]:
            model_cfg = replace(model_cfg, models=tuple(options["models"]))
        config = ExperimentConfig(model=model_cfg)
        for ticker in tickers:
            self.stdout.write(f"Training {len(model_cfg.models)} models for {ticker}…")
            try:
                run = services.train_ticker(ticker, config)
            except services.NoDataError as exc:
                raise CommandError(str(exc)) from exc
            best = max(run.results.all(), key=lambda r: r.roc_auc)
            self.stdout.write(
                self.style.SUCCESS(
                    f"{ticker}: run {run.pk}, test {run.test_start}–{run.test_end}, "
                    f"best AUC {best.roc_auc:.3f} ({best.model_name}), "
                    f"always-up accuracy {run.baseline['accuracy']:.3f}"
                )
            )
