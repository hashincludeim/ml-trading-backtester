from __future__ import annotations

from dataclasses import replace
from typing import Any

from django.core.management.base import BaseCommand, CommandError, CommandParser

from dashboard import services
from stockml.config import TARGET_LABELS, ExperimentConfig, ModelConfig, TargetConfig
from stockml.models.registry import MODEL_REGISTRY


class Command(BaseCommand):
    help = (
        "Train every registered model for one or more tickers and store the results, for the "
        "direction target, the volatility target, or both (the default)."
    )

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--ticker", action="append", default=[], help="Repeatable")
        parser.add_argument("--all", action="store_true", help="Train every cached ticker")
        parser.add_argument("--models", nargs="+", choices=sorted(MODEL_REGISTRY))
        parser.add_argument("--no-tune", action="store_true", help="Skip grid search")
        parser.add_argument(
            "--target",
            choices=[*TARGET_LABELS, "all"],
            default="all",
            help="What to predict (default: all targets)",
        )
        parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs inside sklearn")
        parser.add_argument(
            "--no-walk-forward", action="store_true", help="Skip walk-forward evaluation"
        )
        parser.add_argument(
            "--retrain-every",
            type=int,
            default=ModelConfig.retrain_every,
            help="Walk-forward refit interval in trading days",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        tickers = services.available_tickers() if options["all"] else options["ticker"]
        if not tickers:
            raise CommandError("Pass --ticker SYMBOL (repeatable) or --all.")
        if options["retrain_every"] < 1:
            raise CommandError("--retrain-every must be at least 1.")
        model_cfg = ModelConfig(
            tune=not options["no_tune"],
            n_jobs=options["jobs"],
            walk_forward=not options["no_walk_forward"],
            retrain_every=options["retrain_every"],
        )
        if options["models"]:
            model_cfg = replace(model_cfg, models=tuple(options["models"]))
        kinds = list(TARGET_LABELS) if options["target"] == "all" else [options["target"]]
        jobs = [(t, kind) for t in tickers for kind in kinds]
        for ticker, kind in jobs:
            config = ExperimentConfig(model=model_cfg, target=TargetConfig(kind=kind))  # type: ignore[arg-type]
            self.stdout.write(
                f"Training {len(model_cfg.models)} models for {ticker} "
                f"({TARGET_LABELS[kind].title.lower()})…"
            )
            try:
                run = services.train_ticker(ticker, config)
            except services.NoDataError as exc:
                raise CommandError(str(exc)) from exc
            results = list(run.results.all())
            best = max(results, key=lambda r: r.roc_auc)
            summary = (
                f"{ticker} {kind}: run {run.pk}, test {run.test_start}–{run.test_end}, "
                f"best AUC {best.roc_auc:.3f} ({best.model_name}), "
                f"always-positive accuracy {run.baseline['accuracy']:.3f}"
            )
            if run.heuristic:
                summary += f", no-model rule AUC {run.heuristic['roc_auc']:.3f}"
            if model_cfg.walk_forward:
                wf_best = max(results, key=lambda r: r.walk_forward["metrics"]["roc_auc"])
                summary += (
                    f", best walk-forward AUC {wf_best.walk_forward['metrics']['roc_auc']:.3f} "
                    f"({wf_best.model_name})"
                )
            self.stdout.write(self.style.SUCCESS(summary))
