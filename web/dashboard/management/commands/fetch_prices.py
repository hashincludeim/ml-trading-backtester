from __future__ import annotations

from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from dashboard import services
from stockml.config import DataConfig


class Command(BaseCommand):
    help = (
        "Download daily prices from Yahoo Finance into the Parquet cache. By default every "
        "ticker is re-downloaded so the data runs up to today."
    )

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("tickers", nargs="*", help="Symbols (default: DataConfig.tickers)")
        parser.add_argument("--start", help="ISO start date (default: DataConfig.start)")
        parser.add_argument(
            "--end", help="ISO end date, exclusive (default: tomorrow, i.e. include today)"
        )
        parser.add_argument(
            "--use-cache",
            action="store_true",
            help="Skip the download for tickers that are already cached",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        tickers = options["tickers"] or list(DataConfig().tickers)
        for ticker in services.fetch_prices(
            tickers, options["start"], options["end"], refresh=not options["use_cache"]
        ):
            self.stdout.write(
                self.style.SUCCESS(
                    f"{services.display_name(ticker.symbol)} ({ticker.symbol}): "
                    f"{ticker.n_rows:,} rows ({ticker.first_date} to {ticker.last_date})"
                )
            )
