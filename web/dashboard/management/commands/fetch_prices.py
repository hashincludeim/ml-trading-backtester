from __future__ import annotations

from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from dashboard import services
from stockml.config import DataConfig


class Command(BaseCommand):
    help = "Download daily prices from Yahoo Finance into the Parquet cache."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("tickers", nargs="*", help="Symbols (default: DataConfig.tickers)")
        parser.add_argument("--start", help="ISO start date (default: DataConfig.start)")
        parser.add_argument("--end", help="ISO end date, exclusive (default: DataConfig.end)")
        parser.add_argument("--refresh", action="store_true", help="Re-download cached tickers")

    def handle(self, *args: Any, **options: Any) -> None:
        tickers = options["tickers"] or list(DataConfig().tickers)
        for ticker in services.fetch_prices(
            tickers, options["start"], options["end"], options["refresh"]
        ):
            self.stdout.write(
                self.style.SUCCESS(
                    f"{ticker.symbol}: {ticker.n_rows:,} rows "
                    f"({ticker.first_date} to {ticker.last_date})"
                )
            )
