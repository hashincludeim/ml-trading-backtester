from __future__ import annotations

from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from dashboard import services


class Command(BaseCommand):
    help = "Remove tickers and their training runs from the dashboard."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("tickers", nargs="+", help="Symbols to remove")
        parser.add_argument(
            "--delete-files",
            action="store_true",
            help="Also delete cached prices and saved models for these tickers",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        removed = services.remove_tickers(options["tickers"], options["delete_files"])
        missing = sorted(set(options["tickers"]) - set(removed))
        if removed:
            self.stdout.write(self.style.SUCCESS(f"Removed: {', '.join(removed)}"))
        if missing:
            self.stdout.write(f"Not found: {', '.join(missing)}")
