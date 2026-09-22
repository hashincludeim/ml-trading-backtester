"""Configuration dataclasses.

Every tunable number, date, and ticker used by the library lives here so that logic modules
contain no magic values. All configs are frozen so they can be hashed and used as cache keys.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

TRADING_DAYS_PER_YEAR = 252
PRICE_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True)
class DataConfig:
    """Where prices come from and how they are cleaned.

    Attributes:
        tickers: Default universe fetched by the ``fetch_prices`` command.
        start: First date requested from the data source (inclusive, ISO format).
        end: Last date requested from the data source (exclusive, ISO format).
        data_dir: Root directory for the Parquet price cache.
        auto_adjust: Request split/dividend adjusted OHLC from yfinance.
        outlier_method: ``"zscore"`` scales each daily log return by the rolling volatility of
            the preceding window; ``"iqr"`` applies the full-sample IQR rule to log returns.
        outlier_iqr_k: IQR multiplier for the ``"iqr"`` method.
        outlier_zscore_window: Look-back (trading days) for the rolling volatility.
        outlier_zscore_threshold: Absolute z-score above which a return is extreme.
        outlier_require_reversal: Only flag a return if the next return reverses it, which
            separates one-day data errors from genuine crashes.
    """

    tickers: tuple[str, ...] = ("BARC.L", "HSBA.L", "LLOY.L", "NWG.L", "STAN.L")
    start: str = "2000-01-01"
    end: str = "2022-12-31"
    data_dir: Path = Path("data")
    auto_adjust: bool = True
    outlier_method: Literal["zscore", "iqr"] = "zscore"
    outlier_iqr_k: float = 6.0
    outlier_zscore_window: int = 63
    outlier_zscore_threshold: float = 6.0
    outlier_require_reversal: bool = True


@dataclass(frozen=True)
class FeatureConfig:
    """Technical indicator windows and the resulting model feature list.

    Price-level indicators (SMA/EMA values) are converted to ratios against the close so the
    features are comparable across time and across tickers.
    """

    sma_windows: tuple[int, ...] = (3, 10, 30)
    ema_windows: tuple[int, ...] = (3, 10, 30)
    volatility_window: int = 5
    rsi_window: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_window: int = 20
    bollinger_num_std: float = 2.0
    ema_cross_short: int = 10
    ema_cross_long: int = 30
    volume_window: int = 20

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Named model inputs, in a stable order."""
        return (
            "hl_range",
            "oc_change",
            "return_1d",
            *(f"sma_{w}_ratio" for w in self.sma_windows),
            *(f"ema_{w}_ratio" for w in self.ema_windows),
            f"volatility_{self.volatility_window}",
            "ema_bullish",
            "ema_diff_pct",
            f"rsi_{self.rsi_window}",
            "bb_position",
            "macd_pct",
            "macd_hist_pct",
            "volume_ratio",
        )


@dataclass(frozen=True)
class ModelConfig:
    """Training settings shared by every model in the registry.

    Attributes:
        test_size: Fraction of the (chronologically ordered) rows held out for testing.
        cv_splits: Number of ``TimeSeriesSplit`` folds used for CV and tuning.
        random_state: Seed passed to every stochastic estimator.
        pca_components: If set, insert PCA between the scaler and every estimator.
        tune: Run a small time-series grid search for models that define a grid.
        importance_repeats: Permutation-importance shuffles per feature.
        models: Registry names to train.
    """

    test_size: float = 0.2
    cv_splits: int = 5
    random_state: int = 101
    pca_components: int | None = None
    tune: bool = True
    importance_repeats: int = 5
    n_jobs: int = 1
    models: tuple[str, ...] = (
        "logistic_regression",
        "svm_rbf",
        "linear_svc",
        "random_forest",
        "extra_trees",
        "bagging_knn",
        "gradient_boosting",
    )


@dataclass(frozen=True)
class BacktestConfig:
    """Strategy rules.

    Attributes:
        mode: ``"long_short"`` goes short on a predicted fall; ``"long_flat"`` holds cash.
        cost_bps: Cost in basis points charged per unit of position change.
        rolling_window: Window (trading days) for the rolling Sharpe chart.
    """

    mode: Literal["long_short", "long_flat"] = "long_short"
    cost_bps: float = 5.0
    rolling_window: int = 126
    periods_per_year: int = TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class ExperimentConfig:
    """Bundle of all configs needed to reproduce a training run."""

    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def config_hash(*configs: Any) -> str:
    """Return a short, stable hash of one or more dataclass configs.

    Args:
        *configs: Dataclass instances (or JSON-serialisable values).

    Returns:
        A 12-character hex digest, suitable for cache keys and file names.
    """
    payload = [asdict(c) if hasattr(c, "__dataclass_fields__") else c for c in configs]
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
