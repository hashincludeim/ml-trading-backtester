# StockML

Predict whether a stock will close higher tomorrow, backtest a long/short strategy on those
predictions, and explore everything in an interactive Django + Plotly dashboard.

It began as a single Colab notebook analysing Barclays (`BARC.L`) from 2000 to 2022
([`legacy/machine_learning_project.py`](legacy/machine_learning_project.py)). This repo rebuilds it
as a tested, typed library with a thin web layer, and fixes the leakage and evaluation problems in
the original. The table at the bottom lists each one.

## What's inside

| Layer | Where | Notes |
|---|---|---|
| Core library | `src/stockml/` | No Django imports. Small, pure, typed functions; `mypy --strict` clean. |
| Data | `data/loader.py`, `data/cleaning.py` | yfinance → Parquet cache; handles MultiIndex columns and `auto_adjust` explicitly. Bad ticks are flagged with a volatility-scaled z-score that needs an immediate reversal, so genuine crashes (2008–09) are kept. |
| Features | `features/` | SMA/EMA ratios, RSI (Wilder), MACD, Bollinger position, volatility, EMA trend, volume ratio. Every feature at *t* uses data up to the close of *t* only. Tests check this by changing future prices and asserting earlier features stay the same. |
| Models | `models/registry.py` | 7 classifiers, each built as `StandardScaler → [PCA] → estimator` inside one sklearn `Pipeline`. Small grids are tuned with `TimeSeriesSplit`. |
| Evaluation | `evaluation/` | Accuracy/precision/recall/F1/ROC AUC with naive (always-up / majority) baselines. Vectorised backtest with long/short or long/flat, costs in bp, and annualised Sharpe/Sortino/Calmar. |
| Analysis | `analysis.py` | Descriptive statistics behind the explanatory charts: drawdown episodes, calendar returns, autocorrelation, feature-bucket up-rates, indicator signal tables. |
| Charts | `viz/charts.py` | Functions that return `go.Figure`, all using one theme (`viz/theme.py`). Each model keeps the same colour on every chart, and the palette is colour-blind-validated. Dark mode comes from a single light→dark colour map in the theme, which the browser applies when toggling. |
| Web | `web/dashboard/` | Views only parse input, call `services.py` and render. Figures go to templates as JSON. Results are cached in Django's cache. Training never runs inside a request. |

### Dashboard pages

Each chart has a one-line "how to read this" subtitle, and key charts carry a caption computed
from the data (for example "0 of 17 features clear the noise band"). A header toggle switches
between light and dark themes, remembers the choice, and otherwise follows the OS setting.

1. **Overview**: candlesticks with 50/200-day trend lines and annotated market events (Lehman,
   Eurozone, Brexit, COVID), volume, drawdown from the all-time high, rolling volatility,
   calendar-year returns, and a year × month returns heatmap.
2. **Indicators**: SMA/EMA/Bollinger overlays you can toggle, EMA crossover markers, RSI with
   overbought/oversold zones, MACD, and a "what happened the day after each signal?" dot plot
   with 95% intervals.
3. **Exploration**: return autocorrelation and fat tails against a normal curve, each feature's
   rank correlation with the next-day return against a noise band, next-day up-rate by feature
   decile, feature distributions by outcome, target balance, and feature redundancy.
4. **Models**: comparison table against the naive baseline, CV mean ± std and fold-by-fold
   stability, ROC curves, rolling test-set accuracy, how well each model's scores separate up
   days from down days, confusion matrix, and permutation importance.
5. **Backtest**: equity curves against buy & hold, a risk table, how Sharpe decays as costs rise,
   risk vs return, drawdowns, rolling Sharpe, daily-return histogram, monthly returns, and a live
   transaction-cost slider.
6. **Multi-ticker**: ticker × model heatmap, CV-selected model vs buy & hold, rebased prices,
   buy & hold risk vs return per stock, and cross-ticker return correlation.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # sets DJANGO_DEBUG=true and a dev secret key

python web/manage.py migrate
python web/manage.py fetch_prices                 # BARC.L HSBA.L LLOY.L NWG.L STAN.L, 2000–2022
python web/manage.py train_models --all --jobs 4  # ~1 min per ticker
python web/manage.py runserver
```

Pick your own universe or period with `fetch_prices AZN.L SHEL.L --start 2010-01-01 --end 2024-12-31`.
Use `train_models --ticker AZN.L --models random_forest gradient_boosting --no-tune` for a quick run.

### Quality checks

```bash
ruff check . && ruff format --check .
mypy src
pytest
```

The tests use synthetic prices and never touch the network (yfinance is mocked).

### Production-style run

```bash
DJANGO_DEBUG=false DJANGO_SECRET_KEY=... python web/manage.py collectstatic --noinput
DJANGO_DEBUG=false DJANGO_SECRET_KEY=... python web/manage.py runserver --insecure  # or gunicorn config.wsgi
```

WhiteNoise serves static files, so `DEBUG=False` works without a separate web server.

## Results, honestly

Next-day direction for large UK banks is close to a coin flip. After the leakage fixes, test
accuracy across five UK banks falls between roughly 46% and 54%, and ROC AUC between roughly
0.49 and 0.55. The dashboard compares every model with the best naive guess (always predicting
the more common direction) and with buy & hold, and it says so when the models don't beat them.
The multi-ticker page picks each ticker's model by training-period CV, not by hindsight. The cost slider shows how quickly turnover wipes out any small
edge. The point of the project is a sound methodology and clear communication, not a
money-printing model.

## Legacy issues fixed

| # | Issue in the notebook | Fix here |
|---|---|---|
| 1 | "IQR" used `Q2-Q1` / `Q4-Q3`; replacement hard-coded to `2022-06-14` | `detect_outliers()` uses a rolling z-score (or standard IQR) on log returns and requires a reversal. `replace_outliers()` is generic. |
| 2 | Features picked with `iloc[:, 4:-5]` | Named `FeatureConfig.feature_columns` |
| 3 | `macd()` read a global | `technical.macd(close, …)` is pure. A test checks it uses its argument. |
| 4 | Linear regression / AutoReg on a binary target, scored with R²/MSE | Classifiers only, with classification metrics only |
| 5 | R²/MSE printed from a stale `Y_pred` | `evaluate_model()` uses each model's own predictions |
| 6 | SelectKBest section cross-validated the wrong pipeline | One registry loop, so there are no copy-paste variables |
| 7 | Permutation importance on the test set, then predicted with the wrong features | `validation_importance()` uses the last training fold and the original feature names |
| 8 | Reports refit a different variant from the one cross-validated | The pipeline that is tuned, CV'd and fitted is also the one evaluated, backtested and plotted |
| 9 | Scaler re-fitted on scaled data; PCA fitted on the full dataset | Scaler/PCA sit inside the `Pipeline` and are fitted per fold |
| 10 | Default KFold on time series | `TimeSeriesSplit` everywhere, and `shuffle=False` for the split |
| 11 | Gradient boosting `max_depth=100` | Depth grid 2–5, chosen by time-series CV |
| 12 | Sharpe/Sortino not annualised; Calmar used mean daily return; no costs | √252 annualisation, Calmar = CAGR / max DD, costs in bp on position changes |
| 13 | Histograms of cumulative returns | Daily returns |
| 14 | `EMA_crossover` = 1 when the short EMA was *below* the long EMA | `ema_bullish` = 1 when short > long (tested) |
| 15 | `np.NaN`, `fillna(method="ffill")`, yfinance MultiIndex | Modern APIs; MultiIndex handled and tested |
| 16 | Copy-pasted blocks per model | `run_experiment()` loops over `MODEL_REGISTRY` |

Other changes: raw SMA/EMA price levels were replaced by ratios to the close, so features are
comparable across time and across tickers. RSI now uses Wilder's smoothing (`alpha = 1/n`).

## Layout

```
src/stockml/      config · data · features · models · evaluation · viz · experiment.py
web/              Django project (config/) + dashboard app
tests/core        library unit tests (incl. no-leakage and hand-computed backtests)
tests/web         service and view tests
notebooks/        exploration notebook importing from stockml
legacy/           original notebook export (reference only)
```
