# StockML

Predict whether a stock will close higher tomorrow, backtest a long/short strategy on those
predictions, and explore everything in an interactive Django + Plotly dashboard.

It began as a single Colab notebook analysing Barclays (`BARC.L`) from 2000 to 2022
([`legacy/machine_learning_project.py`](legacy/machine_learning_project.py)). This repo rebuilds it
as a tested, typed library with a thin web layer, and fixes the leakage and evaluation problems in
the original. The table at the bottom lists each one.

## Universe and data

| Ticker | Name | Quoted in | History |
|---|---|---|---|
| `^GSPC` | S&P 500 index | index points | 2000 → today |
| `AMZN` | Amazon | USD | 2000 → today |
| `MSFT` | Microsoft | USD | 2000 → today |
| `GOOGL` | Alphabet (Google), class A | USD | Aug 2004 (IPO) → today |
| `ORCL` | Oracle | USD | 2000 → today |

Prices are daily OHLCV from Yahoo Finance, adjusted for splits and dividends. Downloads run up to
the **system date** (`DataConfig.end` defaults to tomorrow, because yfinance treats `end` as
exclusive). The universe, start date, display names and units are all set in
`src/stockml/config.py`.

## What's inside

| Layer | Where | Notes |
|---|---|---|
| Core library | `src/stockml/` | No Django imports. Small, pure, typed functions; `mypy --strict` clean. |
| Data | `data/loader.py`, `data/cleaning.py` | yfinance → Parquet cache; handles MultiIndex columns and `auto_adjust` explicitly. Bad ticks are flagged with a volatility-scaled z-score that needs an immediate reversal, so genuine crashes (2008, 2020) are kept. |
| Features | `features/` | SMA/EMA ratios, RSI (Wilder), MACD, Bollinger position, volatility, EMA trend, volume ratio. Every feature at *t* uses data up to the close of *t* only. Tests check this by changing future prices and asserting earlier features stay the same. |
| Models | `models/registry.py` | 7 classifiers, each built as `StandardScaler → [PCA] → estimator` inside one sklearn `Pipeline`. Small grids are tuned with `TimeSeriesSplit`. |
| Evaluation | `evaluation/` | Accuracy/precision/recall/F1/ROC AUC with naive (always-up / majority) baselines. Vectorised backtest with long/short or long/flat, costs in bp, and annualised Sharpe/Sortino/Calmar. |
| Analysis | `analysis.py` | Descriptive statistics behind the explanatory charts: drawdown episodes, calendar returns, autocorrelation, feature-bucket up-rates, indicator signal tables. |
| Charts | `viz/charts.py` | Functions that return `go.Figure`, all using one theme (`viz/theme.py`). Each model keeps the same colour on every chart, and the palette is colour-blind-validated. Dark mode comes from a single light→dark colour map in the theme, which the browser applies when toggling. |
| Experiment | `experiment.py` | `run_experiment()` builds features, makes the chronological split, and trains and evaluates every registered model in one loop. |
| Web | `web/dashboard/` | Views only parse input, call `services.py` and render. Figures go to templates as JSON. Results are cached in Django's cache. Training never runs inside a request. |

### Dashboard pages

Each chart has a one-line "how to read this" subtitle, and key charts carry a caption computed
from the data (for example "0 of 17 features clear the noise band"). Pages open in the light
theme; a header toggle switches to dark and remembers the choice.

1. **Overview**: candlesticks with 50/200-day trend lines and annotated market events (dot-com peak,
   Lehman, COVID, Fed hikes, ChatGPT launch), volume, drawdown from the all-time high, rolling volatility,
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
python web/manage.py fetch_prices                 # ^GSPC AMZN MSFT GOOGL ORCL, 2000 to today
python web/manage.py train_models --all --jobs 4  # ~1 min per ticker
python web/manage.py runserver
```

Then open http://127.0.0.1:8000.

### Keeping data current

```bash
python web/manage.py fetch_prices                 # re-download every ticker up to today
python web/manage.py train_models --all --jobs 4  # retrain so backtests include the new days
```

If you fetch during US trading hours, today's bar is provisional until the 16:00 New York close.
A running server picks up new data and new training runs automatically.

### Management commands

| Command | What it does |
|---|---|
| `fetch_prices [TICKER ...] [--start DATE] [--end DATE] [--use-cache]` | Downloads daily prices into `data/prices/*.parquet` and records ticker metadata. With no tickers it fetches the default universe. It re-downloads to today unless you pass `--use-cache`. |
| `train_models (--ticker T ... \| --all) [--models M ...] [--no-tune] [--jobs N]` | Trains the registered models for each ticker, then saves the fitted pipelines, test-set predictions and metrics as a new training run. |
| `remove_tickers TICKER ... [--delete-files]` | Removes tickers and their training runs from the dashboard. With `--delete-files` it also deletes their cached prices and saved models. |

Example of adding a stock: `fetch_prices NVDA --start 2010-01-01`, then `train_models --ticker NVDA`.
Add a display name for it in `TICKER_NAMES` in `config.py`, or it shows as the raw symbol.

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

### Deploying (Google Cloud Run)

The live dashboard runs on Google Cloud Run. It scales to zero when nobody is visiting, so
portfolio-level traffic stays inside the free tier (2 million requests, 180,000 vCPU-seconds and
360,000 GiB-seconds a month). The container only serves pages.
[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) runs the checks, downloads prices,
trains the models, and builds the data into the image. It pushes the image to GitHub Container
Registry (`ghcr.io/<owner>/<repo>`, public) and deploys it to Cloud Run. It runs on every push to
`main` and each weekday at 22:30 UTC, after the US close. If the download or training fails,
nothing is deployed and the site keeps the previous day's data. GitHub signs in to Google Cloud
through Workload Identity Federation, so no key file is stored anywhere.

One-time setup:

1. In the [Google Cloud console](https://console.cloud.google.com), create a project and link a
   billing account (Cloud Run requires one, even within the free tier). Note the project ID.
2. Open Cloud Shell (the terminal icon at the top right of the console), then download and run
   the setup script:

   ```bash
   curl -fsSLO https://raw.githubusercontent.com/<owner>/<repo>/main/deploy/gcp-setup.sh
   bash gcp-setup.sh <project-id> <owner>/<repo>
   ```

   The region defaults to `europe-west2` (London). Pass a third argument to change it.
3. The script prints four values. In this GitHub repo, go to Settings → Secrets and variables →
   Actions → **Variables** and add them: `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_WIF_PROVIDER`,
   `GCP_SERVICE_ACCOUNT`.
4. Run the workflow (Actions → CI and deploy → Run workflow). The run summary shows the live URL,
   `https://stockml-<hash>.<region>.run.app`.
5. Set a budget alert (Billing → Budgets & alerts), for example $5 a month with email alerts, so
   any unexpected usage is caught early.

Until the variables are set, the workflow still builds and pushes the image and skips the deploy
step. The first visit after an idle period waits a few seconds while an instance starts.
`DJANGO_ALLOWED_HOSTS` defaults to `.run.app`; for a custom domain, add it with
`--set-env-vars` in the deploy step.

To try the image locally, after `fetch_prices` and `train_models`:

```bash
docker build -t stockml . && docker run --rm -p 8000:8000 stockml
```

### Troubleshooting

- **Charts don't reflect a code change.** Figures are cached in memory by the running server,
  keyed by data and training run but not by code. Restart `runserver`.
- **The page says "No price data yet" or "No trained models yet".** Run `fetch_prices`, then
  `train_models --all`. The web app never downloads or trains during a request.
- **Port 8000 is taken.** Run `python web/manage.py runserver 8001`. The preview config in
  `.claude/launch.json` picks a free port automatically.

## Results, honestly

Next-day direction for the S&P 500 and large US tech stocks is close to a coin flip. On the latest
run (test period mid-2021 to September 2026):

- Test accuracy across the 35 model/ticker pairs falls between roughly 49% and 54%, and ROC AUC
  between 0.48 and 0.53.
- None of the models chosen by training-period CV beats the naive baseline on accuracy. The
  baseline always predicts whichever direction was more common in the test period.
- Some features are statistically detectable but economically tiny. For the S&P 500, 12 of 17
  features clear the 95% noise band (short-term mean reversion in the index), but no feature on
  any ticker has |ρ| above 0.07 with the next day's return. That explains under 0.5% of its
  variance, which is too little to survive trading costs.

The dashboard says all this on the page, not just in the README:

- It compares every model with that naive baseline and with buy & hold, and flags when a model
  doesn't beat them.
- It picks each ticker's model by training-period CV, not by hindsight.
- It shows how quickly trading costs wipe out any small edge.

The point of the project is a sound methodology and clear communication, not a money-printing
model. Numbers change slightly each time you refresh the data and retrain.

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
src/stockml/
  config.py         universe, dates, windows, model/backtest/analysis settings, ticker names/units
  data/             loader (yfinance + Parquet cache), cleaning (validation, outlier repair)
  features/         technical indicators, next-day target, feature pipeline
  models/           registry of sklearn pipelines, time-series training, joblib persistence
  evaluation/       classification + risk metrics, backtest, cost sensitivity
  analysis.py       descriptive stats for the explanatory charts
  experiment.py     end-to-end train/evaluate loop over the model registry
  viz/              theme (light palette + dark colour map) and chart builders
web/
  config/           settings (env-driven), urls, wsgi/asgi
  dashboard/        models, services, views, forms, commands, templates, static JS/CSS
tests/core          library unit tests (incl. no-leakage and hand-computed backtests)
tests/web           service and view tests
notebooks/          exploration notebook importing from stockml
legacy/             original notebook export (reference only)
data/               (git-ignored) cached prices, training runs, SQLite database
deploy/             container entrypoint (gunicorn) and one-time Google Cloud setup script
.github/workflows/  CI checks, nightly data refresh, image build and deploy to Cloud Run
```
