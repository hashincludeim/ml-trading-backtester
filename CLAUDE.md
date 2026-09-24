# CLAUDE.md

This file tells Claude (and human contributors) how this project is organised, what the goals are, and which conventions to follow. Read it fully before making changes.

## Project overview

**StockML** predicts whether a stock's closing price will rise the next trading day, backtests a long/short strategy built on those predictions, and presents the results in an interactive Django web dashboard.

The project started as a single Colab notebook export (`legacy/machine_learning_project.py`) that analyses Barclays (`BARC.L`) from 2000–2022. It is now a portfolio piece that demonstrates three things to employers:

1. **Clean, reusable Python**: a framework-agnostic core library with small, typed, tested functions.
2. **Strong data visualisation**: interactive, well-labelled Plotly charts that tell a story.
3. **Working with larger data**: many tickers, long histories, cached efficiently, processed without blocking web requests.

The legacy script is reference material only. Do not import from it or copy its structure. All 16 issues listed in "Known issues in the legacy script" have been fixed; keep them fixed.

### Current universe and data

- Tickers (`DataConfig.tickers`): **S&P 500** `^GSPC`, **Amazon** `AMZN`, **Microsoft** `MSFT`, **Alphabet (Google)** `GOOGL`, **Oracle** `ORCL`.
- History runs from `DataConfig.start` (2000-01-01; `GOOGL` from its 2004 IPO) **up to the system date**: `DataConfig.end` defaults to tomorrow because yfinance's `end` is exclusive.
- Display names, price units and currency come from `config.py` (`TICKER_NAMES`, `ticker_label()`, `price_unit()`, `currency_symbol()`): stocks are in USD, `^GSPC` in index points, `.L` tickers in GBX. **Never hard-code a unit, currency or ticker name** in charts, services or templates.
- The chronological 80/20 split puts the test period at roughly mid-2021 to the latest close. The last day has no next-day label and is dropped.

## Architecture

The core rule: **all data, feature, model, backtest, analysis and chart logic lives in `src/stockml/` and has no Django imports.** Django is a thin presentation layer that calls into the core library. This keeps the ML code reusable from a notebook, CLI, test, or any other web framework.

```
stockml/
├── CLAUDE.md                   # this file (git-ignored)
├── README.md
├── pyproject.toml              # deps, ruff, mypy, pytest config
├── .env.example                # DJANGO_SECRET_KEY, DJANGO_DEBUG, DATA_DIR, etc.
├── .claude/launch.json         # dev server for the preview pane (autoPort; honours $PORT)
├── Dockerfile, .dockerignore  # image for Google Cloud Run (serves only; data baked in, no training)
├── deploy/                     # start.sh (gunicorn entrypoint), gcp-setup.sh (one-time Cloud Run + keyless GitHub auth)
├── .github/workflows/deploy.yml  # checks; on main + weekday nights: fetch, train, push image to GHCR, deploy to Cloud Run
├── legacy/
│   └── machine_learning_project.py   # original notebook export (read-only reference)
├── src/stockml/                # framework-agnostic core library
│   ├── config.py               # frozen dataclasses: DataConfig, FeatureConfig, ModelConfig,
│   │                           #   BacktestConfig, AnalysisConfig, ExperimentConfig; ticker
│   │                           #   names/units; config_hash()
│   ├── data/
│   │   ├── loader.py           # download_prices(), load_prices() with Parquet cache
│   │   └── cleaning.py         # validate_prices(), detect_outliers(), replace_outliers()
│   ├── features/
│   │   ├── technical.py        # sma(), ema(), rsi(), macd(), bollinger(), rolling_volatility()
│   │   ├── target.py           # make_price_rise_target()
│   │   └── pipeline.py         # compute_indicators(), build_feature_frame(prices, config) -> (X, y)
│   ├── models/
│   │   ├── registry.py         # MODEL_REGISTRY: name -> ModelSpec building an sklearn Pipeline
│   │   ├── training.py         # chronological_split(), time_series_cv(), train_model(),
│   │   │                       #   validation_importance()
│   │   └── persistence.py      # save_model()/load_model() via joblib
│   ├── evaluation/
│   │   ├── metrics.py          # classification metrics, evaluate_model(), sharpe(), sortino(),
│   │   │                       #   calmar(), max_drawdown(), rolling_sharpe()
│   │   └── backtest.py         # run_backtest() -> BacktestResult, cost_sensitivity()
│   ├── analysis.py             # descriptive stats behind explanatory charts (drawdown episodes,
│   │                           #   calendar returns, ACF, feature buckets, indicator signals)
│   ├── experiment.py           # run_experiment(): features -> split -> every model -> evaluation
│   └── viz/
│       ├── theme.py            # palette, DARK_COLOR_MAP, template, apply_theme(), date_axes()
│       └── charts.py           # functions returning plotly.graph_objects.Figure
├── web/                        # Django project
│   ├── manage.py
│   ├── config/                 # settings/ (base.py, test.py), urls.py, wsgi.py, asgi.py
│   └── dashboard/              # the Django app
│       ├── models.py           # Ticker, TrainingRun, ModelResult (metadata only; prices stay in Parquet)
│       ├── services.py         # the ONLY place views call into stockml
│       ├── views.py            # thin: parse request -> call service -> render
│       ├── forms.py
│       ├── urls.py
│       ├── context_processors.py   # nav, Plotly.js version, dark-mode colour map
│       ├── templatetags/dashboard_tags.py   # plotly_chart tag, pct/num filters
│       ├── management/commands/
│       │   ├── fetch_prices.py     # re-downloads to today by default; --use-cache to skip
│       │   ├── train_models.py     # --ticker SYMBOL (repeatable) | --all; --models; --no-tune
│       │   └── remove_tickers.py   # drop tickers + runs; --delete-files removes cached data
│       ├── templates/dashboard/
│       └── static/dashboard/       # dashboard.css, dashboard.js (rendering + theme toggle)
├── notebooks/
│   └── exploration.ipynb       # imports from stockml; no business logic defined here
├── data/                       # git-ignored: prices/*.parquet, runs/<ticker>/..., db.sqlite3
└── tests/
    ├── core/                   # unit tests for src/stockml
    └── web/                    # Django view/service tests
```

### Data flow

```
yfinance -> data.loader (Parquet cache) -> data.cleaning -> features.pipeline -> (X, y)
        -> models.training (TimeSeriesSplit CV, fit) -> predictions
        -> evaluation.backtest + evaluation.metrics + analysis -> results
        -> viz.charts (Plotly Figure) -> dashboard.services -> template (Plotly JSON)
```

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

# Quality checks (run all before finishing any task)
ruff check . && ruff format --check .
mypy src
pytest

# Data and models (default universe: ^GSPC AMZN MSFT GOOGL ORCL, up to today)
python web/manage.py fetch_prices                    # re-download every ticker to today
python web/manage.py fetch_prices NVDA --start 2010-01-01
python web/manage.py train_models --all --jobs 4     # ~1 min per ticker
python web/manage.py train_models --ticker MSFT --models random_forest --no-tune
python web/manage.py remove_tickers NVDA --delete-files

# Run the app
python web/manage.py migrate
python web/manage.py runserver
```

After fetching new data, re-run `train_models` so models and backtests include the new days.

## Coding conventions

### General
- Python 3.11+. Type hints on every function signature. `mypy src` must pass (strict; `python_version = 3.12` in `pyproject.toml` because numpy's stubs use 3.12 syntax).
- Formatting and linting via `ruff` (line length 100). No unused imports; all imports at the top of the module.
- Google-style docstrings on public functions, stating inputs, outputs, and any look-ahead assumptions.
- Small, pure functions: take a DataFrame/Series in, return a new one. Never mutate inputs; never rely on global state.
- No magic numbers or hard-coded dates/tickers in logic. Put them in `config.py` dataclasses with sensible defaults (market events live in `AnalysisConfig.events`).
- Select columns **by name**, never by position (`df.iloc[:, 4:-5]` is banned). Keep feature lists in `FeatureConfig`.
- Use `logging`, not `print`, in library code.
- Keep compatibility with current libraries: use `np.nan` (not `np.NaN`), `.ffill()` (not `fillna(method="ffill")`), and handle yfinance's MultiIndex columns and `auto_adjust` default explicitly.

### Machine learning rules (non-negotiable)
- **No look-ahead or leakage.** Every feature at row *t* may use only data available at the close of day *t*. The target at *t* is whether close *t+1* > close *t*. Drop the final row, whose target is unknown.
- **Chronological splits only.** Use a train/test split with `shuffle=False` and `TimeSeriesSplit` for cross-validation. Never use plain `KFold` on this data.
- **All preprocessing inside an sklearn `Pipeline`** (scaler -> optional PCA/feature selection -> estimator), so that it is fitted on training folds only. Never call `fit_transform` on the full dataset or on already-scaled data.
- Feature selection (RFE, SelectKBest, permutation importance) is fitted on training data only, never on the test set.
- The model evaluated, backtested, and plotted must be the **same fitted pipeline**. Do not refit a different variant for the classification report.
- Classification metrics only for classifiers: accuracy, precision, recall, F1, ROC AUC, confusion matrix. Do not report MSE or R² for a binary target.
- Always report a naive baseline next to model results. The dashboard uses the **majority-class** accuracy (max of always-up / always-down in the test period) and buy-and-hold. Be honest in the UI when a model does not beat it.
- **Never select a model by test-period results.** Cross-ticker "best model" choices use training-period CV ROC AUC so the reported test numbers stay out-of-sample.
- Fix `random_state` everywhere for reproducibility.

### Backtesting rules
- Log returns; the position at *t* (from the prediction at *t*) earns the return from *t* to *t+1*.
- `BacktestConfig` supports long/short vs long/flat and a transaction cost in basis points, charged on position changes (including the initial entry).
- Annualise Sharpe and Sortino with √252; compute max drawdown on the equity curve; Calmar = annualised return / max drawdown.
- Histograms and distributions use **daily** returns, never cumulative returns.

### Visualisation rules
- All charts are built in `stockml/viz/charts.py` as functions returning `plotly.graph_objects.Figure`. They take data in and return a figure; no I/O, no Django.
- Apply the shared theme with `apply_theme()` (title, optional one-line `subtitle` saying how to read the chart, axis labels, legend-aware top margin). Use `date_axes()` on time-series figures.
- Every chart has a title, labelled axes with units, and readable hover text. Use `model_color()` so each model keeps one colour across all charts; buy & hold is the neutral dashed benchmark.
- Units come from the caller (`unit=price_unit(ticker)`, `currency=currency_symbol(ticker)`); titles use `ticker_label(ticker)`.
- **Dark mode:** figures are built in the light palette and the browser swaps colours using `DARK_COLOR_MAP` in `theme.py`. Any new colour used in a chart must be a named constant in `theme.py` with an entry in `DARK_COLOR_MAP`.
- Prefer interactive, story-driven charts, and pair explanatory charts with a data-driven caption (computed in `services.py`) that states the takeaway honestly.
- Use `make_subplots(shared_xaxes=True)` for price + indicator views rather than twin axes.
- For long histories, downsample (`downsample_ohlcv`), pre-bin histograms (`analysis.binned_density`), or use `Scattergl` so pages stay fast.
- Plotly pitfalls we have hit (and tests guard):
  - Add `add_hrect/add_hline(row=, col=)` shapes **after** that subplot has a trace, or Plotly silently skips them.
  - Never pass `annotation_*` kwargs to `add_hline` without text: Plotly inserts a "new text" placeholder, which on a log axis lands at 10^y.
  - Annotations on log axes use log10 coordinates (e.g. y=0 for a value of 1).
  - Long annotations stretch autorange; pin the axis range or keep labels short.
  - The chart container needs an explicit height (set in `dashboard.js` from `layout.height`), or autosized plots collapse to 0px.
  - `autorangeoptions.minallowed/maxallowed` on date axes must be **epoch milliseconds**; date strings make plotly.js fall back to a 2000–2001 range.
- **Phones (≤640px):** `phoneFigure()` in `dashboard.js` restyles figures client-side: title/subtitle move into HTML above the chart, the legend goes below the plot, the toolbar is hidden, heatmap cell text is dropped, and annotations named `EVENT_ANNOTATION` become numbers listed under the title. Keep chart code desktop-first; name any new event-style labels `EVENT_ANNOTATION`. Tables mark their phone columns with `data-key-cols` (1-based); the rest sit behind "Show all columns". Check new pages at 375px wide in both themes.

### Django rules
- Views stay thin: validate input, call a function in `dashboard/services.py`, render. No pandas or sklearn code in views or templates.
- Charts reach templates as JSON (`services.figure_json`, rendered via the `{% plotly_chart %}` tag) and render with Plotly.js; do not embed full HTML per figure.
- Never download data or train models inside a request. Use the management commands (or a background worker if added later) and read cached results in views.
- Cache expensive service calls with Django's cache framework, keyed by ticker + run id + config hash + price-file mtime. The cache is in-process (LocMem), so **restart the dev server after changing chart or service code**, or it keeps serving old figures.
- File paths saved in the database are relative to `DATA_DIR` (`services.stored_path`/`data_path`), so data built in CI works inside the container.
- Price data lives in Parquet files under `DATA_DIR` (index symbols keep their `^`, e.g. `^GSPC.parquet`); the database stores only metadata (tickers, training runs, metrics, model file paths).
- Settings come from environment variables (`os.environ`, with a minimal `.env` reader). Relative paths in `DATA_DIR`/`DATABASE_PATH` resolve against the project root. No secrets in the repo. `DEBUG=False` must work (WhiteNoise serves static files).
- Use class-based views where they reduce code; templates extend a single `base.html`.

### Pages (implemented)
1. **Overview**: candlesticks with SMA 50/200 and market-event markers, volume, drawdown from the all-time high, rolling volatility, annual returns, year × month heatmap.
2. **Indicators**: SMA/EMA, Bollinger, RSI (shaded zones), MACD with toggles; EMA crossover markers; next-day outcome after each indicator state.
3. **Exploration**: return autocorrelation, fat tails vs normal, feature vs next-day-return correlation with noise band, up-rate by feature decile, distributions, target balance, feature correlations.
4. **Models**: comparison table vs naive baseline, CV mean ± std and per-fold stability, ROC curves, rolling test accuracy, score separation, confusion matrix, permutation importance.
5. **Backtest**: equity curves vs buy-and-hold, risk table, cost sensitivity, risk vs return, drawdowns, rolling Sharpe, return histogram, monthly returns, transaction-cost slider.
6. **Multi-ticker**: ticker × model heatmap, CV-selected model vs buy & hold, rebased prices, risk vs return per ticker, cross-ticker return correlation.

Light is the default theme (it does not follow the OS); a header toggle switches to dark and stores the choice in `localStorage`.

### Visual design
- Editorial research-note look: **Newsreader** (serif) for page titles, section heads and verdict headlines; **Instrument Sans** for everything else, including chart text (`FONT_FAMILY` in `theme.py`); JetBrains Mono for code only.
- Chrome stays black, white and grey so the data carries the colour. No card boxes or shadows: figures sit on the page under a hairline, sections open with an ink rule, key figures form a ruled strip.
- Each page's headline finding is a `.callout` with a label (e.g. "Verdict"), a serif `.callout-head` and a `.callout-body`. Captions under charts carry a small label: "Takeaway" (data-driven, via `_insight.html`), "How to read" or "Method".
- The page colour equals `theme.SURFACE` in both themes (`--bg` in `dashboard.css` matches `DARK_COLOR_MAP[SURFACE]`), so charts blend into the page. Keep the two in sync.

## Testing
- Every function in `src/stockml` gets unit tests with small synthetic DataFrames (no network in tests; mock yfinance).
- Include explicit **no-leakage tests**: altering future prices must not change features at earlier rows.
- Backtest tests with hand-computed expected returns on tiny series.
- Chart tests assert the returned object is a `Figure` with the expected traces, titles and colours (including regression tests for the Plotly pitfalls above).
- Django tests cover each view's status code and that services are called with the right arguments; service tests run a small end-to-end train on synthetic data.
- Tests must not depend on the system date except through `today_inclusive_end()`.

## Known issues in the legacy script (all fixed; keep them fixed)

| # | Issue | Fix |
|---|-------|-----|
| 1 | Outlier "IQR" uses `Q2-Q1` and `Q4-Q3`, and the replacement is hard-coded to `2022-06-14` | Standard IQR on returns (not price level) or a rolling z-score; replace generically |
| 2 | Columns selected with `iloc[:, 4:-5]`; meaning changes once `Y_pred` is appended | Named feature list in `FeatureConfig` |
| 3 | `macd()` ignores its `data` argument and uses the global `dataset` | Pure function taking a Series |
| 4 | Linear regression and AutoReg applied to a binary target and scored with R²/MSE | Drop, or keep only classification models |
| 5 | AutoReg, SVM-linear, and SVM-rbf sections print R²/MSE using a stale `Y_pred` from an earlier model | Metrics computed from that model's own predictions |
| 6 | SelectKBest section cross-validates `hypothesis_svm_rfe` and prints `scores_rfe` (wrong variables) | Use the correct pipeline |
| 7 | Permutation importance computed on the test set, then predicts with `X_test_kbest` (wrong features) | Compute on a validation fold; use matching features |
| 8 | Classification reports refit models on full `X_train` instead of the PCA/RFE variant that was cross-validated | Evaluate the same fitted pipeline |
| 9 | `StandardScaler` repeatedly `fit_transform`ed on already-scaled arrays; PCA fitted on the full dataset | Scaler and PCA inside a `Pipeline` |
| 10 | `cross_val_score` uses default KFold on time series | `TimeSeriesSplit` |
| 11 | Gradient Boosting with `max_depth=100` massively overfits | Tune shallow trees (depth 2–5) with time-series CV |
| 12 | Sharpe/Sortino not annualised; Calmar uses mean daily return; no transaction costs | See backtesting rules |
| 13 | Histograms plot cumulative returns | Plot daily returns |
| 14 | `EMA_crossover` is 1 when the short EMA is *below* the long EMA, contrary to its description | Rename to `ema_bullish` = 1 when short > long |
| 15 | Deprecated APIs: `np.NaN`, `fillna(method="ffill")`; yfinance MultiIndex columns not handled | Modern equivalents |
| 16 | Repeated copy-pasted evaluation/plot blocks per model | One `evaluate_model()` + `run_backtest()` looped over `MODEL_REGISTRY` |

## Working style for Claude
- Work in small, reviewable steps: core library module + its tests first, then the service, then the view/template.
- Before finishing a task, run `ruff`, `mypy src`, and `pytest`, and fix failures. For UI changes, also check the page in the browser (both themes).
- Commit only when asked, on a feature branch (not directly on `main`), and note in the commit message which legacy issue numbers a change resolves.
- If a requirement here conflicts with a user instruction, follow the user and mention the conflict.
- Do not add new dependencies without stating why; prefer the existing stack: pandas, numpy, scikit-learn, yfinance, pyarrow, plotly, Django, statsmodels, joblib, whitenoise, pytest, ruff, mypy.
