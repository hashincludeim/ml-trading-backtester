# CLAUDE.md

This file tells Claude (and human contributors) how this project is organised, what the goals are, and which conventions to follow. Read it fully before making changes.

## Project overview

**StockML** predicts whether a stock's closing price will rise the next trading day, backtests a long/short strategy built on those predictions, and presents the results in an interactive Django web dashboard.

The project started as a single Colab notebook export (`legacy/machine_learning_project.py`) that analyses Barclays (`BARC.L`) from 2000–2022. We are turning it into a portfolio piece that demonstrates three things to employers:

1. **Clean, reusable Python**: a framework-agnostic core library with small, typed, tested functions.
2. **Strong data visualisation**: interactive, well-labelled Plotly charts that tell a story.
3. **Working with larger data**: many tickers, long histories, cached efficiently, processed without blocking web requests.

The legacy script is reference material only. Do not import from it or copy its structure. Port its *ideas* into the new architecture, fixing the issues listed in "Known issues in the legacy script".

## Architecture

The core rule: **all data, feature, model, backtest, and chart logic lives in `src/stockml/` and has no Django imports.** Django is a thin presentation layer that calls into the core library. This keeps the ML code reusable from a notebook, CLI, test, or any other web framework.

```
stockml/
├── CLAUDE.md
├── README.md
├── pyproject.toml              # deps, ruff, mypy, pytest config
├── .env.example                # DJANGO_SECRET_KEY, DEBUG, DATA_DIR, etc.
├── legacy/
│   └── machine_learning_project.py   # original notebook export (read-only reference)
├── src/stockml/                # framework-agnostic core library
│   ├── __init__.py
│   ├── config.py               # frozen dataclasses: DataConfig, FeatureConfig, ModelConfig, BacktestConfig
│   ├── data/
│   │   ├── loader.py           # download_prices(), load_prices() with Parquet cache
│   │   └── cleaning.py         # detect_outliers(), replace_outliers(), validate_prices()
│   ├── features/
│   │   ├── technical.py        # sma(), ema(), rsi(), macd(), bollinger(), rolling_volatility()
│   │   ├── target.py           # make_price_rise_target()
│   │   └── pipeline.py         # build_feature_frame(prices, config) -> (X, y)
│   ├── models/
│   │   ├── registry.py         # MODEL_REGISTRY: name -> factory returning an sklearn Pipeline
│   │   ├── training.py         # train_model(), time_series_cv()
│   │   └── persistence.py      # save_model()/load_model() via joblib
│   ├── evaluation/
│   │   ├── metrics.py          # classification metrics, sharpe(), sortino(), calmar(), max_drawdown()
│   │   └── backtest.py         # run_backtest(prices, predictions, config) -> BacktestResult
│   └── viz/
│       ├── theme.py            # shared colours, fonts, layout defaults
│       └── charts.py           # functions returning plotly.graph_objects.Figure
├── web/                        # Django project
│   ├── manage.py
│   ├── config/                 # settings/, urls.py, wsgi.py, asgi.py
│   └── dashboard/              # the Django app
│       ├── models.py           # Ticker, TrainingRun, ModelResult (metadata only; prices stay in Parquet)
│       ├── services.py         # the ONLY place views call into stockml
│       ├── views.py            # thin: parse request -> call service -> render
│       ├── forms.py
│       ├── urls.py
│       ├── management/commands/
│       │   ├── fetch_prices.py     # python manage.py fetch_prices BARC.L HSBA.L ...
│       │   └── train_models.py     # python manage.py train_models --ticker BARC.L
│       ├── templates/dashboard/
│       └── static/dashboard/
├── notebooks/
│   └── exploration.ipynb       # imports from stockml; no business logic defined here
└── tests/
    ├── core/                   # unit tests for src/stockml
    └── web/                    # Django view/service tests
```

### Data flow

```
yfinance -> data.loader (Parquet cache) -> data.cleaning -> features.pipeline -> (X, y)
        -> models.training (TimeSeriesSplit CV, fit) -> predictions
        -> evaluation.backtest + evaluation.metrics -> results
        -> viz.charts (Plotly Figure) -> dashboard.services -> template (Plotly JSON)
```

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Quality checks (run all before finishing any task)
ruff check . && ruff format --check .
mypy src
pytest

# Data and models
python web/manage.py fetch_prices BARC.L HSBA.L LLOY.L NWG.L
python web/manage.py train_models --ticker BARC.L

# Run the app
python web/manage.py migrate
python web/manage.py runserver
```

## Coding conventions

### General
- Python 3.11+. Type hints on every function signature. `mypy src` must pass.
- Formatting and linting via `ruff` (line length 100). No unused imports; all imports at the top of the module.
- Google-style docstrings on public functions, stating inputs, outputs, and any look-ahead assumptions.
- Small, pure functions: take a DataFrame/Series in, return a new one. Never mutate inputs; never rely on global state.
- No magic numbers or hard-coded dates/tickers in logic. Put them in `config.py` dataclasses with sensible defaults.
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
- Always report a naive baseline (e.g. "always predict up" and buy-and-hold) next to model results. Be honest in the UI when a model does not beat it.
- Fix `random_state` everywhere for reproducibility.

### Backtesting rules
- Log returns; the position at *t* (from the prediction at *t*) earns the return from *t* to *t+1*.
- `BacktestConfig` supports long/short vs long/flat and a transaction cost in basis points, charged on position changes.
- Annualise Sharpe and Sortino with √252; compute max drawdown on the equity curve; Calmar = annualised return / max drawdown.
- Histograms and distributions use **daily** returns, never cumulative returns.

### Visualisation rules
- All charts are built in `stockml/viz/charts.py` as functions returning `plotly.graph_objects.Figure`. They take data in and return a figure; no I/O, no Django.
- Apply the shared theme from `viz/theme.py` to every figure (consistent palette, fonts, hover formatting, margins).
- Every chart has a title, labelled axes with units, and readable hover text. Use a consistent colour for each model across all charts.
- Prefer interactive, story-driven charts: candlestick with volume, overlays for SMA/EMA/Bollinger with toggles, RSI/MACD subplots sharing the x-axis, equity curves for all models vs buy-and-hold, drawdown chart, rolling Sharpe, confusion matrix heatmap, ROC curves for all models on one plot, feature importance mapped to original feature names, correlation heatmap.
- Use `make_subplots(shared_xaxes=True)` for price + indicator views rather than twin axes.
- For long histories, downsample or use `Scattergl` so pages stay fast.

### Django rules
- Views stay thin: validate input, call a function in `dashboard/services.py`, render. No pandas or sklearn code in views or templates.
- Charts reach templates as JSON (`fig.to_json()`) and render with Plotly.js; do not embed full HTML per figure.
- Never download data or train models inside a request. Use the management commands (or a background worker if added later) and read cached results in views.
- Cache expensive service calls with Django's cache framework, keyed by ticker + config hash.
- Price data lives in Parquet files under `DATA_DIR`; the database stores only metadata (tickers, training runs, metrics, model file paths).
- Settings come from environment variables (`django-environ` or `os.environ`). No secrets in the repo. `DEBUG=False` must work.
- Use class-based views where they reduce code; templates extend a single `base.html`.

### Suggested pages
1. **Overview**: pick a ticker and date range; candlestick, volume, key stats.
2. **Indicators**: SMA/EMA, Bollinger, RSI, MACD with toggles; EMA crossover markers.
3. **Exploration**: feature distributions, correlation heatmap, target balance.
4. **Models**: comparison table (CV mean ± std, test metrics), ROC curves, confusion matrices, feature importance.
5. **Backtest**: equity curves vs buy-and-hold, drawdowns, risk metrics table, transaction-cost slider.
6. **Multi-ticker**: one model across many tickers, showing the pipeline scales beyond a single stock.

## Testing
- Every function in `src/stockml` gets unit tests with small synthetic DataFrames (no network in tests; mock yfinance).
- Include explicit **no-leakage tests**: altering future prices must not change features at earlier rows.
- Backtest tests with hand-computed expected returns on tiny series.
- Chart tests assert the returned object is a `Figure` with the expected traces and titles.
- Django tests cover each view's status code and that services are called with the right arguments.

## Known issues in the legacy script (fix these when porting)

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
- Before finishing a task, run `ruff`, `mypy src`, and `pytest`, and fix failures.
- When porting a legacy section, note in the PR/commit message which issue numbers above it resolves.
- If a requirement here conflicts with a user instruction, follow the user and mention the conflict.
- Do not add new dependencies without stating why; prefer the existing stack: pandas, numpy, scikit-learn, yfinance, pyarrow, plotly, Django, statsmodels, joblib, pytest, ruff, mypy.