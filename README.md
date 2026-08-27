# ATLAS — AI Trading & Analysis System

Stock analysis terminal for Indian equities: a cross-sectional ML ranking
model with SHAP-based explanations, plus unsupervised anomaly detection.

## What the model actually does

ATLAS ranks the NSE universe by **expected relative performance over the next
3 sessions**. The buy score is a universe percentile — 10 means "top of the
universe today", 0 means "bottom". It is **not** a forecast of absolute
return, and the terminal does not present it as one.

This framing is deliberate. Absolute daily direction on liquid large caps is
close to unpredictable; relative rank carries measurable signal. See
[docs/MODEL_VALIDATION.md](docs/MODEL_VALIDATION.md) for the evidence,
including the ideas that failed.

### Measured performance

All figures are out-of-sample, from purged walk-forward validation:

| Metric | Value |
|---|---|
| Rank IC | **0.0336** |
| t-stat (Newey-West, autocorrelation-corrected) | 4.78 |
| t-stat (deflated for ~26 configs tried) | 2.80 |
| AUC | 0.5201 |
| Ranking horizon | 3 sessions |
| **Recommended holding period** | **30 sessions** |
| Universe | 48 NSE stocks |
| Training rows | 65,853 |
| Features | 100 |

**Ranking horizon and holding period are different numbers.** The model ranks
on a 3-day target because that is where signal peaks, but a 3-day *hold*
rebalances ~84x/year and measured **−1.9% net** of 10bps costs. A 30-day hold
measured **+126.6% net**, with win rate rising 56.1% → 65.3% and max drawdown
falling −16.1% → −8.7%. Turnover, not signal, was the binding constraint.

**How much of this is alpha?** Hedged against the market, very little: a
dollar-neutral long−short book returns +9.5% over four years at **t = 0.57**,
i.e. indistinguishable from zero. The long-only strategy beats equal-weight
buy-and-hold by roughly **6–8 percentage points per half** — a real but modest
tilt, not market-neutral alpha. Treat this as a long-only strategy that mildly
outperforms its benchmark, not as an alpha engine.

The previous model, measured the same way, had **no signal at all**: 32.6%
accuracy against a 38.4% majority-class baseline, and 51.0% directional
accuracy. Its reported "62%" was never reproducible.

Regenerate with `python -m training.train`; the report is written to
`results/metrics/validation_report.json` and served at
`GET /api/model/performance`.

**The signal is real but small.** That is what a genuine edge in liquid
large-cap equities looks like. Any daily equity model claiming 60%+
directional accuracy is measuring something other than what it thinks — as
the previous version of this one was.

## Features

- **Cross-sectional ML ranking** — LightGBM ensemble over 100 features
- **SHAP explanations** — every score traces to named features and their
  actual values, not hand-written rules
- **NSE microstructure** — delivery %, trade counts and turnover from
  bhavcopy, which is where much of the India-specific signal lives
- **Anomaly detection** — 4-model unsupervised consensus (Isolation Forest,
  One-Class SVM, Autoencoder, LOF)
- **Live quotes** — NSE prices via Yahoo Finance
- **Terminal UI** — dark, keyboard-driven, Bloomberg-style

## Quick start

```bash
pip install -r requirements.txt
```

Build the data panel and train the model (first run only, ~5 minutes):

```bash
python -m data.build_panel
python -m training.train
```

Launch the terminal:

```bash
python api.py
```

Opens at http://localhost:8001. On Windows, `ATLAS.bat` does all of the above.

## Architecture

```
data/build_panel.py        joins yfinance OHLCV with NSE bhavcopy microstructure
features/predictive.py     100 features: momentum, volatility, trend, delivery,
                           and cross-sectional rank/z-score against the universe
training/train.py          purged walk-forward validation, then final fit
reasoning/engine.py        SHAP attribution -> human-readable explanations
models/predictor.py        scores the whole universe (a relative score has no
                           meaning for one stock alone)
api.py                     FastAPI backend
web/                       frontend
```

### Why the universe is scored all at once

The target is cross-sectional, so a score only means something relative to
the other stocks scored on the same day. `predict_universe()` therefore takes
the whole panel and ranks within a date. Scoring a single stock in isolation
would produce a number with no defined meaning.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/universe` | all stocks with scores, labels, reasons, live prices |
| `GET /api/prediction/{ticker}` | score plus ranked SHAP signals behind it |
| `GET /api/model/performance` | out-of-sample validation metrics |
| `GET /api/anomalies/{ticker}` | anomaly detail |
| `GET /api/history/{ticker}` | OHLCV history for charts |
| `POST /api/refresh` | force quote refresh |
| `POST /api/predictions/refresh` | recompute scores after new daily data |

## Tech stack

- **Backend**: FastAPI, Python 3.10+
- **ML**: LightGBM, scikit-learn, SHAP
- **Data**: pandas, yfinance, NSE bhavcopy
- **Frontend**: Vanilla JS, Chart.js

## Known limitations

Stated plainly, because they affect how the output should be read:

- **No survivorship correction.** The universe is 48 currently-listed stocks,
  so historical results are optimistic to the extent delisted names are absent.
- **It is not market-neutral.** Returns are dominated by market exposure; a
  10/10 score does not protect you in a falling market. In the flat half of
  the test period the long-only book returned +9.6% against the market's
  +3.4%, and in down markets the score's edge over the *worst*-rated stocks
  was under 0.1%.
- **Transaction costs matter more than the signal.** A daily-rebalanced
  long-short book is gross-profitable and net-negative past ~5bps. The score
  is a ranking signal, not a trading strategy.
- **The anomaly pipeline must be run separately.** Until `main.py` writes
  `results/anomaly_results.csv`, the terminal reports no anomalies.
- **Predictions are as-of the last complete session**, not intraday. Live
  prices update every 60s; scores update when new daily data lands.

## Documentation

- [docs/MODEL_VALIDATION.md](docs/MODEL_VALIDATION.md) — methodology, measured
  results, the backtest, and the approaches that did not work

---

**Version**: 3.0 | **Last updated**: Aug 2026
