# Experiment Archive

**Read this before trying to improve the ATLAS prediction model.** Every
entry below is a technique that was actually implemented and measured
against the shipped configuration on identical purged walk-forward folds —
not a guess about what might work. Nineteen of twenty-one were rejected.
Re-running a rejected idea without reading why it failed here is the single
most likely way to waste a session on this project.

**The shipped configuration**, unchanged by everything in this folder:
cross-sectional ranking model, 3-day ranking horizon, LightGBM ensemble
(2 tail quantiles × 3 seeds), **30-day holding period**, long-only, top
decile. Rank IC 0.0336 (Newey-West t = 4.78, deflated t = 2.80 for ~26
configurations tried). Full methodology in
[`docs/MODEL_VALIDATION.md`](../docs/MODEL_VALIDATION.md) — this folder is
the *evidence*, that file is the *narrative*.

Every script here is runnable from the repo root:

```bash
python experiments/07_holding_period_sweep.py
```

Each requires `results/models/prediction/walk_forward_predictions.csv` to
exist (`python -m training.train` produces it). Several take multiple
minutes — each fits dozens to hundreds of LightGBM models.

## How to use this before starting new work

1. Skim the **verdict column** below for anything close to your idea.
2. If it's REJECTED, read that entry's "why" before trying a variant. Most
   rejections here have a specific, non-obvious mechanism (below) — a
   superficially different idea often fails for the *same* mechanism.
3. If you find a genuinely new angle, add a numbered script here in the same
   format, and add a row to the table. Keep negative results — they're the
   expensive ones and the ones most likely to be silently retried later.

## Index

| # | Script | Question | Verdict | Root cause |
|---|---|---|---|---|
| — | *(no code preserved — see [`docs/MODEL_VALIDATION.md`](../docs/MODEL_VALIDATION.md) §2–§5)* | Cross-sectional feature normalisation | REJECTED (IC 0.0209→0.0134) | Ranking away raw levels destroys information that matters as a *level* (absolute vol, absolute delivery %) |
| — | *(same, §5)* | Feature selection to top 40 by stability | REJECTED (0.0209→0.0109) | At this sample size, marginal features aren't noise — regularisation already handles them; removing them removes signal |
| — | *(same, §4)* | Sector-neutral prediction target | REJECTED — **overfitting trap** | A zero-feature baseline (each stock's historical win-rate) beat the "model" (0.6026 vs 0.5886 AUC); 6 of 15 sectors had ≤2 stocks, so the target degenerated into memorising stock identity |
| — | *(same, §3)* | Prediction horizon: 5d / 7d / 10d / 20d vs 3d | REJECTED for all — signal vanishes past ~7 days (10d: t=−0.13) | 3-day cross-sectional signal decays fast; not investigated further here since horizon is a settled input, not a candidate to retry |
| — | *(same, §4)* | Widen universe 48→100 stocks via bhavcopy | REJECTED (0.0336→0.0258, confirmed like-for-like) | Extra names less liquid, no sector map (all fell into "Other"), unadjusted bhavcopy prices noisier than yfinance's adjusted feed |
| 01 | [`01_sample_weighting_winsorization_diversifier.py`](01_sample_weighting_winsorization_diversifier.py) | Sample-uniqueness weighting, feature winsorization, ElasticNet diversifier | REJECTED, all three (0.0336→0.0320/0.0323/0.0325) | Uniqueness weighting discards info with only 48 stocks; winsorization solves a problem tree splits don't have; linear models can't capture the nonlinear interactions the signal lives in |
| 02 | [`02_beta_sector_neutralization_leak.py`](02_beta_sector_neutralization_leak.py) | Does beta/sector-neutralizing the score help? | **IMPLEMENTED, SHIPPED, THEN REVERTED** — see below | Non-causal experiment design (pooled fit across the whole test period) measured +37%; the only causally-valid version (fixed coefficients from training data) measured *worse* than doing nothing (0.0237–0.0311 vs 0.0336) |
| 03 | [`03_triple_barrier_anomaly_meta_labeling.py`](03_triple_barrier_anomaly_meta_labeling.py) | Triple-barrier labels, anomaly-detector features, meta-labeling | REJECTED / neutral (0.0142, 0.0337, 0.0340 vs 0.0337) | Triple-barrier suits path-dependent problems, not ranking; the 4 unsupervised anomaly detectors add literally nothing the prediction features don't already carry; meta-labeling's 0.9% lift is fold-to-fold noise |
| 04 | [`04_regime_filtering.py`](04_regime_filtering.py) | Trade only in favourable market regimes | REJECTED — **opportunity-cost trap** | Raises per-trade win rate but more than halves total return (+283%→+130%) once idle-capital periods are counted; no advance-knowable regime signal correlated with next-day IC above \|ρ\|≈0.14 anyway |
| 05 | [`05_more_history_vs_delivery_features.py`](05_more_history_vs_delivery_features.py) | Is the ceiling the model or the data? (2015 vs 2021 history; delivery features ablated) | Doubling history: NEUTRAL (0.0277→0.0276). Removing delivery features: **−21% of total signal** | Model saturates on price/volume history; NSE delivery/microstructure data is the real value driver — informative finding, not a rejection |
| 06 | [`06_conviction_filters_accuracy.py`](06_conviction_filters_accuracy.py) | Do stricter conviction filters raise accuracy? Where does "accuracy" even come from? | Filter REJECTED at portfolio level; **diagnostic finding**: top-decile win rate tracks the market's own up-day rate almost exactly | A delivery+volume filter's per-trade lift replicated out-of-sample (+2.6%→+4.1%) but the resulting portfolio underperformed unfiltered (concentration → variance cost > hit-rate gain) |
| 07 | [`07_holding_period_sweep.py`](07_holding_period_sweep.py) | What holding period actually makes money net of costs? | **ADOPTED — the single best change found in this project** | Short holds are best gross, worst net (1d: +270% gross → **−93.9% net**); turnover, not signal, was the binding constraint all along. 30d hold: net −1.9%→**+126.6%**, win rate 56.1%→65.3% |
| 08 | [`08_portfolio_construction.py`](08_portfolio_construction.py) | Overlapping tranches, inverse-vol weighting, signal-decay exit | Overlapping tranches: robustness CONFIRMED (not luck on one entry grid). Inverse-vol: neutral. Decay-exit: REJECTED (turnover) | Overlapping tranches also revealed the single-grid drawdown (−14.6%) was partly lucky — true figure is −24.1%; decay-exit's adaptive holding drove turnover to 15.2%/day |
| 09 | [`09_market_neutral_hedging.py`](09_market_neutral_hedging.py) | How much of the return is genuine alpha vs market beta? | **Most important finding in the project** — dollar-neutral long-short is statistically zero (t=0.57) | Long-only book beats market by only ~6–8pp per regime-half; the +117% headline is ~90% a rising market, not stock-picking |
| 10 | [`10_sector_relative_pairs.py`](10_sector_relative_pairs.py) | Does a narrower, sector-relative hedge recover the alpha the market-wide hedge missed? | REJECTED — worse than the hedge it was meant to improve (t=−0.31 vs t=0.57) | Confirms rather than contradicts #09: there is no stock-selection edge here strong enough to survive removing market exposure, at any granularity tried |

## The three lessons worth internalising before writing new code

These recur across multiple independent experiments above, which is why
they're pulled out here instead of left buried in individual scripts.

### 1. A per-trade or per-day metric will recommend a worse portfolio

Happened twice, in unrelated experiments (#04 regime filtering, #06
conviction filters). Both raised win rate. Both destroyed total return once
turnover/opportunity-cost was priced in. **Always evaluate a candidate change
as a full backtested portfolio, never as an isolated per-signal statistic.**

### 2. An offline experiment can leak even when the walk-forward split is clean

Experiment #02 is the sharpest example in this codebase: the *fold
boundaries* were correct (no test-period date ever appeared in training),
but the neutralization regression was fit by *pooling every date within a
test fold into one regression*, which lets a given day's score-adjustment
implicitly depend on the composition of other days in that same window — a
window that, from that day's point of view, is partly in the future. This
is invisible unless you ask "could a predictor scoring one day at a time,
with no knowledge of subsequent days, actually reproduce this exact number?"
**Before trusting a walk-forward result, ask that question about every
post-processing step, not just the model fit.**

The failure mode this produces is not subtle once caught: production scored
**0.0237, below the 0.0336 baseline**, despite the offline experiment showing
+37%. It was caught only by running the actual training pipeline and noticing
the number went down — not by reviewing the experiment code, which looked
correct in isolation.

### 3. Multiple testing is not a footnote

Roughly 26 distinct configurations have now been compared across this
project's history (§2 of `docs/MODEL_VALIDATION.md` and this table
combined). The shipped model's naive t-stat (6.37) overstated significance
by autocorrelation alone (corrected: 4.78); after additionally deflating for
having been selected as the best of ~26 trials, it's t = 2.80 — still real,
but a different number than the one a single experiment would report in
isolation. **Any new result should be evaluated against the deflated
figure, not the raw one, and should increment the trial count in
`training/train.py::N_TRIALS_SEARCHED`.**
