# Model Validation

How the ATLAS prediction model was built and measured, including the things
that did not work. Every number here comes from purged walk-forward
validation on held-out data — none of it is in-sample.

Reproduce with:

```bash
python -m training.train
```

---

## 1. What the previous model actually did

The README claimed ~62% directional accuracy. Measured against the model
that was actually being served:

| Metric | Value |
|---|---|
| Test accuracy (5-class) | **32.6%** |
| Majority-class baseline | **38.4%** |
| Directional accuracy | **51.0%** |
| Directional accuracy, non-HOLD calls | **50.0%** |
| Base rate of up days | 47.2% |

The model was **worse than always predicting HOLD**, and its directional
calls were indistinguishable from a coin flip.

Three root causes:

1. **The train/test split was not temporal.** `train_simple.py` concatenated
   INFY, SBIN and TCS and then took a chronological 80/20 slice of the
   *concatenated* frame. The test set was 100% TCS; the model was never
   evaluated on a held-out time period at all.
2. **Delivery % was hardcoded to 0.5 at inference.** The feature was trained
   on real values but served a constant, so the single most India-specific
   input was dead in production.
3. **Trained on 3 stocks, applied to 48.**

Separately, 7 of the 44 features in `features/prediction_features.py`
evaluated to **entirely NaN** — the ADX family and the relative-strength
family. `pd.Series(numpy_array)` inside those functions built a `RangeIndex`
that could not align with the frame's `DatetimeIndex`, so every arithmetic
result was NaN. Nothing failed loudly; the columns just silently carried no
information.

## 2. Choosing what to predict

Absolute direction ("will this stock rise tomorrow") is close to
unpredictable for liquid large caps. Cross-sectional rank ("will this stock
beat its peers") is measurably easier. Both were tested.

Two metrics are used throughout:

- **rank IC** — Spearman correlation between predicted score and realised
  forward return, computed *within each date* and averaged. This is the
  standard measure of signal quality for a ranking model. IC 0.02–0.04 is a
  real signal in daily equities; 0.05+ is strong.
- **within-date AUC** — ranking accuracy on a single day. Pooled AUC across
  dates can look good purely by timing the market; within-date AUC cannot.

### A trap worth documenting: sector-neutral targets

A sector-neutral target ("beat your own sector today") initially scored
**0.5886 within-date AUC**, far above everything else. It was an artifact.

A baseline using **no features at all** — just each stock's historical rate
of beating its sector — scored **0.6026**, i.e. *better* than the 100-feature
model. Six of the 15 sectors contained two stocks or fewer, so "beat your
sector" collapsed into a fixed pairwise comparison. The model was learning
**which stock it was looking at**, not what would happen next. Its long-short
payoff was correspondingly ~0%.

Rejected.

## 3. Horizon

Signal decays fast, and past ~7 days there is nothing left:

| Horizon | rank IC | t-stat |
|---|---|---|
| 3 days | **0.0283** | **5.45** |
| 5 days | 0.0209 | 3.85 |
| 7 days | 0.0177 | 3.31 |
| 10 days | −0.0007 | −0.13 |
| 20 days | −0.0051 | −1.03 |

A t-stat above 2 means the signal is statistically distinguishable from
zero. At 10 and 20 days it is not.

## 4. What did not work

Recorded because negative results are the expensive part:

| Idea | rank IC | vs baseline |
|---|---|---|
| Baseline (raw features, binary tail target) | 0.0209 | — |
| Cross-sectional normalisation of all features | 0.0134 | **worse** |
| Continuous rank target instead of binary | 0.0139 | worse |
| Feature selection to top 40 by stability | 0.0109 | **worse** |
| XGBoost + LightGBM ensemble | 0.0135 | worse |
| Raw + normalised features together | 0.0183 | worse |

Cross-sectional normalisation is standard quant practice and was expected to
help. It did not: ranking away the raw levels destroys information that
matters here — absolute volatility, absolute delivery percentage, and where
a price sits in its own 52-week range are informative as *levels*, not just
as ranks.

Feature selection hurting is the same story in a different form. With ~66k
rows and a low signal-to-noise ratio, the marginal features are not noise the
model needs protecting from; the regularisation already handles them.

### Widening the universe

The bhavcopy carries 104 symbols, not just the 48 with yfinance OHLCV files.
Rebuilding the panel from bhavcopy prices gave 100 usable tickers and 143,122
rows — more than double the data, and a cross-section twice as wide each day.
For a ranking model that should help twice over.

It did not: **rank IC 0.0258 (t = 5.40)** against 0.0336 on the 48-stock
panel. Scored only on the original 48 tickers — the like-for-like comparison
— the wide-trained model still came in at **0.0258 (t = 4.52)**, so this is
not an artifact of grading it on harder names. Three plausible reasons, not
separated here:

- The extra names are less liquid and behave differently.
- They have no sector mapping, so all 52 landed in one "Other" bucket and
  degraded the sector-relative features.
- Bhavcopy prices are unadjusted. The panel reconstructs an adjusted series
  from `CLOSE_PRICE / PREV_CLOSE`, which is correct in principle but noisier
  than yfinance's adjusted feed.

Kept at 48 stocks. Worth revisiting with a proper sector map for the wider
list, which is the cheapest of the three fixes.

## 5. What did work

| Idea | effect |
|---|---|
| Shorter horizon (3d over 5d) | IC 0.0209 → 0.0283 |
| Shallower, more regularised trees | IC 0.0209 → 0.0225 |
| Averaging targets at several tail cutoffs | IC 0.0209 → 0.0221 |
| Multi-seed averaging | small, reliable |

The winning ideas all reduce variance rather than adding representational
power — the usual outcome when signal-to-noise is this low.

Combining them, measured against each variant's own horizon:

| Variant | rank IC | t-stat | AUC | L-S gross | Sharpe |
|---|---|---|---|---|---|
| h=3, single cutoff | 0.0310 | 5.87 | 0.5178 | 20.8% | 1.22 |
| **h=3, two cutoffs (shipped)** | **0.0336** | **6.37** | **0.5201** | **25.9%** | **1.46** |
| h=3, multi-horizon | 0.0310 | 5.68 | 0.5180 | 26.5% | 1.53 |
| h=3, multi-horizon + cutoffs | 0.0318 | 5.86 | 0.5184 | 29.0% | 1.64 |
| h=5, multi-horizon + cutoffs | 0.0251 | 4.48 | 0.5174 | 18.5% | 0.93 |

The shipped configuration is **rank IC 0.0336, t = 6.37** — a 61% improvement
in signal over the 0.0209 starting point, and significant at any conventional
threshold. All six folds were positive (0.0138 to 0.0549).

Decile monotonicity — mean 3-day forward return by predicted decile:

| Decile | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Return | +0.074% | +0.117% | +0.154% | +0.191% | +0.278% | +0.181% | +0.236% | +0.235% | +0.295% | +0.349% |

The trend is upward but not strictly monotonic (D4 and D5 invert), which is
expected at this signal strength. Every decile is positive because the market
drifted up over the sample; the **spread** (D9 − D0 = 0.275% per 3 days) is
the part attributable to the model.

## 6. Transaction costs

Signal existing is not the same as signal being tradeable.

Daily rebalancing of a decile long-short book produced **32.4% gross**
annualised at a **1.60 Sharpe**, which collapsed to **−18%** at 5bps
round-trip and **−68%** at 10bps. Turnover ate everything.

Holding for the full horizon instead of rebalancing daily cuts turnover by a
factor of the horizon and is the only version that survives. Reported
net-of-cost figures use non-overlapping holds.

**The score is a ranking signal, not a trading strategy.** It says which
stocks the model prefers today relative to the rest of the universe. Turning
that into money requires cost control this repo does not attempt.

## 7. Backtest: what happens if you actually trade the score

Reproduce with `python -m evaluation.score_backtest`. Every number is
out-of-sample: 49,245 predictions across 1,026 trading days,
2022-06-01 to 2026-07-22.

### Buy a 10/10 stock, sell after N days

| Hold | Trades | Win rate | Avg return | Median | Worst trade |
|---|---|---|---|---|---|
| 1 day | 5,126 | **53.0%** | +0.134% | +0.096% | −9.7% |
| 2 days | 5,121 | 54.4% | +0.258% | +0.200% | −13.1% |
| 3 days | 5,116 | **55.0%** | +0.350% | +0.289% | −20.9% |
| 5 days | 5,106 | 56.4% | +0.532% | +0.493% | −26.7% |
| 10 days | 5,081 | 57.2% | +0.928% | +0.747% | −28.1% |

You lose money on **47% of 1-day trades**. The edge is an average, not a
guarantee, and single trades have lost up to 9.7% in a day.

### The score is monotonic — it does work

Win rate rises with score at a 1-day hold: 50.6% at score 1, 52.2% at 5,
53.0% at 10. Average return rises from +0.025% to +0.134%. The ordering is
real; the magnitude is small.

### The market dominates the outcome, not the score

This is the most important result in this document:

| 1-day hold | Trades | 10/10 avg | 10/10 win rate |
|---|---|---|---|
| Market rose that day | 3,001 | **+0.664%** | **68.1%** |
| Market fell that day | 2,125 | **−0.616%** | **31.7%** |

Whether a single 1-day trade makes money is overwhelmingly decided by the
market's direction that day. In down markets the 10/10 bucket beat the 1/10
bucket by only **+0.056%** — the model's entire edge, swamped by a market
move an order of magnitude larger.

The score also carries a mild beta tilt (1.044 vs 0.926 for the bottom
bucket), so some of the raw out-performance is amplified market exposure
rather than stock selection.

### Costs decide whether any of this is real

Average return per trade on 10/10 names, net of round-trip cost:

| Hold | 0 bps | 5 bps | 10 bps | 20 bps |
|---|---|---|---|---|
| 1 day | +0.134% | +0.084% | +0.034% | **−0.066%** |
| 3 days | +0.350% | +0.300% | +0.250% | +0.150% |
| 10 days | +0.928% | +0.878% | +0.828% | +0.728% |

At the portfolio level, holding every 10/10 name:

| Rebalance | Gross cumulative | Market | Net @10bps |
|---|---|---|---|
| Every 3 days | **+283.0%** | +97.9% | **−1.9%** |
| Every day | +270.0% | +97.2% | **−93.9%** |

The gross numbers look spectacular and are almost entirely consumed by
turnover. Longer holds are the only version that survives.

Portfolio detail (3-day rebalance, ~4.1 years): 39.2% annualised gross vs
18.3% for the equal-weight universe, 61.0% of periods profitable, max
drawdown −16.1%. By quarter: profitable in 15 of 18, beat the market in 13
of 18. Worst quarter was 2026Q1 at −11.0%, when the market fell 11.6%.

## 8. Methodology

- **Purged walk-forward.** Six expanding-window folds. The last `horizon`
  days of every training window are dropped, because those rows' forward
  returns overlap the test period and would leak the answer.
- **Cross-sectional target.** Per day, the top and bottom tails of the
  forward-return distribution are labelled 1 and 0; the ambiguous middle is
  dropped from training.
- **No survivorship correction.** The universe is 48 currently-listed NSE
  stocks, so results are optimistic to the extent that delisted names are
  absent. Worth stating plainly rather than hiding.
- **Scores are relative.** A score of 10 means "top of this universe today",
  not "will go up 10%".

## 9. Honest summary

The signal is **real and statistically significant** (t-stat > 5) but
**small**, which is what a genuine edge in liquid large-cap equities looks
like. Anyone reporting 60%+ directional accuracy on daily equity prediction
is measuring something other than what they think.
