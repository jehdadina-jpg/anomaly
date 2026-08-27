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

### A significance bug, found and fixed

Every t-stat reported earlier in this document (e.g. "t = 5.45" in §3) used
`t = mean(IC) / std(IC) * sqrt(n_days)`, which assumes each day's IC is an
independent draw. It is not: with an h-day forward return, day t's IC and day
t+1's IC are computed from windows that share h−1 days, so the daily IC
series is autocorrelated by construction — measured lag-1 autocorrelation is
**0.49** on the shipped model.

Recomputed with a Newey-West HAC standard error (lag = horizon − 1):

| | naive t | Newey-West t |
|---|---|---|
| Shipped model (h=3, IC 0.0336) | 6.37 | **4.78** |

The naive figure overstated significance by **1.33x**. 4.78 is still solidly
significant, so the conclusion doesn't change — but the earlier number was
wrong, not just imprecise, and every t-stat in this document from here on is
Newey-West.

A second correction: roughly 20 configurations were compared during model
selection (§3–§5 below). Reporting only the winning configuration's t-stat
has the same bias as p-hacking — the "best of many" is expected to look
better than its true out-of-sample edge. Applying a haircut equal to the
expected maximum of 20 independent standard-normal draws (1.87, via direct
Monte Carlo — a first attempt using a closed-form Gumbel approximation was
wrong by 3-4x and is why this was simulated rather than trusted from a
formula):

| | t-stat |
|---|---|
| Newey-West | 4.78 |
| **Deflated (20 trials)** | **2.91** |

Still comfortably above the conventional t > 2 bar. This is a conservative
estimate — many of the 20 trials were correlated variants of each other
(horizon=3 and horizon=5 share most of the pipeline), so true selection bias
is likely less than this haircut assumes. `t-stat (naive)`,
`t-stat (Newey-West)`, and `t-stat (deflated)` are all reported by
`python -m training.train` and saved to `validation_report.json`, so this
isn't a one-time calculation — it's re-derived on every retrain.

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

## 8. Where the ceiling actually is

Short version: **the model is at a local optimum for this data.** Sixteen
distinct techniques have now been measured against the shipped configuration
on identical purged walk-forward folds. Every one was neutral or negative.

### Advanced techniques (all rejected)

| Technique | rank IC | vs baseline |
|---|---|---|
| Baseline (shipped) | **0.0337** | — |
| Triple-barrier labels (AFML ch.3) | 0.0142 | **much worse** |
| Anomaly-detector scores as features | 0.0337 | neutral |
| Meta-labeling (AFML ch.3) | 0.0340 | within noise |
| Anomaly + meta combined | 0.0339 | within noise |

Triple-barrier labelling — sizing profit-take/stop-loss barriers by each
stock's own volatility — is standard practice and was the most likely of
these to work. It halved the signal. The fixed-horizon cross-sectional
target the model already uses is better suited to a *ranking* problem than a
path-dependent one.

Meta-labeling (a second model predicting when the first is right) moved IC
from 0.0337 to 0.0340: a 0.9% change, far inside fold-to-fold noise. Not
adopted — an extra model doubling inference cost needs to earn more than that.

The anomaly detectors were the interesting case, since this repo already
ships four of them that the predictor never used. IsolationForest scores over
the feature panel added **exactly nothing** (0.0337 → 0.0337). Whatever the
anomaly models detect, the prediction features already capture.

### Regime filtering: a trap worth documenting

Out-of-sample IC is **0.0555 on days the market rose** and **0.0030 on days
it fell** — the signal essentially disappears in down markets. That looks
like an obvious opportunity: trade only in good regimes.

It is not, for two reasons.

First, that split uses *same-day* market return, which is not knowable at
decision time. Tested against regime signals that *are* knowable in advance
(trailing 20/60-day market return, trailing volatility, breadth, drawdown),
correlation with next-day IC was weak — the best was |ρ| ≈ 0.14.

Second, and more importantly, filtering **loses money** once idle capital is
counted:

| Strategy | Periods traded | Total return | Annualised | Win rate |
|---|---|---|---|---|
| **Always trade** | 341 | **+283.0%** | **+39.2%** | 61.0% |
| Only when 60d market return > median | 170 | +88.6% | +16.9% | 61.8% |
| Only when near market highs | 170 | +130.0% | +22.8% | 65.3% |

Filtering raises *per-trade* win rate (65.3% vs 61.0%) while cutting *total*
return by more than half, because it sits in cash for the skipped periods. A
per-trade quality metric that ignores opportunity cost will recommend this
filter; a total-return metric correctly rejects it.

### The real constraint is data, not algorithm

Since no modelling change helped, the binding constraint was tested directly.
The panel was rebuilt from 2015 instead of 2021 — 134,211 rows against
65,997, roughly double the history — and evaluated on the identical 2022-06+
test period:

| Configuration | Training rows | Features | rank IC |
|---|---|---|---|
| 2021+, all features (shipped) | 57,648 | 100 | **0.0334** |
| 2021+, delivery features removed | 57,648 | 86 | 0.0277 |
| 2015+, delivery features removed | 116,982 | 86 | 0.0276 |

Two clear conclusions:

1. **Doubling the training history changed nothing** (0.0276 vs 0.0277). The
   model saturates on price/volume history; it is not data-starved in that
   dimension.
2. **NSE delivery data is worth ~21% of the total signal** (0.0334 vs
   0.0277). Fourteen delivery/microstructure features contribute more than
   six extra years of price history.

So the path to a materially better model is **more microstructure data**, not
more history, more features, or a cleverer algorithm. Concretely, the highest-
value unexplored input is **F&O open interest**: `data/fetch_fo_data.py`
already exists to download it and `data/raw/fo/` is empty — it has never been
run. Open interest is the same category of information as delivery percentage
(who is actually positioned, versus who is merely trading) and is
particularly informative in Indian markets, where derivatives volume dwarfs
cash volume. Intraday bars (`data/fetch_intraday.py`, also never run) are the
second candidate.

## 9. A beta-neutralization attempt, and why it was reverted

§7 found the shipped model's top decile ran a beta of ~1.04 against ~0.93 for
the bottom decile — some of its apparent edge was amplified market exposure,
not stock selection. That, plus the Newey-West correction in §2, prompted a
second round of measurement: five candidate fixes, each tested against the
identical purged walk-forward folds so the comparison is apples-to-apples.

Reproduce the full comparison with `python -m evaluation.compare_configs`
(slow — fits ~200 models); see the shipped result alone, faster, via
`python -m training.train`.

| Addition | rank IC | Newey-West t | beta-tilt |
|---|---|---|---|
| Baseline (shipped in §5) | 0.0336 | 4.78 | +0.117 |
| + sample-uniqueness weighting (Lopez de Prado) | 0.0320 | 4.53 | +0.112 |
| + feature winsorization (1st/99th pct, per-day) | 0.0323 | 4.57 | +0.109 |
| **+ beta/sector neutralization** | **0.0459** | **7.21** | **−0.073** |
| + linear (ElasticNet) diversifier blended in | 0.0325 | 4.45 | +0.127 |
| All five combined | 0.0433 | 6.59 | −0.053 |

Three of five made things worse and were rejected:

- **Sample-uniqueness weighting** downweights rows whose forward-return
  windows overlap (standard practice for overlapping labels). With only 48
  stocks and a 3-day horizon, this just discards information without adding
  real diversity — theoretically well-motivated, empirically negative here.
- **Winsorization** protects against outliers distorting a fit. Tree splits
  are already robust to extreme values by construction (a threshold split
  isn't pulled toward outliers the way a mean or a linear coefficient is),
  so this was solving a problem the model didn't have.
- **A linear diversifier** (30% weight) blended into the ensemble measured
  worse alone, confirming linear models don't add much here — consistent
  with cross-sectional normalisation and feature selection also hurting
  in §4. The signal appears to live in nonlinear feature interactions that
  linear models can't capture at this sample size.

Beta/sector neutralization initially appeared to win big: rank IC 0.0336 →
**0.0459** (+37%), Newey-West t 4.78 → 7.21, with the beta-tilt closing from
+0.117 to −0.073. It was implemented and shipped.

**It was then caught and reverted.** Wiring it into production dropped rank
IC to **0.0237 — well below simply not neutralizing at all.** The cause was
a mismatch between how the experiment measured it and how a live predictor
must apply it:

| How the regression is fit | rank IC | Deployable? |
|---|---|---|
| No neutralization (baseline) | **0.0336** | yes |
| Per-date (~48 rows, ~17 params) | 0.0237 | yes, but worse |
| Coefficients fit on training panel, applied fixed | 0.0311 | yes, still worse |
| Pooled across the whole test period | 0.0459 | **no — uses future dates** |

Only the pooled-across-test-period variant beat the baseline, and it beats it
precisely *because* it pools: fitting one regression over ~170 test days lets
each day's residual borrow information about the composition of other days in
that window. A live predictor scoring one day at a time cannot reproduce
that without seeing the future. The two genuinely causal variants — per-date,
and fixed coefficients from training data — both underperform doing nothing.

The per-date version fails for a separate and more mundane reason: ~48 stocks
against beta plus ~15 sector dummies is barely more observations than
parameters, so the fitted residual is mostly noise.

**Shipped configuration is no neutralization.** `models/predictor.py` and
`training/train.py` both carry comments recording this, so it does not get
re-attempted. `evaluation/stats.py` retains
`fit_neutralization_coefs`/`apply_neutralization_coefs` for exploratory use,
documented as not-for-production.

The broader lesson, and the reason this section is kept rather than deleted:
**an offline experiment can be accidentally non-causal even when the
walk-forward split itself is clean.** The fold boundaries were correct; the
leak was in a post-processing step applied across the whole test block. The
check that caught it was simply running the real production pipeline and
noticing the number went *down*.

## 10. Methodology

- **Purged walk-forward.** Six expanding-window folds. The last `horizon`
  days of every training window are dropped, because those rows' forward
  returns overlap the test period and would leak the answer.
- **Cross-sectional target.** Per day, the top and bottom tails of the
  forward-return distribution are labelled 1 and 0; the ambiguous middle is
  dropped from training.
- **Autocorrelation-aware significance.** All t-stats use Newey-West HAC
  standard errors (§2), and the headline figure is additionally deflated for
  the number of configurations compared during development (§2, §8).
- **No beta/sector neutralization.** Tried and reverted (§9): every causally
  valid form of it measured worse than leaving scores alone. The score
  therefore carries a mild beta tilt, documented in §7.
- **No survivorship correction.** The universe is 48 currently-listed NSE
  stocks, so results are optimistic to the extent that delisted names are
  absent. Worth stating plainly rather than hiding.
- **Scores are relative.** A score of 10 means "top of this universe today",
  not "will go up 10%".

## 11. Honest summary

The signal is **real and statistically significant even after correcting for
autocorrelation and the number of configurations tried** (rank IC 0.0336,
Newey-West t = 4.78, deflated t = 2.80), but it remains **small**, which is what a genuine edge in liquid large-cap
equities looks like. Anyone reporting 60%+ directional accuracy on daily
equity prediction is measuring something other than what they think.
