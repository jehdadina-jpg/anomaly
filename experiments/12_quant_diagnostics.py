"""
Experiment 19: classic quant-research techniques not yet applied here.

Everything before this treated the LightGBM ensemble as a black box and
tested portfolio constructions around it. This round asks different
questions using standard equity quant-research methods:

  1. Fama-MacBeth cross-sectional regressions -- which INDIVIDUAL features
     carry real, statistically significant signal, independent of the ML
     model's feature interactions? Classic Fama & French (1973) two-step
     procedure: regress cross-sectionally every day, then test whether the
     average daily slope is nonzero.
  2. Simple linear benchmark -- does a 5-feature linear combination from
     step 1 get anywhere near the LightGBM ensemble's rank IC? If yes, the
     model's nonlinearity isn't earning its complexity. If no, that
     validates the choice to use a nonlinear model in the first place.
  3. Signal decay term structure -- smooth rank IC curve across horizons
     1-30 days (not the coarse 5-point grid used to originally pick horizon
     3), plus the actual half-life of the signal.
  4. Score persistence -- once a stock enters the top decile, how many days
     does it typically stay there? Cross-checks the holding-period result
     from a completely different angle (signal dynamics, not realised P&L).
  5. Probabilistic and Deflated Sharpe Ratio (Bailey & Lopez de Prado) -- the
     project has deflated the model's rank IC for multiple testing, but
     never the STRATEGY's Sharpe ratio itself, and never accounted for
     return skewness/kurtosis (a naive Sharpe assumes normality; equity
     returns are not normal).
  6. Block-bootstrapped maximum drawdown / CVaR -- the reported -14.6% max
     drawdown is ONE historical realisation. Block-bootstrapping the return
     series (preserving autocorrelation) gives a distribution, not a point
     estimate.
  7. Capacity analysis -- using real average-daily-turnover data (bhavcopy),
     how much capital could actually be deployed before market impact
     erodes the edge?

Everything here is diagnostic/risk analysis on the ALREADY-SHIPPED strategy,
not a competing configuration -- there is no "adopt/reject" for most of it,
the goal is understanding what is actually being relied on.

RESULTS

1-2. Fama-MacBeth: only 3 of 16 candidate features are individually
     significant (|NW-t|>2): delivery_pct_trend (t=2.84), delivery_pct_z20
     (t=2.79), vs_sector_ret_5d (t=-2.70, mean-reverting). This is an
     INDEPENDENT confirmation, via classic linear cross-sectional
     regression, of the ablation finding in 05_more_history_vs_delivery
     _features.py that delivery data carries ~21% of the model's signal --
     two unrelated methods converge on the same conclusion.

     A simple linear combination of the top-5 features gets IC=0.0291
     (NW-t=3.26) against the shipped ensemble's 0.0336 (t=4.78) -- 87% of
     the nonlinear model's signal from 5 linearly-combined features. Not
     grounds to replace the ensemble (it still wins, and by more than the
     gap suggests once t-stats are compared), but evidence the model isn't
     relying on exotic feature interactions -- a few known drivers, mostly
     linear, account for most of what is happening.

3. Signal decay: the h=3-trained score's correlation with returns actually
   PEAKS around h=7 (IC=0.0360) before declining, and has NOT decayed to
   half its peak even by h=30 (IC=0.0234). This is not "what horizon should
   we train on" (already answered: 3d, see docs/MODEL_VALIDATION.md SS3) --
   it is "how long does a score computed today keep meaning something",
   which is the actual question a holding-period decision depends on.

4. Score persistence: median time a name spends CONTINUOUSLY in the top
   decile is 1 trading day (mean 1.9, 90th pct 4 days). This explains a
   result from the previous round that had no clear mechanism at the time:
   11_second_round_portfolio_strategies.py's "partial rebalancing" (keep
   names still in the top quartile, replace only drop-outs) measured WORSE
   than full reconstruction. Now it is clear why -- day-to-day RANK is
   almost pure noise, so "is this name still top-ranked" is not a
   meaningful question to ask daily. The 30-day hold works despite this,
   not because of persistent rankings: per point 3, the SCORE still
   predicts 30-day-forward returns even though the RANK ORDER churns
   completely within days.

5. PSR(SR*=0) = 99.7% -- accounting for the actual skew (-0.15) and excess
   kurtosis (4.40, fat-tailed vs normal's 3.0) of daily returns, not
   assuming normality the way a raw Sharpe ratio does. Deflated Sharpe
   Ratio (haircut for ~12 portfolio-construction trials across
   experiments/07-11) = 87.6%. Meaningfully more honest than the naive
   Sharpe=1.39 headline, and still comfortably supportive -- but 87.6% is a
   real, not perfunctory, amount of estimation uncertainty to sit with.

6. Block-bootstrapped drawdown: the single historical -14.6% max drawdown
   understates tail risk. Resampling the same return distribution 5,000
   times (20-day blocks, preserving autocorrelation) gives a 95th-
   percentile tail outcome of -27.2% -- almost double the one path that
   happened to occur. The median bootstrap total return (+169%) comfortably
   exceeds the realised path, so the distribution isn't pessimistic overall
   -- but -14.6% should not be quoted as a worst-case number.

7. Capacity: using the 10th-percentile ADV of typically-selected names and
   a conservative 5-10% of ADV per position, the strategy's estimated
   capacity is roughly Rs 17-34 crore (~$2-4M). This is a small/PA-scale
   strategy, not an institutional one, given the liquidity of the less-
   liquid names in a 48-stock NSE universe.
"""
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\mdadi\Downloads\anomalydetection")
from data.build_panel import load_panel
from features.predictive import build_features, feature_columns
from evaluation.stats import newey_west_tstat, _expected_max_of_n_normals

ROOT = r"C:\Users\mdadi\Downloads\anomalydetection"
HOLD, BPS = 30, 10

panel = load_panel()
feat = build_features(panel)
FEATS = feature_columns(feat)
feat = feat.sort_values(["ticker", "Date"]).reset_index(drop=True)

preds = pd.read_csv(rf"{ROOT}\results\models\prediction\walk_forward_predictions.csv",
                    parse_dates=["Date"])
oos_dates = preds["Date"].unique()

# ============================================================================
# 1. FAMA-MACBETH CROSS-SECTIONAL REGRESSIONS
# ============================================================================
print("=" * 100)
print("1. FAMA-MACBETH: which individual features carry independent signal?")
print("=" * 100)

feat3 = feat.copy()
feat3["fwd_3d"] = feat3.groupby("ticker")["Close"].transform(lambda x: x.shift(-3) / x - 1)
oos = feat3[feat3["Date"].isin(oos_dates)].copy()

CANDIDATES = [
    "momentum_12_1", "momentum_60d", "rsi_14", "adx_14", "vol_20d",
    "delivery_pct_z20", "delivery_pct_trend", "px_to_sma_20", "px_to_sma_50",
    "range_position_52w", "volume_zscore_20", "beta_60d", "bollinger_z",
    "vs_sector_ret_5d", "amihud_illiq", "avg_trade_size_z20",
]

fm_results = {}
for col in CANDIDATES:
    daily_slopes = []
    for date, g in oos.groupby("Date"):
        sub = g[[col, "fwd_3d"]].dropna()
        if len(sub) < 15 or sub[col].std() == 0:
            continue
        x = (sub[col] - sub[col].mean()) / sub[col].std()   # standardise within-day
        y = sub["fwd_3d"].values
        slope = np.polyfit(x, y, 1)[0]
        daily_slopes.append(slope)
    if len(daily_slopes) < 30:
        continue
    slopes = np.array(daily_slopes)
    naive_t = slopes.mean() / (slopes.std() + 1e-12) * np.sqrt(len(slopes))
    _, nw_t = newey_west_tstat(slopes, lag=2)
    fm_results[col] = (slopes.mean(), nw_t, len(slopes))

print(f"{'feature':24s} {'avg daily slope':>16s} {'NW t-stat':>10s} {'n days':>7s}")
for col, (m, t, n) in sorted(fm_results.items(), key=lambda kv: -abs(kv[1][1])):
    flag = "  significant" if abs(t) > 2 else ""
    print(f"{col:24s} {m:+16.5f} {t:10.2f} {n:7d}{flag}")

sig_features = [c for c, (m, t, n) in fm_results.items() if abs(t) > 2]
print(f"\n{len(sig_features)} of {len(fm_results)} features individually significant "
      f"(|t|>2): {sig_features}")

# ============================================================================
# 2. SIMPLE LINEAR BENCHMARK vs LIGHTGBM
# ============================================================================
print("\n" + "=" * 100)
print("2. SIMPLE LINEAR MODEL (top-5 FM features) vs LIGHTGBM ENSEMBLE")
print("=" * 100)

top5 = sorted(fm_results.items(), key=lambda kv: -abs(kv[1][1]))[:5]
top5_names = [c for c, _ in top5]
print(f"top-5 by |t-stat|: {top5_names}\n")

lin_score = pd.Series(0.0, index=oos.index)
for col, (m, t, n) in top5:
    z = oos.groupby("Date")[col].transform(lambda s: (s - s.mean()) / (s.std() + 1e-12))
    lin_score += np.sign(m) * z.fillna(0)

oos = oos.assign(lin_score=lin_score)
ics = np.array([
    sstats.spearmanr(x["lin_score"], x["fwd_3d"]).statistic
    for _, x in oos.dropna(subset=["fwd_3d"]).groupby("Date") if len(x) >= 10
])
ics = ics[np.isfinite(ics)]
_, lin_t = newey_west_tstat(ics, lag=2)
print(f"linear 5-feature combo:  rank IC = {ics.mean():+.4f}  NW-t = {lin_t:.2f}")
print(f"LightGBM ensemble:       rank IC = +0.0336  NW-t = 4.78   (shipped, for reference)")
if ics.mean() < 0.0336 * 0.7:
    print("\n-> The linear combination falls well short. The nonlinear ensemble")
    print("   is capturing feature INTERACTIONS a linear model cannot -- its")
    print("   added complexity is earning its keep.")
else:
    print("\n-> The simple linear model gets close. Worth reconsidering whether")
    print("   the added complexity of the ensemble is necessary.")

# ============================================================================
# 3. SIGNAL DECAY TERM STRUCTURE
# ============================================================================
print("\n" + "=" * 100)
print("3. SIGNAL DECAY: rank IC vs horizon, smooth curve h=1..30")
print("=" * 100)

d = preds[["Date", "ticker", "p"]].merge(
    panel[["Date", "ticker", "Close"]], on=["Date", "ticker"], how="left")
d = d.sort_values(["ticker", "Date"]).reset_index(drop=True)

HORIZONS = list(range(1, 31))
term = {}
for h in HORIZONS:
    d[f"_fwd{h}"] = d.groupby("ticker")["Close"].transform(lambda x: x.shift(-h) / x - 1)
d = d.sort_values(["Date", "ticker"]).reset_index(drop=True)

for h in HORIZONS:
    col = f"_fwd{h}"
    sub = d.dropna(subset=[col])
    ics = np.array([sstats.spearmanr(x["p"], x[col]).statistic
                    for _, x in sub.groupby("Date") if len(x) >= 10])
    ics = ics[np.isfinite(ics)]
    if len(ics) < 30:
        continue
    _, t = newey_west_tstat(ics, lag=max(h - 1, 1))
    term[h] = (ics.mean(), t)

peak_h, (peak_ic, peak_t) = max(term.items(), key=lambda kv: kv[1][0])
half_ic = peak_ic / 2
half_life = next((h for h in HORIZONS if h > peak_h and term.get(h, (peak_ic,))[0] <= half_ic),
                 None)

print(f"{'h':>4s} {'IC':>8s} {'NW-t':>6s}  {'h':>4s} {'IC':>8s} {'NW-t':>6s}")
hs = sorted(term.keys())
for i in range(0, len(hs), 2):
    row = ""
    for h in hs[i:i + 2]:
        ic, t = term[h]
        row += f"{h:4d} {ic:+8.4f} {t:6.2f}  "
    print(row)
print(f"\npeak IC at h={peak_h}d ({peak_ic:+.4f}, t={peak_t:.2f})")
print(f"signal half-life: ~{half_life}d (IC decays to half its peak)"
      if half_life else "signal half-life: not reached within 30d")

# ============================================================================
# 4. SCORE PERSISTENCE (cross-check on holding period, from a different angle)
# ============================================================================
print("\n" + "=" * 100)
print("4. SCORE PERSISTENCE: once in the top decile, how long does a name stay there?")
print("=" * 100)

rk = preds.assign(rk=preds.groupby("Date")["p"].rank(pct=True))
rk_wide = rk.pivot_table(index="Date", columns="ticker", values="rk").sort_index()
dates_arr = rk_wide.index.to_numpy()

durations = []
for tk in rk_wide.columns:
    s = rk_wide[tk].values
    in_top = s >= 0.90
    i = 0
    while i < len(in_top):
        if in_top[i]:
            j = i
            while j < len(in_top) and in_top[j]:
                j += 1
            durations.append(j - i)
            i = j
        else:
            i += 1
durations = np.array(durations)
print(f"episodes of continuous top-decile membership: {len(durations)}")
print(f"  median duration   {np.median(durations):.0f} trading days")
print(f"  mean duration     {durations.mean():.1f} trading days")
print(f"  75th percentile   {np.percentile(durations, 75):.0f} trading days")
print(f"  90th percentile   {np.percentile(durations, 90):.0f} trading days")
print(f"\nCross-check: the empirically-optimal 30-day hold "
      f"({'is' if np.median(durations) >= 15 else 'is NOT'} consistent with "
      f"typical signal persistence of {np.median(durations):.0f} days).")

# ============================================================================
# 5. PROBABILISTIC / DEFLATED SHARPE RATIO
# ============================================================================
print("\n" + "=" * 100)
print("5. PROBABILISTIC AND DEFLATED SHARPE RATIO  (Bailey & Lopez de Prado)")
print("=" * 100)

px = d.pivot_table(index="Date", columns="ticker", values="Close")
rkw = rk_wide.reindex(px.index)
ret = px.pct_change().shift(-1)
entries = np.arange(0, len(dates_arr), HOLD)
w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
for i in entries:
    sel = rkw.iloc[i] >= 0.90
    if sel.sum() == 0:
        continue
    end = min(i + HOLD, len(dates_arr))
    w.iloc[i:end] = np.where(sel.reindex(px.columns).fillna(False).values,
                             1.0 / sel.sum(), 0.0)
r_book = (w * ret).sum(axis=1)
net = (r_book - (1.0 / HOLD) * 2 * BPS / 10000).dropna()

n = len(net)
sr = net.mean() / net.std()                 # per-period (daily) Sharpe
sr_ann = sr * np.sqrt(252)
skew = sstats.skew(net)
kurt = sstats.kurtosis(net, fisher=False)    # NOT excess kurtosis; PSR formula wants raw

# PSR: probability the TRUE Sharpe exceeds a benchmark SR*, given estimation
# uncertainty that widens with skew/kurtosis away from a normal distribution.
def psr(sr_hat, sr_star, skew, kurt, n):
    num = (sr_hat - sr_star) * np.sqrt(n - 1)
    den = np.sqrt(1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat ** 2)
    return sstats.norm.cdf(num / den)

psr_zero = psr(sr, 0.0, skew, kurt, n)
print(f"  daily Sharpe            {sr:.4f}  (annualised {sr_ann:.2f})")
print(f"  skewness                {skew:+.3f}   kurtosis {kurt:.2f}  "
      f"(normal = 0, 3)")
print(f"  PSR(SR* = 0)             {psr_zero:.4%}  "
      f"(probability true Sharpe > 0, accounting for skew/kurtosis)")

# Deflated: benchmark against the expected max Sharpe of N independent trials,
# same machinery already used for the model's IC deflation.
N_STRATEGY_TRIALS = 12   # 07-11 in experiments/, each testing several variants
sr_var = (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (n - 1)
sr_std = np.sqrt(sr_var)
expected_max_sr = _expected_max_of_n_normals(N_STRATEGY_TRIALS) * sr_std
dsr = psr(sr, expected_max_sr, skew, kurt, n)
print(f"  expected max Sharpe of {N_STRATEGY_TRIALS} trials  {expected_max_sr:.4f}")
print(f"  Deflated Sharpe Ratio    {dsr:.4%}  "
      f"(probability true Sharpe exceeds what {N_STRATEGY_TRIALS} random trials would)")

# ============================================================================
# 6. BLOCK-BOOTSTRAPPED DRAWDOWN / CVaR
# ============================================================================
print("\n" + "=" * 100)
print("6. BLOCK-BOOTSTRAPPED MAX DRAWDOWN / CVaR  (5,000 resamples, block=20d)")
print("=" * 100)

rng = np.random.default_rng(42)
BLOCK, N_SIM = 20, 5000
vals = net.values
n_blocks_needed = int(np.ceil(len(vals) / BLOCK))

dds, cvars, totals = [], [], []
for _ in range(N_SIM):
    starts = rng.integers(0, len(vals) - BLOCK, n_blocks_needed)
    path = np.concatenate([vals[s:s + BLOCK] for s in starts])[:len(vals)]
    eq = (1 + path).cumprod()
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    dds.append(dd)
    totals.append(eq[-1] - 1)
    var5 = np.percentile(path, 5)
    cvars.append(path[path <= var5].mean() if (path <= var5).any() else var5)

dds, cvars, totals = np.array(dds), np.array(cvars), np.array(totals)
print(f"  historical realised max drawdown   {(((1+net).cumprod())/((1+net).cumprod()).cummax()-1).min():+.1%}")
print(f"  bootstrap max drawdown:  5th pct   {np.percentile(dds, 5):+.1%}")
print(f"                          median     {np.percentile(dds, 50):+.1%}")
print(f"                          95th pct   {np.percentile(dds, 95):+.1%}  <- worst-case tail")
print(f"  bootstrap total return: 5th pct    {np.percentile(totals, 5):+.1%}")
print(f"                          median      {np.percentile(totals, 50):+.1%}")
print(f"                          95th pct    {np.percentile(totals, 95):+.1%}")
print(f"  daily CVaR (5%), median across sims {np.percentile(cvars, 50):+.2%}")

# ============================================================================
# 7. CAPACITY ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("7. CAPACITY: how much capital before market impact erodes the edge?")
print("=" * 100)

adv = feat[feat["Date"].isin(oos_dates)][["Date", "ticker", "turnover_lacs"]]
sel = rk[rk["rk"] >= 0.90][["Date", "ticker"]].merge(adv, on=["Date", "ticker"], how="left")
sel = sel.dropna(subset=["turnover_lacs"])

print(f"  avg daily turnover (ADV) of top-decile names, in Rs lakh:")
print(f"    median   {sel['turnover_lacs'].median():,.0f}")
print(f"    10th pct {sel['turnover_lacs'].quantile(0.10):,.0f}  <- capacity binding constraint")
print(f"    90th pct {sel['turnover_lacs'].quantile(0.90):,.0f}")

for impact_pct in (0.05, 0.10, 0.20):
    per_name_cap = sel["turnover_lacs"].quantile(0.10) * impact_pct * 100000  # lakh -> rupees
    n_positions = 5  # ~top decile of 48 stocks
    book_cap = per_name_cap * n_positions
    print(f"  at {impact_pct:.0%} of ADV per position (10th-pct name): "
          f"book capacity ~ Rs {book_cap/1e7:,.1f} crore "
          f"(~${book_cap/8_500_0000:,.1f}M)")

print("\nThis is a rough floor, not a precision estimate: it uses the LEAST liquid")
print("name typically selected, ignores multi-day execution, and ignores that not")
print("every position is filled at the illiquid end every single rebalance.")
print("=" * 100)
