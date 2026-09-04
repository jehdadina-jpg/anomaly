"""
Experiment 18: eight more portfolio-construction strategies.

Everything prior has established: the ranking signal is real but thin
(IC 0.0336), turnover is the dominant cost (holding period 3d->30d was the
single biggest win in the project), and the long-only book's return is
mostly market beta (hedging removes almost all of it, t=0.57).

This round deliberately avoids retrying anything that failed for a KNOWN
mechanism (see experiments/README.md "three lessons"):
  - no per-trade-only metrics (evaluate full portfolios, net of cost)
  - no binary regime in/out filtering (opportunity-cost trap, already killed)
  - no additional market-wide or sector hedge variant (already tested twice)

Eight genuinely different mechanisms, all built on the existing 30-day
long-only baseline:

  1. Volatility targeting     scale exposure (not binary in/out) so the book
                              runs constant risk instead of constant capital.
  2. Trend filter             exclude candidates below their 200d SMA even
                              if ML-ranked top decile -- a floor, not a filter
                              that shrinks the book to near-nothing.
  3. Partial rebalancing      keep names that stay in the top quartile
                              instead of dumping the whole book every 30
                              days -- attacks turnover directly, which is
                              the one lever already proven to matter.
  4. Stop-loss overlay        exit a single name early if it falls >10% from
                              entry, instead of riding the full 30-day hold.
  5. Rank persistence         require top-decile status at BOTH this
                              rebalance and the prior one before buying.
  6. Long-only/hedge blend    split capital between the long-only book and
                              the (already measured, t=0.57) dollar-neutral
                              hedge -- tests genuine diversification of two
                              return streams, not the hedge alone again.
  7. Rank momentum            select on IMPROVING rank (5-day change) rather
                              than absolute rank level.
  8. Sector cap               cap any single sector at 40% of the book.

All measured net of 10bps/leg, with Newey-West significance, split by the
same bull/flat regime halves used throughout this project.

RESULT: none robustly beat the baseline. Full-period Sharpe, baseline 1.39:

  1. vol target        1.41  <- best full-period number, but FAILS regime
                               split (bull 1.89->2.08 better, flat 0.96->0.76
                               worse) and worsens maxDD (-14.6%->-20.2%).
                               It levers up when trailing vol is low, which
                               is exactly the setup before a choppy period --
                               amplifies the market-beta the model already
                               rides (see 09_market_neutral_hedging.py) rather
                               than adding a new source of edge.
  2. trend filter       1.08  worse in both regimes
  3. partial rebalance  0.98  worse in both regimes -- the attempt to reduce
                               turnover backfired: checking every 10 days
                               with a top-quartile "keep" threshold churns
                               through marginal names about as often as full
                               reconstruction, without the benefit of buying
                               strictly the current top decile
  4. stop-loss          1.40  net roughly unchanged, but WORSE max drawdown
                               (-14.6%->-16.8%): cutting losers at -10% locks
                               in losses on names that would have mean-
                               reverted, a known failure mode of fixed stops
                               on a positive-drift universe
  5. rank persistence   0.92  worse in both regimes
  6. long/hedge blend   <1.4  monotonically worse as hedge weight rises --
                               the hedge (t=0.57, ~zero expected return) only
                               dilutes the working long-only book
  7. rank momentum      1.07  worse in both regimes
  8. sector cap         1.36  statistically indistinguishable from baseline
                               in both regimes; a legitimate, ~free
                               concentration-risk control, but not a
                               performance improvement

Verdict: REJECTED, all eight. Consistent with the project's established
finding (09_market_neutral_hedging.py) that the model's edge is real but too
thin to survive being amplified, filtered, or blended into something better
-- every attempt either reduces to "more market beta, timed differently" or
loses to added turnover/complexity. Shipped configuration unchanged.
"""
import sys, warnings
from pathlib import Path

import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\mdadi\Downloads\anomalydetection")
from data.build_panel import load_panel
from features.predictive import build_features
from evaluation.stats import newey_west_tstat

ROOT = r"C:\Users\mdadi\Downloads\anomalydetection"
HOLD, BPS = 30, 10

preds = pd.read_csv(rf"{ROOT}\results\models\prediction\walk_forward_predictions.csv",
                    parse_dates=["Date"])
feat = build_features(load_panel())[
    ["Date", "ticker", "Close", "vol_20d", "px_to_sma_200", "sector", "beta_60d"]]
d = preds[["Date", "ticker", "p"]].merge(feat, on=["Date", "ticker"], how="left")
d = d.sort_values(["Date", "ticker"]).reset_index(drop=True)
d["rk"] = d.groupby("Date")["p"].rank(pct=True)

px = d.pivot_table(index="Date", columns="ticker", values="Close")
rk = d.pivot_table(index="Date", columns="ticker", values="rk")
trend = d.pivot_table(index="Date", columns="ticker", values="px_to_sma_200")
vol = d.pivot_table(index="Date", columns="ticker", values="vol_20d")
sector_of = feat.drop_duplicates("ticker").set_index("ticker")["sector"].to_dict()

ret = px.pct_change().shift(-1)     # 1-day-ahead return, used to build any path
dates = px.index.to_numpy()
mid = dates[len(dates) // 2]
tickers = px.columns.tolist()
entries = np.arange(0, len(dates), HOLD)


def summarise(daily_ret: pd.Series, cost_per_day: float, label: str, subset=None):
    s = daily_ret.dropna()
    if subset is not None:
        s = s[subset.reindex(s.index).fillna(False)]
    if len(s) < 30:
        print(f"{label:38s}  (insufficient data)")
        return None
    net = s - cost_per_day
    yrs = len(net) / 252
    g = (1 + s).prod() - 1
    n = (1 + net).prod() - 1
    sh = net.mean() / (net.std() + 1e-12) * np.sqrt(252)
    eq = (1 + net).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    _, t = newey_west_tstat(net.values, lag=5)
    print(f"{label:38s} {g:+9.1%} {n:+10.1%} {(1 + n) ** (1 / yrs) - 1:+8.1%} "
          f"{sh:7.2f} {dd:+8.1%} {t:6.2f}")
    return dict(net=n, sharpe=sh, dd=dd, t=t)


def top_decile_weights():
    """Baseline single-tranche book, for reference and as an input to others."""
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for i in entries:
        sel = rk.iloc[i] >= 0.90
        if sel.sum() == 0:
            continue
        end = min(i + HOLD, len(dates))
        w.iloc[i:end] = np.where(sel.values, 1.0 / sel.sum(), 0.0)
    return w


w_base = top_decile_weights()
r_base = (w_base * ret).sum(axis=1)
cost_base = 1.0 / HOLD * 2 * BPS / 10000

results = {}
mkt = ret.mean(axis=1)

print("=" * 100)
print(f"REFERENCE  (single-tranche 30-day book already established)")
print("=" * 100)
print(f"{'strategy':38s} {'gross':>9s} {'net':>10s} {'ann.':>8s} {'Sharpe':>7s} "
      f"{'maxDD':>8s} {'NW-t':>6s}")
summarise(r_base, cost_base, "0. baseline (shipped)")
summarise(mkt, 0.0, "   market (no cost)")

# --------------------------------------------------------------- 1. vol target
print("\n" + "=" * 100)
print("1. VOLATILITY TARGETING  (scale exposure to hold risk constant, never zero)")
print("=" * 100)
realized_vol = r_base.rolling(60, min_periods=20).std() * np.sqrt(252)
target_vol = 0.20
lev = (target_vol / realized_vol.shift(1)).clip(0.3, 2.0).fillna(1.0)
r_vt = r_base * lev
summarise(r_base, cost_base, "unscaled (reference)")
summarise(r_vt, cost_base, "vol-targeted (20% ann. target)")
results["1_vol_target"] = summarise(r_vt, cost_base, "  -> full period", None)

# --------------------------------------------------------------- 2. trend filter
print("\n" + "=" * 100)
print("2. TREND FILTER  (exclude top-decile names trading below their 200d SMA)")
print("=" * 100)
w_trend = pd.DataFrame(0.0, index=px.index, columns=px.columns)
for i in entries:
    r_i = rk.iloc[i]
    tr_i = trend.iloc[i]
    sel = (r_i >= 0.90) & (tr_i > 0)
    if sel.sum() == 0:
        sel = r_i >= 0.90    # fall back rather than sit in cash entirely
    end = min(i + HOLD, len(dates))
    w_trend.iloc[i:end] = np.where(sel.values, 1.0 / sel.sum(), 0.0)
r_trend = (w_trend * ret).sum(axis=1)
summarise(r_trend, cost_base, "trend-filtered top decile")
results["2_trend_filter"] = summarise(r_trend, cost_base, "  -> full period", None)

# --------------------------------------------- 3. partial / incremental rebalance
print("\n" + "=" * 100)
print("3. PARTIAL REBALANCING  (keep top-quartile holdings, replace only drop-outs)")
print("=" * 100)
CHECK = 10   # review the book every 10 days, only replace names below Q1
w_partial = pd.DataFrame(0.0, index=px.index, columns=px.columns)
held = set()
turnover_events = 0
check_points = np.arange(0, len(dates), CHECK)
for i in check_points:
    r_i = rk.iloc[i].dropna()
    if r_i.empty:
        continue
    keep = {t for t in held if r_i.get(t, 0) >= 0.75}
    n_needed = 10 - len(keep)  # target book size similar to a typical top-decile count
    if n_needed > 0:
        candidates = r_i.drop(index=list(keep), errors="ignore").nlargest(n_needed)
        add = set(candidates.index)
    else:
        add = set()
    turnover_events += len(held ^ (keep | add))
    held = keep | add
    end = min(i + CHECK, len(dates))
    if held:
        w_partial.iloc[i:end] = 0.0
        for tk in held:
            w_partial.loc[w_partial.index[i:end], tk] = 1.0 / len(held)
r_partial = (w_partial * ret).sum(axis=1)
avg_daily_turnover = turnover_events / len(check_points) / 10 / (CHECK)
summarise(r_partial, avg_daily_turnover * 2 * BPS / 10000, "partial rebalance (10d checks)")
results["3_partial_rebal"] = summarise(
    r_partial, avg_daily_turnover * 2 * BPS / 10000, "  -> full period", None)

# --------------------------------------------------------------- 4. stop-loss
print("\n" + "=" * 100)
print("4. STOP-LOSS OVERLAY  (exit a single name if it falls >10% from entry)")
print("=" * 100)
STOP = -0.10
w_stop = pd.DataFrame(0.0, index=px.index, columns=px.columns)
stop_events = 0
for i in entries:
    sel = rk.iloc[i] >= 0.90
    names = [t for t in tickers if sel.get(t, False)]
    if not names:
        continue
    end = min(i + HOLD, len(dates))
    entry_px = px.iloc[i][names]
    n = len(names)
    for j in range(i, end):
        cum = px.iloc[j][names] / entry_px - 1
        alive = (cum > STOP) | cum.isna()
        for tk, ok in alive.items():
            if ok:
                w_stop.loc[w_stop.index[j], tk] = 1.0 / n
        stop_events += int((~alive).sum())
r_stop = (w_stop * ret).sum(axis=1)
summarise(r_stop, cost_base, f"stop-loss at {STOP:.0%} ({stop_events} exits triggered)")
results["4_stop_loss"] = summarise(r_stop, cost_base, "  -> full period", None)

# --------------------------------------------------------- 5. rank persistence
print("\n" + "=" * 100)
print("5. RANK PERSISTENCE  (require top-decile now AND at the prior rebalance)")
print("=" * 100)
w_persist = pd.DataFrame(0.0, index=px.index, columns=px.columns)
for k, i in enumerate(entries):
    r_i = rk.iloc[i]
    if k == 0:
        sel = r_i >= 0.90
    else:
        r_prev = rk.iloc[entries[k - 1]]
        sel = (r_i >= 0.90) & (r_prev.reindex(r_i.index) >= 0.90)
        if sel.sum() == 0:
            sel = r_i >= 0.90
    if sel.sum() == 0:
        continue
    end = min(i + HOLD, len(dates))
    w_persist.iloc[i:end] = np.where(sel.values, 1.0 / sel.sum(), 0.0)
r_persist = (w_persist * ret).sum(axis=1)
summarise(r_persist, cost_base, "persistence-filtered top decile")
results["5_rank_persistence"] = summarise(r_persist, cost_base, "  -> full period", None)

# ------------------------------------------------------- 6. long-only/hedge blend
print("\n" + "=" * 100)
print("6. LONG-ONLY / HEDGE BLEND  (diversify two return streams, not the hedge alone)")
print("=" * 100)
w_hedge = pd.DataFrame(0.0, index=px.index, columns=px.columns)
for i in entries:
    r_i = rk.iloc[i]
    longs = r_i[r_i >= 0.90].index
    shorts = r_i[r_i <= 0.10].index
    if len(longs) == 0 or len(shorts) == 0:
        continue
    vec = pd.Series(0.0, index=tickers)
    vec[longs] = 1.0 / len(longs)
    vec[shorts] -= 1.0 / len(shorts)
    end = min(i + HOLD, len(dates))
    w_hedge.iloc[i:end] = vec.values
r_hedge = (w_hedge * ret).sum(axis=1)
cost_hedge = 2.0 / HOLD * 2 * BPS / 10000
for blend in (0.25, 0.5, 0.75):
    r_blend = blend * r_base + (1 - blend) * r_hedge
    cost_blend = blend * cost_base + (1 - blend) * cost_hedge
    summarise(r_blend, cost_blend, f"{blend:.0%} long-only / {1-blend:.0%} hedge")
    results[f"6_blend_{int(blend*100)}"] = summarise(r_blend, cost_blend, "  ->", None)

# --------------------------------------------------------------- 7. rank momentum
print("\n" + "=" * 100)
print("7. RANK MOMENTUM  (select on IMPROVING rank, not absolute level)")
print("=" * 100)
rank_chg = rk - rk.shift(5)
w_mom = pd.DataFrame(0.0, index=px.index, columns=px.columns)
for i in entries:
    if i < 5:
        continue
    chg = rank_chg.iloc[i].dropna()
    if chg.empty:
        continue
    sel = chg.nlargest(max(int(len(chg) * 0.10), 1)).index
    end = min(i + HOLD, len(dates))
    w_mom.iloc[i:end] = 0.0
    for tk in sel:
        w_mom.loc[w_mom.index[i:end], tk] = 1.0 / len(sel)
r_mom = (w_mom * ret).sum(axis=1)
summarise(r_mom, cost_base, "top decile by 5d rank improvement")
results["7_rank_momentum"] = summarise(r_mom, cost_base, "  -> full period", None)

# --------------------------------------------------------------- 8. sector cap
print("\n" + "=" * 100)
print("8. SECTOR CAP  (no sector above 40% of book weight)")
print("=" * 100)
CAP = 0.40
w_cap = pd.DataFrame(0.0, index=px.index, columns=px.columns)
for i in entries:
    r_i = rk.iloc[i]
    sel = r_i[r_i >= 0.90].index.tolist()
    if not sel:
        continue
    vec = pd.Series(1.0 / len(sel), index=sel)
    secs = pd.Series({tk: sector_of.get(tk, "Other") for tk in sel})
    for sec in secs.unique():
        members = secs[secs == sec].index
        w_sum = vec[members].sum()
        if w_sum > CAP:
            vec[members] *= CAP / w_sum
    vec = vec / vec.sum()   # renormalise to fully invested
    end = min(i + HOLD, len(dates))
    w_cap.iloc[i:end] = 0.0
    for tk, wv in vec.items():
        w_cap.loc[w_cap.index[i:end], tk] = wv
r_cap = (w_cap * ret).sum(axis=1)
summarise(r_cap, cost_base, f"sector-capped at {CAP:.0%}")
results["8_sector_cap"] = summarise(r_cap, cost_base, "  -> full period", None)

# ------------------------------------------------------------------- regime split
print("\n" + "=" * 100)
print("ALL STRATEGIES, SPLIT BY REGIME  (bull vs flat half)")
print("=" * 100)
strategies = {
    "0 baseline":            (r_base, cost_base),
    "1 vol target":          (r_vt, cost_base),
    "2 trend filter":        (r_trend, cost_base),
    "3 partial rebalance":   (r_partial, avg_daily_turnover * 2 * BPS / 10000),
    "4 stop-loss":           (r_stop, cost_base),
    "5 rank persistence":    (r_persist, cost_base),
    "6 blend 50/50":         (0.5 * r_base + 0.5 * r_hedge, 0.5 * cost_base + 0.5 * cost_hedge),
    "7 rank momentum":       (r_mom, cost_base),
    "8 sector cap":          (r_cap, cost_base),
}
for period, subset in (("HALF A (bull)", pd.Series(px.index < mid, index=px.index)),
                       ("HALF B (flat)", pd.Series(px.index >= mid, index=px.index))):
    print(f"\n--- {period} ---")
    print(f"{'strategy':38s} {'gross':>9s} {'net':>10s} {'ann.':>8s} {'Sharpe':>7s} "
          f"{'maxDD':>8s} {'NW-t':>6s}")
    for name, (r, c) in strategies.items():
        summarise(r, c, name, subset)

print("\n" + "=" * 100)
print("SUMMARY: FULL-PERIOD SHARPE vs BASELINE 1.40")
print("=" * 100)
for k, v in results.items():
    if v:
        flag = " <-- BEATS BASELINE" if v["sharpe"] > 1.40 and v["t"] > 2.77 else ""
        print(f"  {k:24s} Sharpe={v['sharpe']:.2f}  net={v['net']:+.1%}  t={v['t']:.2f}{flag}")
print("=" * 100)
