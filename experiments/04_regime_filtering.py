"""
Experiment 11: can the model know in advance when to trust itself?

Diagnostic finding: out-of-sample IC is 0.0555 on days the market rose and
0.0030 on days it fell -- the signal essentially vanishes in down markets.
That split uses SAME-DAY market return, which is not knowable at decision
time, so it is a description, not a strategy.

This tests whether regime variables that ARE knowable before the bell
(trailing market trend, trailing volatility, breadth) predict when the model
works. If they do, the score can be down-weighted in hostile regimes instead
of being published with false confidence.

Strictly causal: every regime feature uses data up to and including t-1.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\mdadi\Downloads\anomalydetection")
from data.build_panel import load_panel
from evaluation.stats import newey_west_tstat

H = 3
preds = pd.read_csv(
    r"C:\Users\mdadi\Downloads\anomalydetection\results\models\prediction\walk_forward_predictions.csv",
    parse_dates=["Date"]).dropna(subset=["fwd_3d"])

panel = load_panel()[["Date", "ticker", "Close"]].sort_values(["ticker", "Date"])
panel["ret_1d"] = panel.groupby("ticker")["Close"].pct_change()

# --- market series, then LAG everything by one day ---
mkt = panel.groupby("Date")["ret_1d"].mean().sort_index()
regime = pd.DataFrame(index=mkt.index)
regime["mkt_ret_20d"] = mkt.rolling(20).sum()
regime["mkt_ret_60d"] = mkt.rolling(60).sum()
regime["mkt_vol_20d"] = mkt.rolling(20).std()
regime["mkt_vol_ratio"] = mkt.rolling(5).std() / (mkt.rolling(60).std() + 1e-12)
breadth = panel.assign(up=panel["ret_1d"] > 0).groupby("Date")["up"].mean()
regime["breadth_20d"] = breadth.rolling(20).mean()
regime["drawdown"] = (mkt.cumsum() - mkt.cumsum().cummax())

# CRITICAL: shift(1) so day t only ever sees information through t-1.
regime = regime.shift(1)

# --- per-day realised IC ---
ic = preds.groupby("Date").apply(
    lambda x: spearmanr(x["p"], x["fwd_3d"]).statistic if len(x) >= 10 else np.nan
).dropna().rename("ic")

j = pd.DataFrame(ic).join(regime, how="inner").dropna()
print(f"{len(j)} days with both IC and lagged regime data\n")

print("=" * 78)
print("DOES A KNOWABLE-IN-ADVANCE REGIME PREDICT WHEN THE MODEL WORKS?")
print("=" * 78)
print(f"{'regime variable':18s} {'corr w/ IC':>11s} {'low-half IC':>12s} {'high-half IC':>13s} {'gap':>8s}")
results = {}
for col in ["mkt_ret_20d", "mkt_ret_60d", "mkt_vol_20d", "mkt_vol_ratio",
            "breadth_20d", "drawdown"]:
    c = spearmanr(j[col], j["ic"]).statistic
    med = j[col].median()
    lo = j.loc[j[col] <= med, "ic"].mean()
    hi = j.loc[j[col] > med, "ic"].mean()
    results[col] = (c, lo, hi, hi - lo)
    print(f"{col:18s} {c:+11.4f} {lo:+12.4f} {hi:+13.4f} {hi - lo:+8.4f}")
print("=" * 78)

best = max(results.items(), key=lambda kv: abs(kv[1][3]))
print(f"\nStrongest separator: {best[0]}  (gap {best[1][3]:+.4f})")

# --- does filtering on it actually pay? ---
print("\n" + "=" * 78)
print("TRADING ONLY IN THE FAVOURABLE REGIME  (top-decile forward return)")
print("=" * 78)
d = preds.copy()
d["rk"] = d.groupby("Date")["p"].rank(pct=True)
top = d[d["rk"] >= 0.9].groupby("Date")["fwd_3d"].mean().rename("top_ret")
t = pd.DataFrame(top).join(regime, how="inner").dropna()

print(f"{'filter':34s} {'days':>6s} {'avg 3d ret':>11s} {'ann.':>8s} {'win%':>7s}")
allr = t["top_ret"]
print(f"{'no filter (always trade)':34s} {len(allr):6d} {allr.mean():+11.3%} "
      f"{allr.mean() * (252 / H):+8.1%} {(allr > 0).mean():6.1%}")

for col in ["mkt_ret_20d", "mkt_ret_60d", "breadth_20d", "drawdown"]:
    for label, mask in ((f"{col} > 0", t[col] > 0),
                        (f"{col} > median", t[col] > t[col].median())):
        sel = t.loc[mask, "top_ret"]
        if len(sel) < 30:
            continue
        print(f"{label:34s} {len(sel):6d} {sel.mean():+11.3%} "
              f"{sel.mean() * (252 / H):+8.1%} {(sel > 0).mean():6.1%}")
print("=" * 78)
print("A filter only helps if it beats 'no filter' AFTER giving up the days it skips.")
