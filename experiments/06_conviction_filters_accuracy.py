"""
Experiment 13: maximise ACCURACY, not rank IC.

Different objective from everything before. Rank IC measures how well the
model orders all 48 stocks. Accuracy measures: of the calls it actually
makes, how many win. Those are optimised by different things -- you raise
accuracy by being SELECTIVE, making fewer but better calls.

The current terminal always publishes a top decile (~5 names/day) regardless
of conviction. This tests whether demanding higher conviction, longer holds,
and feature agreement produces a materially higher hit rate.

Anti-overfitting protocol: every filter is selected on the FIRST HALF of the
out-of-sample period and then validated on the SECOND HALF, which is never
looked at while choosing. A filter that only works in-sample is exactly the
failure mode that has already bitten twice in this project.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\mdadi\Downloads\anomalydetection")
from data.build_panel import load_panel
from features.predictive import build_features

ROOT = r"C:\Users\mdadi\Downloads\anomalydetection"
preds = pd.read_csv(rf"{ROOT}\results\models\prediction\walk_forward_predictions.csv",
                    parse_dates=["Date"])

print("Building features for filter conditions...")
feat = build_features(load_panel())
KEEP = ["Date", "ticker", "Close", "vol_20d", "rsi_14", "adx_14", "momentum_60d",
        "delivery_pct", "delivery_pct_z20", "px_to_sma_20", "px_to_sma_50",
        "range_position_52w", "volume_ratio_20", "beta_60d", "market_breadth",
        "vol_ratio_5_20", "amihud_illiq"]
feat = feat[KEEP].sort_values(["ticker", "Date"])

# forward returns at several horizons
for h in (1, 3, 5, 10, 20):
    feat[f"fwd_{h}d"] = feat.groupby("ticker")["Close"].transform(
        lambda x: x.shift(-h) / x - 1)

d = preds[["Date", "ticker", "p"]].merge(feat, on=["Date", "ticker"], how="left")
d = d.sort_values(["Date", "ticker"]).reset_index(drop=True)
d["rk"] = d.groupby("Date")["p"].rank(pct=True)

# split OOS period in half: derive on A, validate on B
dates = np.sort(d["Date"].unique())
mid = dates[len(dates) // 2]
A = d[d["Date"] < mid]
B = d[d["Date"] >= mid]
print(f"derivation half: {pd.Timestamp(dates[0]).date()} -> {pd.Timestamp(mid).date()} "
      f"({A['Date'].nunique()} days)")
print(f"validation half: {pd.Timestamp(mid).date()} -> {pd.Timestamp(dates[-1]).date()} "
      f"({B['Date'].nunique()} days)\n")


def stats(sel, h):
    col = f"fwd_{h}d"
    s = sel.dropna(subset=[col])
    if len(s) < 30:
        return None
    return {
        "n": len(s),
        "per_day": len(s) / max(s["Date"].nunique(), 1),
        "win": (s[col] > 0).mean(),
        "avg": s[col].mean(),
        "med": s[col].median(),
    }


# ---------------------------------------------------------------- 1. conviction
print("=" * 92)
print("1. DOES DEMANDING MORE CONVICTION RAISE THE HIT RATE?  (derivation half)")
print("=" * 92)
print(f"{'rule':30s} {'hold':>5s} {'signals':>8s} {'per day':>8s} {'WIN RATE':>9s} {'avg ret':>9s}")
for h in (3, 5, 10, 20):
    for label, thr in [("top decile (current)", 0.90), ("top 5%", 0.95),
                       ("top 2%", 0.98)]:
        r = stats(A[A["rk"] >= thr], h)
        if r:
            print(f"{label:30s} {h:4d}d {r['n']:8d} {r['per_day']:8.1f} "
                  f"{r['win']:8.1%} {r['avg']:+8.2%}")
    print()

# ------------------------------------------------------------------ 2. filters
print("=" * 92)
print("2. FEATURE FILTERS ON TOP-DECILE NAMES  (derivation half, 10-day hold)")
print("=" * 92)
H = 10
base = A[A["rk"] >= 0.90]
b = stats(base, H)
print(f"{'filter':42s} {'signals':>8s} {'WIN RATE':>9s} {'avg ret':>9s} {'lift':>7s}")
print(f"{'(no filter)':42s} {b['n']:8d} {b['win']:8.1%} {b['avg']:+8.2%} {'--':>7s}")

candidates = {
    "uptrend: px > 20d SMA":            base["px_to_sma_20"] > 0,
    "uptrend: px > 50d SMA":            base["px_to_sma_50"] > 0,
    "strong trend: ADX > 25":           base["adx_14"] > 25,
    "not overbought: RSI < 70":         base["rsi_14"] < 70,
    "oversold: RSI < 40":               base["rsi_14"] < 40,
    "high delivery: z > 0":             base["delivery_pct_z20"] > 0,
    "high delivery: pct > 60%":         base["delivery_pct"] > 0.60,
    "low vol (bottom half)":            base["vol_20d"] < base["vol_20d"].median(),
    "high vol (top half)":              base["vol_20d"] > base["vol_20d"].median(),
    "low beta < 1":                     base["beta_60d"] < 1.0,
    "not near 52w high":                base["range_position_52w"] < 0.8,
    "near 52w high":                    base["range_position_52w"] > 0.8,
    "volume confirm: >1.2x avg":        base["volume_ratio_20"] > 1.2,
    "calm regime: vol_ratio < 1":       base["vol_ratio_5_20"] < 1.0,
    "broad market: breadth > 0.5":      base["market_breadth"] > 0.5,
}
scored = []
for name, mask in candidates.items():
    r = stats(base[mask], H)
    if r:
        lift = r["win"] - b["win"]
        scored.append((name, mask, r, lift))
        print(f"{name:42s} {r['n']:8d} {r['win']:8.1%} {r['avg']:+8.2%} {lift:+7.1%}")

# ------------------------------------------------------- 3. best combo, validated
print("\n" + "=" * 92)
print("3. BEST COMBINATION -- DERIVED ON HALF A, VALIDATED ON UNSEEN HALF B")
print("=" * 92)
top3 = sorted(scored, key=lambda x: -x[3])[:3]
print("strongest single filters on half A:")
for name, _, r, lift in top3:
    print(f"   {name:40s} win {r['win']:.1%} ({lift:+.1%}), n={r['n']}")

def apply_named(frame, names):
    m = pd.Series(True, index=frame.index)
    for nm in names:
        if nm == "uptrend: px > 20d SMA":      m &= frame["px_to_sma_20"] > 0
        elif nm == "uptrend: px > 50d SMA":    m &= frame["px_to_sma_50"] > 0
        elif nm == "strong trend: ADX > 25":   m &= frame["adx_14"] > 25
        elif nm == "not overbought: RSI < 70": m &= frame["rsi_14"] < 70
        elif nm == "oversold: RSI < 40":       m &= frame["rsi_14"] < 40
        elif nm == "high delivery: z > 0":     m &= frame["delivery_pct_z20"] > 0
        elif nm == "high delivery: pct > 60%": m &= frame["delivery_pct"] > 0.60
        elif nm == "low vol (bottom half)":    m &= frame["vol_20d"] < base["vol_20d"].median()
        elif nm == "high vol (top half)":      m &= frame["vol_20d"] > base["vol_20d"].median()
        elif nm == "low beta < 1":             m &= frame["beta_60d"] < 1.0
        elif nm == "not near 52w high":        m &= frame["range_position_52w"] < 0.8
        elif nm == "near 52w high":            m &= frame["range_position_52w"] > 0.8
        elif nm == "volume confirm: >1.2x avg":m &= frame["volume_ratio_20"] > 1.2
        elif nm == "calm regime: vol_ratio < 1":m &= frame["vol_ratio_5_20"] < 1.0
        elif nm == "broad market: breadth > 0.5":m &= frame["market_breadth"] > 0.5
    return m

combos = [[top3[0][0]], [top3[0][0], top3[1][0]], [top3[0][0], top3[1][0], top3[2][0]]]
print(f"\n{'configuration':52s} {'half':>5s} {'signals':>8s} {'per day':>8s} {'WIN':>7s} {'avg':>8s}")
for names in combos:
    label = " + ".join(n.split(":")[0] for n in names)
    for half_name, half in (("A", A), ("B", B)):
        sel = half[half["rk"] >= 0.90]
        sel = sel[apply_named(sel, names)]
        r = stats(sel, H)
        if r:
            print(f"{label[:52]:52s} {half_name:>5s} {r['n']:8d} {r['per_day']:8.2f} "
                  f"{r['win']:6.1%} {r['avg']:+8.2%}")
    print()

# baseline on both halves for reference
print(f"{'BASELINE top-decile, no filter':52s}")
for half_name, half in (("A", A), ("B", B)):
    r = stats(half[half["rk"] >= 0.90], H)
    print(f"{'':52s} {half_name:>5s} {r['n']:8d} {r['per_day']:8.2f} "
          f"{r['win']:6.1%} {r['avg']:+8.2%}")
print("=" * 92)
print("A filter is only real if its half-B win rate holds up. Half A is where it was chosen.")
