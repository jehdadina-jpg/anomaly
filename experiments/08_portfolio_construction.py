"""
Experiment 15: advanced portfolio construction.

The 30-day result rests on 34 rebalances that all start on one particular
grid of entry dates. That is both a small sample and a hidden bet on entry
timing. This tests four constructions that real books actually use:

  A single-tranche (current)   rebalance the whole book every 30 days
  B overlapping tranches       run 30 staggered sub-books, each entering on a
                               different day and holding 30 days. Same holding
                               period and same per-tranche turnover, but the
                               aggregate is diversified across entry dates
                               instead of betting on one grid. Standard
                               practice; also a direct robustness test -- if
                               the 30-day edge only exists on one entry grid,
                               this destroys it.
  C inverse-vol weights        size positions by 1/vol instead of equally, so
                               no single volatile name dominates book risk.
  D signal-decay exit          hold while the name stays in the top tercile
                               rather than for a fixed 30 days; exit when the
                               model stops liking it (capped at 60 days).

Reported net of costs, with turnover made explicit, because turnover is the
binding constraint established in section 8 of the validation doc.
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
HOLD = 30
BPS = 10

preds = pd.read_csv(rf"{ROOT}\results\models\prediction\walk_forward_predictions.csv",
                    parse_dates=["Date"])
feat = build_features(load_panel())[["Date", "ticker", "Close", "vol_20d"]]
feat = feat.sort_values(["ticker", "Date"])

d = preds[["Date", "ticker", "p"]].merge(feat, on=["Date", "ticker"], how="left")
d = d.sort_values(["Date", "ticker"]).reset_index(drop=True)
d["rk"] = d.groupby("Date")["p"].rank(pct=True)

# Wide price matrix for path-dependent holding.
px = d.pivot_table(index="Date", columns="ticker", values="Close")
rk = d.pivot_table(index="Date", columns="ticker", values="rk")
vol = d.pivot_table(index="Date", columns="ticker", values="vol_20d")
dates = px.index.to_numpy()
ret = px.pct_change().shift(-1)          # return earned from t to t+1


def summarise(daily_ret: pd.Series, turnover_per_day: float, label: str):
    """daily_ret is a per-day portfolio return series (already position-weighted)."""
    r = daily_ret.dropna()
    if len(r) < 30:
        print(f"{label:34s}  (insufficient data)")
        return
    cost = turnover_per_day * 2 * BPS / 10000     # buy+sell legs
    net = r - cost
    yrs = len(r) / 252
    g = (1 + r).prod() - 1
    n = (1 + net).prod() - 1
    sharpe = net.mean() / (net.std() + 1e-12) * np.sqrt(252)
    eq = (1 + net).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    _, t = newey_west_tstat(net.values, lag=5)
    print(f"{label:34s} {g:+9.1%} {n:+10.1%} {(1 + n) ** (1 / yrs) - 1:+8.1%} "
          f"{sharpe:7.2f} {dd:+8.1%} {t:6.2f} {turnover_per_day:8.1%}")


print("=" * 108)
print(f"PORTFOLIO CONSTRUCTION  ({HOLD}-day holds, net of {BPS}bps per leg)")
print("=" * 108)
print(f"{'construction':34s} {'gross':>9s} {'net':>10s} {'ann.':>8s} "
      f"{'Sharpe':>7s} {'maxDD':>8s} {'NW-t':>6s} {'turnover':>9s}")

# ---------------------------------------------------------------- A: single grid
entry_idx = np.arange(0, len(dates), HOLD)
w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
for i in entry_idx:
    sel = rk.iloc[i] >= 0.90
    if sel.sum() == 0:
        continue
    end = min(i + HOLD, len(dates))
    w.iloc[i:end] = np.where(sel.values, 1.0 / sel.sum(), 0.0)
port_a = (w * ret).sum(axis=1)
summarise(port_a, turnover_per_day=1.0 / HOLD, label="A single tranche (current)")

# ------------------------------------------------------- B: overlapping tranches
# 30 sub-books, one entering each day, each holding HOLD days.
w_b = pd.DataFrame(0.0, index=px.index, columns=px.columns)
for offset in range(HOLD):
    for i in range(offset, len(dates), HOLD):
        sel = rk.iloc[i] >= 0.90
        if sel.sum() == 0:
            continue
        end = min(i + HOLD, len(dates))
        w_b.iloc[i:end] += np.where(sel.values, 1.0 / sel.sum(), 0.0) / HOLD
port_b = (w_b * ret).sum(axis=1)
summarise(port_b, turnover_per_day=1.0 / HOLD, label="B overlapping tranches")

# ------------------------------------------------------------- C: inverse-vol
w_c = pd.DataFrame(0.0, index=px.index, columns=px.columns)
for offset in range(HOLD):
    for i in range(offset, len(dates), HOLD):
        sel = rk.iloc[i] >= 0.90
        if sel.sum() == 0:
            continue
        iv = (1.0 / vol.iloc[i]).where(sel, 0.0).fillna(0.0)
        if iv.sum() <= 0:
            continue
        iv = iv / iv.sum()
        end = min(i + HOLD, len(dates))
        w_c.iloc[i:end] += iv.values / HOLD
port_c = (w_c * ret).sum(axis=1)
summarise(port_c, turnover_per_day=1.0 / HOLD, label="C overlapping + inverse-vol")

# --------------------------------------------------------- D: signal-decay exit
# Enter on the same staggered grid, but exit early if the name drops out of
# the top tercile; cap the hold at 60 days.
MAXH = 60
w_d = pd.DataFrame(0.0, index=px.index, columns=px.columns)
n_held_days = []
for offset in range(HOLD):
    for i in range(offset, len(dates), HOLD):
        sel = rk.iloc[i] >= 0.90
        if sel.sum() == 0:
            continue
        for tk in px.columns[sel.values]:
            j = i
            while j < min(i + MAXH, len(dates)) and (
                    pd.isna(rk.iloc[j][tk]) or rk.iloc[j][tk] >= 0.667):
                j += 1
            j = max(j, i + 1)
            w_d.iloc[i:j, w_d.columns.get_loc(tk)] += (1.0 / sel.sum()) / HOLD
            n_held_days.append(j - i)
port_d = (w_d * ret).sum(axis=1)
avg_hold = np.mean(n_held_days) if n_held_days else HOLD
summarise(port_d, turnover_per_day=1.0 / max(avg_hold, 1),
          label=f"D decay exit (avg {avg_hold:.0f}d hold)")

print("=" * 108)

# ------------------------------------------------------------------ benchmark
mkt = ret.mean(axis=1)
summarise(mkt, turnover_per_day=0.0, label="market (equal-weight, no cost)")
print("=" * 108)
print("B is also a robustness test: if the 30-day edge only existed on one entry")
print("grid, diversifying across all 30 possible entry days would erase it.")
