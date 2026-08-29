"""
Experiment 16: hedge out the market and see what is actually left.

Established so far: top-decile win rate tracks the market's up-day rate
(64.1% vs 63.7% in the bull half, 50.1% vs 52.6% in the flat half), and the
long-only book made +87% net in the bull half but -5% in the flat half. That
is the signature of a position dominated by market exposure.

If the model has genuine stock-selection alpha, removing market exposure
should leave something positive that works in BOTH halves. If it does not,
the long-only strategy is mostly a leveraged bet on the market and should be
described that way.

Three hedged constructions, all on overlapping 30-day tranches:
  LS   long top decile, short bottom decile (dollar neutral)
  LSb  same, but short sized by beta ratio so the book is beta neutral
  LM   long top decile, short the equal-weight universe (market neutral)
Compared against long-only and against the market itself, split by regime.
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
feat = build_features(load_panel())[["Date", "ticker", "Close", "beta_60d"]]
d = preds[["Date", "ticker", "p"]].merge(feat, on=["Date", "ticker"], how="left")
d = d.sort_values(["Date", "ticker"]).reset_index(drop=True)
d["rk"] = d.groupby("Date")["p"].rank(pct=True)

px = d.pivot_table(index="Date", columns="ticker", values="Close")
rk = d.pivot_table(index="Date", columns="ticker", values="rk")
bt = d.pivot_table(index="Date", columns="ticker", values="beta_60d")
ret = px.pct_change().shift(-1)
dates = px.index.to_numpy()
mid = dates[len(dates) // 2]


def build_weights(kind: str) -> tuple[pd.DataFrame, float]:
    """Overlapping 30-day tranches. Returns (weights, gross exposure)."""
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    gross = 1.0
    for offset in range(HOLD):
        for i in range(offset, len(dates), HOLD):
            r_i = rk.iloc[i]
            longs = r_i >= 0.90
            shorts = r_i <= 0.10
            if longs.sum() == 0:
                continue
            lw = np.where(longs.values, 1.0 / longs.sum(), 0.0)

            if kind == "long_only":
                vec = lw
            elif kind == "long_short":
                if shorts.sum() == 0:
                    continue
                sw = np.where(shorts.values, 1.0 / shorts.sum(), 0.0)
                vec = lw - sw
                gross = 2.0
            elif kind == "long_short_beta":
                if shorts.sum() == 0:
                    continue
                bl = np.nansum(np.where(longs.values, bt.iloc[i].values, np.nan)
                               * lw) / max(lw.sum(), 1e-9)
                sw_raw = np.where(shorts.values, 1.0 / shorts.sum(), 0.0)
                bs = np.nansum(np.where(shorts.values, bt.iloc[i].values, np.nan)
                               * sw_raw) / max(sw_raw.sum(), 1e-9)
                ratio = (bl / bs) if (bs and np.isfinite(bs) and bs != 0) else 1.0
                ratio = float(np.clip(ratio, 0.3, 3.0))
                vec = lw - sw_raw * ratio
                gross = 1.0 + ratio
            elif kind == "long_minus_market":
                mw = np.full(len(px.columns), 1.0 / len(px.columns))
                vec = lw - mw
                gross = 2.0
            else:
                raise ValueError(kind)

            end = min(i + HOLD, len(dates))
            w.iloc[i:end] += vec / HOLD
    return w, gross


def summarise(r: pd.Series, gross_exposure: float, label: str, subset=None):
    s = r.dropna()
    if subset is not None:
        s = s[subset.reindex(s.index).fillna(False)]
    if len(s) < 30:
        return
    # Turnover: one full rotation of the book every HOLD days, both legs.
    cost = (gross_exposure / HOLD) * 2 * BPS / 10000
    net = s - cost
    yrs = len(net) / 252
    g = (1 + s).prod() - 1
    n = (1 + net).prod() - 1
    sh = net.mean() / (net.std() + 1e-12) * np.sqrt(252)
    eq = (1 + net).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    _, t = newey_west_tstat(net.values, lag=5)
    print(f"{label:34s} {g:+9.1%} {n:+10.1%} {(1 + n) ** (1 / yrs) - 1:+8.1%} "
          f"{sh:7.2f} {dd:+8.1%} {t:6.2f}")


mkt = ret.mean(axis=1)
books = {}
for kind, label in (("long_only", "long-only top decile"),
                    ("long_short", "long-short (dollar neutral)"),
                    ("long_short_beta", "long-short (beta neutral)"),
                    ("long_minus_market", "long minus market")):
    w, gx = build_weights(kind)
    books[label] = ((w * ret).sum(axis=1), gx)

for period, subset in (("FULL PERIOD", None),
                       ("HALF A (bull, market +90%)", pd.Series(px.index < mid, index=px.index)),
                       ("HALF B (flat, market +3%)", pd.Series(px.index >= mid, index=px.index))):
    print("\n" + "=" * 96)
    print(period)
    print("=" * 96)
    print(f"{'book':34s} {'gross':>9s} {'net':>10s} {'ann.':>8s} {'Sharpe':>7s} "
          f"{'maxDD':>8s} {'NW-t':>6s}")
    for label, (r, gx) in books.items():
        summarise(r, gx, label, subset)
    summarise(mkt, 0.0, "market (equal-weight, no cost)", subset)
print("=" * 96)
print("A hedged book that works in BOTH halves is alpha. One that only works in")
print("half A is market exposure wearing a model as a hat.")
