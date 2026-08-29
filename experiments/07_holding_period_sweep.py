"""
Experiment 14: final backtest of the high-conviction configuration.

Established by experiment 13:
  * Top-decile win rate tracks the market's up-day rate almost exactly
    (64.1% vs 63.7% in the bull half; 50.1% vs 52.6% in the flat half).
    Absolute "accuracy" is therefore mostly market drift, not model skill.
  * The model's genuine contribution is the LIFT over that base rate, and
    a conviction filter (high delivery + volume confirmation) added
    +2.6% in the derivation half and +4.1% in the untouched validation
    half. That lift replicated out of sample, so it is real.
  * Longer holds win more often, because drift has more time to work.

This backtests the resulting configuration honestly, against the current
one, net of costs, split by regime so the market-dependence is visible
rather than hidden in an average.
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
preds = pd.read_csv(rf"{ROOT}\results\models\prediction\walk_forward_predictions.csv",
                    parse_dates=["Date"])
feat = build_features(load_panel())
cols = ["Date", "ticker", "Close", "delivery_pct", "volume_ratio_20"]
feat = feat[cols].sort_values(["ticker", "Date"])
for h in (3, 10):
    feat[f"fwd_{h}d"] = feat.groupby("ticker")["Close"].transform(
        lambda x: x.shift(-h) / x - 1)

d = preds[["Date", "ticker", "p"]].merge(feat, on=["Date", "ticker"], how="left")
d = d.sort_values(["Date", "ticker"]).reset_index(drop=True)
d["rk"] = d.groupby("Date")["p"].rank(pct=True)

# The validated conviction rule.
d["conviction"] = (d["rk"] >= 0.90) & (d["delivery_pct"] > 0.60) & (d["volume_ratio_20"] > 1.2)
d["baseline"] = d["rk"] >= 0.90

dates = np.sort(d["Date"].unique())
mid = dates[len(dates) // 2]


def backtest(sel_col, h, frame, label, bps=10):
    """Non-overlapping h-day holds, equal-weight, costs charged per rebalance."""
    col = f"fwd_{h}d"
    f = frame.dropna(subset=[col])
    rebal = set(np.sort(f["Date"].unique())[::h])
    held = f[f["Date"].isin(rebal) & f[sel_col]]
    if held.empty:
        return None
    per = held.groupby("Date")[col].mean().dropna()
    if len(per) < 5:
        return None

    ppy = 252 / h
    yrs = len(per) / ppy
    gross = (1 + per).prod() - 1
    net_per = per - 4 * bps / 10000
    net = (1 + net_per).prod() - 1
    sharpe = per.mean() / (per.std() + 1e-12) * np.sqrt(ppy)
    _, nw_t = newey_west_tstat(per.values, lag=1)
    eq = (1 + per).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    trade_win = (held[col] > 0).mean()

    print(f"{label:34s} {len(held):7d} {trade_win:7.1%} {per.mean():+8.2%} "
          f"{gross:+9.1%} {net:+9.1%} {sharpe:6.2f} {dd:+7.1%} {nw_t:6.2f}")
    return dict(n=len(held), win=trade_win, gross=gross, net=net, sharpe=sharpe)


for half_name, frame in (("FULL PERIOD", d),
                         ("half A (bull market, +90%)", d[d["Date"] < mid]),
                         ("half B (flat market, +3%)", d[d["Date"] >= mid])):
    print("\n" + "=" * 100)
    print(f"{half_name}")
    print("=" * 100)
    print(f"{'strategy':34s} {'trades':>7s} {'WIN':>7s} {'avg/hold':>8s} "
          f"{'gross':>9s} {'net@10bp':>9s} {'Sharpe':>6s} {'maxDD':>7s} {'NW-t':>6s}")
    backtest("baseline", 3, frame, "current: top-decile, 3d hold")
    backtest("baseline", 10, frame, "top-decile, 10d hold")
    backtest("conviction", 10, frame, "CONVICTION: +delivery+volume, 10d")

print("\n" + "=" * 100)
print("SIGNAL FREQUENCY (how often the conviction rule fires)")
print("=" * 100)
tot_days = d["Date"].nunique()
conv_days = d[d["conviction"]]["Date"].nunique()
print(f"  trading days covered            {tot_days}")
print(f"  days with >=1 conviction signal {conv_days}  ({conv_days / tot_days:.0%})")
print(f"  avg conviction names per day    {d['conviction'].sum() / tot_days:.2f}")
print(f"  avg top-decile names per day    {d['baseline'].sum() / tot_days:.2f}")
