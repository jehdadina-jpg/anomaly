"""
Experiment 17: sector-relative pairs -- a structurally different hedge.

The dollar-neutral hedge tested previously (long top decile / short bottom
decile, whole market) measured t = 0.57: statistically zero. It failed
partly because the short leg pools stocks from every sector, so it hedges
market beta reasonably but mixes in sector rotation noise that has nothing
to do with the model's stock-picking skill.

This tries a narrower, more defensible construction: within EACH sector,
long the single highest-ranked stock and short the single lowest-ranked
stock. A sector with only one name contributes nothing (both legs would be
the same stock). This removes both market beta AND sector rotation from the
return, leaving (if it exists) pure within-sector stock selection.

Compared against:
  A  long-only top decile, 30d hold          (current shipped strategy)
  B  dollar-neutral long-short, whole market  (already measured: t=0.57)
  C  sector-relative pairs                    (this experiment)
  D  sector-relative pairs, held only in liquid/large sectors (>=4 names)

All on identical overlapping 30-day tranches, net of realistic costs, tested
on the same purged out-of-sample predictions used throughout this project.
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
feat = build_features(load_panel())[["Date", "ticker", "Close", "sector"]]
d = preds[["Date", "ticker", "p"]].merge(feat, on=["Date", "ticker"], how="left")
d = d.sort_values(["Date", "ticker"]).reset_index(drop=True)
d["rk"] = d.groupby("Date")["p"].rank(pct=True)

sector_map = feat.drop_duplicates("ticker").set_index("ticker")["sector"].to_dict()
sector_size = pd.Series(sector_map).value_counts()

px = d.pivot_table(index="Date", columns="ticker", values="Close")
rk = d.pivot_table(index="Date", columns="ticker", values="rk")
ret = px.pct_change().shift(-1)
dates = px.index.to_numpy()
mid = dates[len(dates) // 2]
tickers = px.columns.tolist()


def build_weights(kind: str, min_sector_size: int = 2):
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    gross_series = []
    for offset in range(HOLD):
        for i in range(offset, len(dates), HOLD):
            r_i = rk.iloc[i].dropna()
            if r_i.empty:
                continue
            vec = pd.Series(0.0, index=tickers)

            if kind == "long_only":
                longs = r_i[r_i >= 0.90].index
                if len(longs) == 0:
                    continue
                vec[longs] = 1.0 / len(longs)
                gross = 1.0

            elif kind == "dollar_neutral":
                longs = r_i[r_i >= 0.90].index
                shorts = r_i[r_i <= 0.10].index
                if len(longs) == 0 or len(shorts) == 0:
                    continue
                vec[longs] = 1.0 / len(longs)
                vec[shorts] -= 1.0 / len(shorts)
                gross = 2.0

            elif kind in ("sector_pairs", "sector_pairs_liquid"):
                pair_longs, pair_shorts = [], []
                for sec, names in feat.drop_duplicates("ticker").groupby("sector")["ticker"]:
                    if kind == "sector_pairs_liquid" and sector_size.get(sec, 0) < min_sector_size:
                        continue
                    sub = r_i.reindex(names.tolist()).dropna()
                    if len(sub) < 2:
                        continue
                    best = sub.idxmax()
                    worst = sub.idxmin()
                    if best == worst:
                        continue
                    pair_longs.append(best)
                    pair_shorts.append(worst)
                if not pair_longs:
                    continue
                n_pairs = len(pair_longs)
                vec[pair_longs] += 1.0 / n_pairs
                vec[pair_shorts] -= 1.0 / n_pairs
                gross = 2.0
            else:
                raise ValueError(kind)

            end = min(i + HOLD, len(dates))
            w.iloc[i:end] += vec.values / HOLD
            gross_series.append(gross)
    avg_gross = float(np.mean(gross_series)) if gross_series else 1.0
    return w, avg_gross


def summarise(r: pd.Series, gross_exposure: float, label: str, subset=None):
    s = r.dropna()
    if subset is not None:
        s = s[subset.reindex(s.index).fillna(False)]
    if len(s) < 30:
        print(f"{label:38s}  (insufficient data)")
        return None
    cost = (gross_exposure / HOLD) * 2 * BPS / 10000
    net = s - cost
    yrs = len(net) / 252
    g = (1 + s).prod() - 1
    n = (1 + net).prod() - 1
    sh = net.mean() / (net.std() + 1e-12) * np.sqrt(252)
    eq = (1 + net).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    _, t = newey_west_tstat(net.values, lag=5)
    print(f"{label:38s} {g:+9.1%} {n:+10.1%} {(1 + n) ** (1 / yrs) - 1:+8.1%} "
          f"{sh:7.2f} {dd:+8.1%} {t:6.2f}")
    return dict(net=n, sharpe=sh, t=t)


print("Sector sizes:")
for sec, n in sector_size.sort_values(ascending=False).items():
    print(f"  {sec:16s} {n}")
print()

mkt = ret.mean(axis=1)
books = {}
for kind, label in (("long_only", "A long-only top decile"),
                    ("dollar_neutral", "B dollar-neutral (whole market)"),
                    ("sector_pairs", "C sector pairs (all sectors)"),
                    ("sector_pairs_liquid", "D sector pairs (sectors >=4 names)")):
    w, gx = build_weights(kind, min_sector_size=4)
    books[label] = ((w * ret).sum(axis=1), gx)

for period, subset in (("FULL PERIOD", None),
                       ("HALF A (bull, market +90%)", pd.Series(px.index < mid, index=px.index)),
                       ("HALF B (flat, market +3%)", pd.Series(px.index >= mid, index=px.index))):
    print("\n" + "=" * 96)
    print(period)
    print("=" * 96)
    print(f"{'book':38s} {'gross':>9s} {'net':>10s} {'ann.':>8s} {'Sharpe':>7s} "
          f"{'maxDD':>8s} {'NW-t':>6s}")
    for label, (r, gx) in books.items():
        summarise(r, gx, label, subset)
    summarise(mkt, 0.0, "market (equal-weight, no cost)", subset)

print("=" * 96)
print("A sector-pair book with t > 2 in BOTH halves would be the first genuine")
print("evidence of hedgeable stock-selection alpha found in this project.")
