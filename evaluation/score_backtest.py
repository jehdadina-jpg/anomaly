"""
What actually happens if you trade the buy score?

Answers the practical question directly: buy the stocks scoring 10/10, sell
after N days — how often do you make money, and how much?

Uses ONLY out-of-sample walk-forward predictions (results/models/prediction/
walk_forward_predictions.csv), so every number here is from data the model
had not seen when it made the call.

Reports win rate rather than just averages, because an average return hides
how often a strategy actually loses. It also separates the stock's own move
from the market's: a 10/10 stock that falls 2% on a day the market falls 3%
did what the model predicted (relative out-performance) while still losing
money. Both facts matter and are shown separately.

Usage
-----
    python -m evaluation.score_backtest
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.build_panel import load_panel

logger = logging.getLogger(__name__)

PRED_PATH = config.MODELS_DIR / "prediction" / "walk_forward_predictions.csv"
HORIZONS = (1, 2, 3, 5, 10)

# Round-trip cost in basis points. Indian large-cap retail cost is roughly
# 10-30bps once brokerage, STT, exchange fees and slippage are included.
COST_BPS = (0, 5, 10, 20)


def load_scored_panel() -> pd.DataFrame:
    """Join out-of-sample predictions to realised forward returns."""
    if not PRED_PATH.exists():
        raise SystemExit(f"No walk-forward predictions at {PRED_PATH}.\n"
                         "Run: python -m training.train")

    preds = pd.read_csv(PRED_PATH, parse_dates=["Date"])
    panel = load_panel()[["Date", "ticker", "Close"]].copy()
    panel = panel.sort_values(["ticker", "Date"])

    for h in HORIZONS:
        panel[f"fwd_{h}d"] = panel.groupby("ticker")["Close"].transform(
            lambda x: x.shift(-h) / x - 1)

    df = preds[["Date", "ticker", "p", "fold"]].merge(
        panel, on=["Date", "ticker"], how="left")

    # Reproduce the production score exactly: percentile within the day,
    # scaled to 1-10.
    df["score"] = df.groupby("Date")["p"].transform(
        lambda s: np.clip(np.ceil(s.rank(pct=True) * 10), 1, 10)).astype(int)

    # Market return = equal-weight universe, for separating alpha from beta.
    for h in HORIZONS:
        df[f"mkt_{h}d"] = df.groupby("Date")[f"fwd_{h}d"].transform("mean")
        df[f"excess_{h}d"] = df[f"fwd_{h}d"] - df[f"mkt_{h}d"]

    return df


def bucket_table(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Per-score-bucket outcome stats at horizon h."""
    col, exc = f"fwd_{h}d", f"excess_{h}d"
    d = df.dropna(subset=[col])
    g = d.groupby("score")
    return pd.DataFrame({
        "n": g[col].size(),
        "win_rate": g[col].apply(lambda s: (s > 0).mean()),
        "mean_ret": g[col].mean(),
        "median_ret": g[col].median(),
        "beat_market": g[exc].apply(lambda s: (s > 0).mean()),
        "mean_excess": g[exc].mean(),
        "worst": g[col].min(),
        "best": g[col].max(),
    })


def report_headline(df: pd.DataFrame):
    """The direct answer: buy 10/10, hold N days."""
    print("=" * 78)
    print("BUY A 10/10 STOCK, SELL AFTER N DAYS")
    print("=" * 78)
    print(f"{'hold':>5s} {'trades':>7s} {'win rate':>9s} {'avg ret':>9s} "
          f"{'median':>8s} {'beats mkt':>10s} {'worst':>8s}")
    for h in HORIZONS:
        col = f"fwd_{h}d"
        d = df[(df["score"] == 10)].dropna(subset=[col])
        if d.empty:
            continue
        print(f"{h:4d}d {len(d):7d} {(d[col] > 0).mean():8.1%} "
              f"{d[col].mean():+8.3%} {d[col].median():+7.3%} "
              f"{(d[f'excess_{h}d'] > 0).mean():9.1%} {d[col].min():+7.1%}")
    print("=" * 78)


def report_costs(df: pd.DataFrame):
    """Average per-trade return on 10/10 names, net of round-trip costs."""
    print("\n" + "=" * 78)
    print("SAME TRADE, NET OF COSTS  (average return per trade)")
    print("=" * 78)
    header = "hold  " + "".join(f"{f'{b}bps':>12s}" for b in COST_BPS)
    print(header)
    for h in HORIZONS:
        col = f"fwd_{h}d"
        d = df[df["score"] == 10].dropna(subset=[col])
        if d.empty:
            continue
        cells = "".join(f"{d[col].mean() - b / 10000:+11.3%} " for b in COST_BPS)
        print(f"{h:3d}d  {cells}")
    print("=" * 78)


def report_spread(df: pd.DataFrame):
    """Top score vs bottom score - does the score separate winners at all?"""
    print("\n" + "=" * 78)
    print("10/10 vs 1/10  (does the score actually separate outcomes?)")
    print("=" * 78)
    print(f"{'hold':>5s} {'10/10 avg':>11s} {'1/10 avg':>10s} {'spread':>9s} "
          f"{'10/10 win':>10s} {'1/10 win':>9s}")
    for h in HORIZONS:
        col = f"fwd_{h}d"
        hi = df[df["score"] == 10].dropna(subset=[col])[col]
        lo = df[df["score"] == 1].dropna(subset=[col])[col]
        if hi.empty or lo.empty:
            continue
        print(f"{h:4d}d {hi.mean():+10.3%} {lo.mean():+9.3%} "
              f"{hi.mean() - lo.mean():+8.3%} {(hi > 0).mean():9.1%} "
              f"{(lo > 0).mean():8.1%}")
    print("=" * 78)


def report_buckets(df: pd.DataFrame, h: int):
    print(f"\n{'=' * 78}")
    print(f"EVERY SCORE BUCKET AT {h}-DAY HOLD")
    print("=" * 78)
    t = bucket_table(df, h)
    print(f"{'score':>5s} {'n':>6s} {'win rate':>9s} {'avg ret':>9s} "
          f"{'beats mkt':>10s} {'avg excess':>11s}")
    for s, r in t.iterrows():
        print(f"{s:5d} {int(r['n']):6d} {r['win_rate']:8.1%} "
              f"{r['mean_ret']:+8.3%} {r['beat_market']:9.1%} "
              f"{r['mean_excess']:+10.3%}")
    print("=" * 78)


def report_portfolio(df: pd.DataFrame, h: int = 3):
    """
    Hold all 10/10 names equally, rebalancing every h days.

    Non-overlapping holds, so turnover matches the horizon rather than
    silently assuming daily rebalancing.
    """
    col = f"fwd_{h}d"
    d = df.dropna(subset=[col])
    rebal = np.sort(d["Date"].unique())[::h]
    held = d[d["Date"].isin(set(rebal)) & (d["score"] == 10)]
    per = held.groupby("Date")[col].mean().dropna()
    if per.empty:
        return

    mkt = d[d["Date"].isin(set(rebal))].groupby("Date")[f"mkt_{h}d"].first().dropna()
    mkt = mkt.reindex(per.index).dropna()
    per = per.reindex(mkt.index)

    ppy = 252 / h
    cum = (1 + per).prod() - 1
    cum_mkt = (1 + mkt).prod() - 1
    years = len(per) / ppy

    print("\n" + "=" * 78)
    print(f"PORTFOLIO: hold all 10/10 names, rebalance every {h} days")
    print("=" * 78)
    print(f"  periods                  {len(per)}  (~{years:.1f} years)")
    print(f"  cumulative return        {cum:+.1%}")
    print(f"  equal-weight market      {cum_mkt:+.1%}")
    print(f"  annualised               {(1 + cum) ** (1 / years) - 1:+.1%}")
    print(f"  market annualised        {(1 + cum_mkt) ** (1 / years) - 1:+.1%}")
    print(f"  win rate per period      {(per > 0).mean():.1%}")
    print(f"  beat market per period   {(per.values > mkt.values).mean():.1%}")
    print(f"  worst period             {per.min():+.1%}")
    print(f"  best period              {per.max():+.1%}")

    eq = (1 + per).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    print(f"  max drawdown             {dd:+.1%}")

    net = per - 4 * 10 / 10000
    cum_net = (1 + net).prod() - 1
    print(f"  cumulative net @10bps    {cum_net:+.1%}")
    print("=" * 78)


def main():
    df = load_scored_panel()
    print(f"\nOut-of-sample predictions: {len(df):,} rows, "
          f"{df['Date'].nunique():,} trading days, "
          f"{df['Date'].min().date()} -> {df['Date'].max().date()}\n")

    report_headline(df)
    report_spread(df)
    report_costs(df)
    report_buckets(df, 1)
    report_buckets(df, 3)
    report_portfolio(df, 3)
    report_portfolio(df, 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    main()
