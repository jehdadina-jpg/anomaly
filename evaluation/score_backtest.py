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

    from features.predictive import build_features

    preds = pd.read_csv(PRED_PATH, parse_dates=["Date"])
    raw_panel = load_panel()
    panel = raw_panel[["Date", "ticker", "Close"]].copy()
    panel = panel.sort_values(["ticker", "Date"])

    for h in HORIZONS:
        panel[f"fwd_{h}d"] = panel.groupby("ticker")["Close"].transform(
            lambda x: x.shift(-h) / x - 1)

    # vol_20d for inverse-volatility position sizing (see report_portfolio).
    vol = build_features(raw_panel)[["Date", "ticker", "vol_20d"]]
    panel = panel.merge(vol, on=["Date", "ticker"], how="left")

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
    from evaluation.stats import newey_west_tstat

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

    # Significance on the average return, not the win rate. Individual
    # trades are NOT independent observations: several tickers scoring 10/10
    # on the same day share that day's market move (cross-sectional
    # correlation), and the same ticker's overlapping forward-return windows
    # create time-series correlation across nearby dates. Collapsing to one
    # observation per day and applying a Newey-West correction handles both.
    print("\nSignificance of the average return (per-day, autocorrelation-aware):")
    for h in HORIZONS:
        col = f"fwd_{h}d"
        daily = df[df["score"] == 10].dropna(subset=[col]).groupby("Date")[col].mean()
        if len(daily) < 10:
            continue
        naive_t = daily.mean() / (daily.std() + 1e-12) * np.sqrt(len(daily))
        _, nw_t = newey_west_tstat(daily.values, lag=max(h - 1, 1))
        flag = "" if nw_t > 2 else "  <- does not clear t>2"
        print(f"  {h:2d}d: naive t={naive_t:5.2f}  Newey-West t={nw_t:5.2f}{flag}")


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


def _daily_tracked_drawdown(df: pd.DataFrame, h: int, entry_dates: np.ndarray) -> float:
    """
    Max drawdown from the DAILY equity path through each hold, not just the
    return realised at each rebalance boundary.

    Sampling equity only every h days (as this function's caller originally
    did) misses any drawdown that occurs and recovers WITHIN a hold -- a
    name down 25% on day 10 of a 30-day hold that recovers to +5% by day 30
    contributes zero to a period-only drawdown calculation despite the
    -25% being real, live risk for anyone holding it. Measured on this
    project's shipped 30-day book: period-only drawdown reads -8.7%; the
    daily-tracked figure below reads -14.6%, and a block-bootstrap of the
    daily path (experiments/12_quant_diagnostics.py) puts the 95th-
    percentile tail outcome at -27.2%. Report the daily figure.
    """
    px = df.pivot_table(index="Date", columns="ticker", values="Close").sort_index()
    scorewide = df.pivot_table(index="Date", columns="ticker", values="score")
    ret = px.pct_change().shift(-1)
    dates = px.index.to_numpy()
    date_to_i = {d: i for i, d in enumerate(dates)}

    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for entry in entry_dates:
        i = date_to_i.get(entry)
        if i is None:
            continue
        sel = scorewide.iloc[i] == 10
        if sel.sum() == 0:
            continue
        end = min(i + h, len(dates))
        w.iloc[i:end] = np.where(sel.values, 1.0 / sel.sum(), 0.0)

    daily = (w * ret).sum(axis=1).dropna()
    if daily.empty:
        return float("nan")
    # Anchor the curve at 1.0 BEFORE the first return is applied. Without
    # this, cumprod's first element already has day-1's return baked in, so
    # a drop on day 1 of a hold is measured from an already-reduced value
    # instead of the true entry price -- understating exactly the kind of
    # drawdown this function exists to catch (caught by
    # tests/test_score_backtest.py::test_catches_intra_hold_drawdown_that_
    # fully_recovers, which failed by ~3pp before this line was added).
    eq = pd.concat([pd.Series([1.0]), (1 + daily).cumprod()], ignore_index=True)
    return float((eq / eq.cummax() - 1).min())


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
    dd_period_only = (eq / eq.cummax() - 1).min()
    dd_daily = _daily_tracked_drawdown(df, h, per.index.to_numpy())
    print(f"  max drawdown (daily-tracked)   {dd_daily:+.1%}")
    print(f"  max drawdown (rebalance-only)  {dd_period_only:+.1%}  "
          f"<- understates risk, see _daily_tracked_drawdown docstring")

    net = per - 4 * 10 / 10000
    cum_net = (1 + net).prod() - 1
    print(f"  cumulative net @10bps    {cum_net:+.1%}")

    # Autocorrelation-aware significance. Equal-weighting can accidentally
    # over-concentrate in a handful of volatile names on any given rebalance;
    # this doesn't fix that, but the CI at least isn't overstated because of
    # the horizon-driven autocorrelation these returns inherit from the score.
    from evaluation.stats import newey_west_tstat
    _, nw_t = newey_west_tstat(per.values, lag=max(h - 1, 1))
    naive_t = per.mean() / (per.std() + 1e-12) * np.sqrt(len(per))
    print(f"  significance: naive t={naive_t:.2f}, Newey-West t={nw_t:.2f} "
          f"(autocorrelation makes the naive figure optimistic)")
    print("=" * 78)


def report_portfolio_vol_scaled(df: pd.DataFrame, h: int = 3):
    """
    Same 10/10 book, but inverse-volatility weighted instead of equal-weighted.

    Equal-weighting silently lets whichever 10/10 names happen to be most
    volatile that period dominate the book's variance. Weighting by 1/vol_20d
    (renormalised to sum to 1 across the day's holdings) is the standard
    risk-parity fix: every name contributes similar risk, not similar capital.
    """
    col = f"fwd_{h}d"
    d = df.dropna(subset=[col, "vol_20d"])
    rebal = set(np.sort(d["Date"].unique())[::h])
    held = d[d["Date"].isin(rebal) & (d["score"] == 10) & (d["vol_20d"] > 0)]

    def _weighted_return(x):
        w = 1.0 / x["vol_20d"]
        w = w / w.sum()
        return float((w * x[col]).sum())

    per = held.groupby("Date").apply(_weighted_return).dropna()
    per_eq = held.groupby("Date")[col].mean().dropna().reindex(per.index)
    if per.empty:
        return

    ppy = 252 / h
    years = len(per) / ppy
    cum = (1 + per).prod() - 1
    cum_eq = (1 + per_eq).prod() - 1

    print("\n" + "=" * 78)
    print(f"PORTFOLIO: same 10/10 book, INVERSE-VOLATILITY weighted, {h}-day rebalance")
    print("=" * 78)
    print(f"  cumulative return (vol-scaled)   {cum:+.1%}")
    print(f"  cumulative return (equal-weight)  {cum_eq:+.1%}")
    print(f"  annualised (vol-scaled)          {(1 + cum) ** (1 / years) - 1:+.1%}")
    print(f"  return volatility (vol-scaled)   {per.std() * np.sqrt(ppy):.1%}")
    print(f"  return volatility (equal-weight) {per_eq.std() * np.sqrt(ppy):.1%}")
    print(f"  Sharpe (vol-scaled)              "
          f"{per.mean() / (per.std() + 1e-12) * np.sqrt(ppy):.2f}")
    print(f"  Sharpe (equal-weight)            "
          f"{per_eq.mean() / (per_eq.std() + 1e-12) * np.sqrt(ppy):.2f}")
    eq_curve = (1 + per).cumprod()
    dd = (eq_curve / eq_curve.cummax() - 1).min()
    print(f"  max drawdown (vol-scaled)        {dd:+.1%}")
    print("=" * 78)


def report_holding_period_sweep(df: pd.DataFrame):
    """
    Net-of-cost return by holding period -- the most consequential result in
    this file.

    The model is TRAINED on a 3-day target because that is where ranking
    signal is strongest, but that does not make 3 days the right time to
    HOLD. Holding is a separate decision governed by turnover: a 3-day hold
    rebalances 84 times a year and hands most of the edge to costs, while a
    30-day hold rebalances 8 times and keeps it.
    """
    from evaluation.stats import newey_west_tstat

    holds = [1, 2, 3, 5, 7, 10, 15, 20, 30]
    d = df.copy()
    d["rk"] = d.groupby("Date")["p"].rank(pct=True)

    # Forward returns for holds beyond those precomputed in load_scored_panel.
    panel = load_panel()[["Date", "ticker", "Close"]].sort_values(["ticker", "Date"])
    for h in holds:
        col = f"fwd_{h}d"
        if col not in d.columns:
            panel[col] = panel.groupby("ticker")["Close"].transform(
                lambda x: x.shift(-h) / x - 1)
            d = d.merge(panel[["Date", "ticker", col]], on=["Date", "ticker"], how="left")

    print("\n" + "=" * 96)
    print("HOLDING PERIOD SWEEP  (top-decile book, equal weight)")
    print("=" * 96)
    print(f"{'hold':>5s} {'rebals':>7s} {'WIN':>7s} {'gross':>9s} {'net@10bp':>10s} "
          f"{'net@20bp':>10s} {'Sharpe':>7s} {'maxDD':>9s} {'NW-t':>6s}")
    for h in holds:
        col = f"fwd_{h}d"
        x = d.dropna(subset=[col])
        rebal = set(np.sort(x["Date"].unique())[::h])
        held = x[x["Date"].isin(rebal) & (x["rk"] >= 0.90)]
        per = held.groupby("Date")[col].mean().dropna()
        if len(per) < 5:
            continue
        ppy = 252 / h
        gross = (1 + per).prod() - 1
        n10 = (1 + (per - 4 * 10 / 10000)).prod() - 1
        n20 = (1 + (per - 4 * 20 / 10000)).prod() - 1
        sharpe = per.mean() / (per.std() + 1e-12) * np.sqrt(ppy)
        # Daily-tracked, not sampled only at rebalance boundaries -- a period-
        # only drawdown misses intra-hold dips that recover by the next
        # rebalance. See _daily_tracked_drawdown's docstring for the size of
        # the gap this closes (roughly 2x at a 30-day hold).
        dsrc = x.rename(columns={col: "fwd_h"})
        dsrc["score"] = np.where(dsrc["rk"] >= 0.90, 10, 0)
        dd = _daily_tracked_drawdown(dsrc, h, per.index.to_numpy())
        _, t = newey_west_tstat(per.values, lag=1)
        print(f"{h:4d}d {len(per):7d} {(held[col] > 0).mean():6.1%} {gross:+9.1%} "
              f"{n10:+10.1%} {n20:+10.1%} {sharpe:7.2f} {dd:+8.1%} {t:6.2f}")
    print("=" * 96)
    print("Short holds look best GROSS and lose worst NET. Turnover, not signal,")
    print("is the binding constraint on this strategy. maxDD is daily-tracked,")
    print("not sampled only at rebalance points.")


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
    report_portfolio_vol_scaled(df, 3)
    report_holding_period_sweep(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    main()
