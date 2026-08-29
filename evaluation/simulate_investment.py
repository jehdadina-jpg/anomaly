"""
What actually happens to a fixed sum invested in the top-N scored stocks?

Answers the practical question directly: put a lump sum into the N
highest-scoring names, rebalance on the recommended schedule, hold for a
year -- what is it worth at the end, and how often does that go well?

Uses ONLY out-of-sample walk-forward predictions, so every position is one
the model would actually have picked at the time, with no hindsight.

Reports every available 1-year window rather than a single flattering one.
A single year is a tiny sample: the spread across start dates is usually
wider than the average, and quoting one number would hide that.

Usage
-----
    python -m evaluation.simulate_investment
    python -m evaluation.simulate_investment --capital 10000 --top-n 3
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.build_panel import load_panel
from models.predictor import RECOMMENDED_HOLD_DAYS

logger = logging.getLogger(__name__)

PRED_PATH = config.MODELS_DIR / "prediction" / "walk_forward_predictions.csv"
TRADING_DAYS_PER_YEAR = 252
COST_BPS_PER_LEG = 10          # round trip = 2 legs


def load_prices_and_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PRED_PATH.exists():
        raise SystemExit(f"No predictions at {PRED_PATH}.\nRun: python -m training.train")
    preds = pd.read_csv(PRED_PATH, parse_dates=["Date"])
    panel = load_panel()[["Date", "ticker", "Close"]]
    d = preds[["Date", "ticker", "p"]].merge(panel, on=["Date", "ticker"], how="left")
    d = d.dropna(subset=["Close"]).sort_values(["Date", "ticker"])
    px = d.pivot_table(index="Date", columns="ticker", values="Close")
    sc = d.pivot_table(index="Date", columns="ticker", values="p")
    return px, sc


def run_window(px: pd.DataFrame, sc: pd.DataFrame, start_i: int, n_days: int,
               capital: float, top_n: int, hold: int,
               cost_bps: float = COST_BPS_PER_LEG) -> dict | None:
    """
    Simulate one investment window.

    Equal-weight the top `top_n` names, rebalance every `hold` days, charge
    `cost_bps` on each leg of every switch. Capital compounds: gains are
    reinvested at the next rebalance rather than skimmed off.
    """
    dates = px.index
    end_i = min(start_i + n_days, len(dates) - 1)
    if end_i - start_i < hold * 2:
        return None

    equity = capital
    held: list[str] = []
    trades, wins = [], 0
    curve = []

    for i in range(start_i, end_i, hold):
        j = min(i + hold, end_i)
        row = sc.iloc[i].dropna()
        if row.empty:
            continue
        picks = row.nlargest(top_n).index.tolist()

        # Cost is charged only on the names actually changing hands.
        turnover = len(set(picks) ^ set(held)) / max(len(picks) * 2, 1)
        equity *= (1 - turnover * 2 * cost_bps / 10000)

        p0 = px.iloc[i][picks]
        p1 = px.iloc[j][picks]
        leg = ((p1 / p0) - 1).dropna()
        if leg.empty:
            continue

        period_ret = float(leg.mean())
        equity *= (1 + period_ret)
        trades.append(period_ret)
        wins += int(period_ret > 0)
        held = picks
        curve.append(equity)

    if not trades:
        return None

    mkt = float(px.iloc[end_i].div(px.iloc[start_i]).sub(1).mean(skipna=True))
    arr = np.array(curve)
    peak = np.maximum.accumulate(arr)
    max_dd = float((arr / peak - 1).min()) if len(arr) else 0.0

    return {
        "start": dates[start_i].date(),
        "end": dates[end_i].date(),
        "final": equity,
        "total_return": equity / capital - 1,
        "market_return": mkt,
        "n_rebalances": len(trades),
        "win_rate": wins / len(trades),
        "best_period": max(trades),
        "worst_period": min(trades),
        "max_drawdown": max_dd,
    }


def main(capital: float, top_n: int, hold: int):
    px, sc = load_prices_and_scores()
    dates = px.index
    print(f"\nOut-of-sample scores available: {dates[0].date()} -> {dates[-1].date()} "
          f"({len(dates)} trading days)")
    print(f"Simulating: Rs {capital:,.0f} into the top {top_n} scored stocks, "
          f"rebalanced every {hold} trading days, {COST_BPS_PER_LEG}bps per leg\n")

    # Every 1-year window, stepped monthly, so the spread across start dates
    # is visible rather than hidden behind one cherry-picked year.
    results = []
    step = 21
    for s in range(0, len(dates) - TRADING_DAYS_PER_YEAR - 1, step):
        r = run_window(px, sc, s, TRADING_DAYS_PER_YEAR, capital, top_n, hold)
        if r:
            results.append(r)
    if not results:
        raise SystemExit("Not enough data for a 1-year window.")

    res = pd.DataFrame(results)

    print("=" * 100)
    print(f"EVERY 1-YEAR WINDOW  (Rs {capital:,.0f} -> final value)")
    print("=" * 100)
    print(f"{'start':>12s} {'end':>12s} {'final value':>14s} {'return':>9s} "
          f"{'market':>9s} {'vs mkt':>8s} {'win rate':>9s} {'maxDD':>8s}")
    for _, r in res.iterrows():
        print(f"{str(r['start']):>12s} {str(r['end']):>12s} "
              f"{r['final']:>13,.0f} {r['total_return']:+8.1%} "
              f"{r['market_return']:+8.1%} {r['total_return'] - r['market_return']:+7.1%} "
              f"{r['win_rate']:8.0%} {r['max_drawdown']:+7.1%}")
    print("=" * 100)

    print("\n" + "=" * 100)
    print("SUMMARY ACROSS ALL 1-YEAR WINDOWS")
    print("=" * 100)
    print(f"  windows simulated            {len(res)}")
    print(f"  median final value           Rs {res['final'].median():,.0f}  "
          f"({res['total_return'].median():+.1%})")
    print(f"  average final value          Rs {res['final'].mean():,.0f}  "
          f"({res['total_return'].mean():+.1%})")
    print(f"  best year                    Rs {res['final'].max():,.0f}  "
          f"({res['total_return'].max():+.1%})")
    print(f"  worst year                   Rs {res['final'].min():,.0f}  "
          f"({res['total_return'].min():+.1%})")
    print(f"  windows that made money      {(res['total_return'] > 0).mean():.0%} "
          f"({int((res['total_return'] > 0).sum())} of {len(res)})")
    print(f"  windows that beat the market {(res['total_return'] > res['market_return']).mean():.0%} "
          f"({int((res['total_return'] > res['market_return']).sum())} of {len(res)})")
    print(f"  average rebalance win rate   {res['win_rate'].mean():.0%}")
    print(f"  worst drawdown seen          {res['max_drawdown'].min():+.1%}")
    print("=" * 100)

    # Concentration: how much does top_n matter?
    print("\n" + "=" * 100)
    print("HOW MANY STOCKS TO HOLD  (median across the same windows)")
    print("=" * 100)
    print(f"{'top N':>6s} {'median final':>14s} {'median return':>14s} "
          f"{'worst year':>12s} {'beat mkt':>9s}")
    for n in (1, 2, 3, 5, 10, 15):
        rows = [run_window(px, sc, s, TRADING_DAYS_PER_YEAR, capital, n, hold)
                for s in range(0, len(dates) - TRADING_DAYS_PER_YEAR - 1, step)]
        rows = [r for r in rows if r]
        if not rows:
            continue
        t = pd.DataFrame(rows)
        print(f"{n:6d} {t['final'].median():>13,.0f} {t['total_return'].median():+13.1%} "
              f"{t['total_return'].min():+11.1%} "
              f"{(t['total_return'] > t['market_return']).mean():8.0%}")
    print("=" * 100)
    print("Concentration raises both the ceiling and the floor risk. The worst-year")
    print("column is the honest cost of holding fewer names.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(description="Simulate investing a lump sum in the top-N scored stocks.")
    ap.add_argument("--capital", type=float, default=10000)
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--hold", type=int, default=RECOMMENDED_HOLD_DAYS)
    a = ap.parse_args()
    main(a.capital, a.top_n, a.hold)
