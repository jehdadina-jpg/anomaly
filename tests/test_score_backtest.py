"""
Tests for evaluation/score_backtest.py.

The one property that matters: max drawdown must be tracked from the DAILY
equity path through a hold, not sampled only at rebalance boundaries. A
period-only calculation was shipped for a while and understated the 30-day
book's real drawdown by roughly 60% (-8.7% reported vs -14.0% actual) because
it cannot see a dip that occurs and recovers within a single hold. These
tests construct a case where the true intra-hold drawdown is known exactly,
so a regression back to period-only sampling is caught immediately rather
than silently understating risk again.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.score_backtest import _daily_tracked_drawdown


def _panel(prices: dict[str, list[float]], hold: int) -> pd.DataFrame:
    """Build a minimal (Date, ticker, Close, score) frame for one ticker series."""
    n = len(next(iter(prices.values())))
    dates = pd.bdate_range("2022-01-03", periods=n)
    frames = []
    for tk, closes in prices.items():
        frames.append(pd.DataFrame({
            "Date": dates,
            "ticker": tk,
            "Close": closes,
            "score": 10,
        }))
    return pd.concat(frames, ignore_index=True)


def test_catches_intra_hold_drawdown_that_fully_recovers():
    """
    A name that falls 40% mid-hold and fully recovers by the rebalance date
    has a real max drawdown of -40%, even though its PERIOD return is 0%.
    Sampling equity only at the rebalance boundary would report -0%.
    """
    hold = 10
    # Day 0 = 100 (entry). Falls to 60 by day 5 (-40%). Recovers to 100 by
    # day 10 (period return = 0%, but the low was real).
    closes = [100, 95, 85, 75, 65, 60, 70, 80, 90, 100, 100]
    df = _panel({"A": closes}, hold)
    entries = np.array([df["Date"].iloc[0]])

    dd = _daily_tracked_drawdown(df, hold, entries)
    assert dd == pytest.approx(-0.40, abs=0.02), (
        f"expected roughly -40% drawdown, got {dd:.1%} -- "
        f"a period-only calculation would report ~0%")


def test_matches_period_return_when_path_is_monotonic():
    """When a name falls monotonically to its period-end low, daily-tracked
    and period-only drawdown must agree -- no bug should change this case."""
    hold = 5
    closes = [100, 95, 90, 85, 80, 75]   # smooth decline, no recovery
    df = _panel({"A": closes}, hold)
    entries = np.array([df["Date"].iloc[0]])

    dd = _daily_tracked_drawdown(df, hold, entries)
    expected_period_return = closes[-1] / closes[0] - 1
    assert dd == pytest.approx(expected_period_return, abs=0.02)


def test_flat_book_has_zero_drawdown():
    hold = 5
    closes = [100, 100, 100, 100, 100, 100]
    df = _panel({"A": closes}, hold)
    entries = np.array([df["Date"].iloc[0]])
    dd = _daily_tracked_drawdown(df, hold, entries)
    assert dd == pytest.approx(0.0, abs=1e-9)


def test_multiple_holdings_use_the_worse_drawdown():
    """
    Two names held together: one crashes intra-hold, the other is flat. The
    equal-weight book's drawdown should reflect the crash (halved by
    equal weighting), not be diluted away to zero.
    """
    hold = 6
    crash = [100, 60, 60, 60, 60, 100, 100]     # -40% then fully recovers
    flat = [100, 100, 100, 100, 100, 100, 100]
    df = _panel({"A": crash, "B": flat}, hold)
    entries = np.array([df["Date"].iloc[0]])

    dd = _daily_tracked_drawdown(df, hold, entries)
    # Equal-weight book return at the crash's trough: 0.5*(-0.40) + 0.5*0 = -20%.
    assert dd == pytest.approx(-0.20, abs=0.02)


def test_sequential_entries_track_drawdown_across_holds():
    """Two consecutive holds; a crash in the second hold must still register."""
    hold = 4
    # hold 1 (days 0-3): flat. hold 2 (days 4-7): crashes then recovers.
    closes = [100, 100, 100, 100, 100, 70, 100, 100, 100]
    df = _panel({"A": closes}, hold)
    entries = np.array([df["Date"].iloc[0], df["Date"].iloc[4]])

    dd = _daily_tracked_drawdown(df, hold, entries)
    assert dd < -0.20, f"expected the second-hold crash to register, got {dd:.1%}"


def test_zero_drawdown_when_nothing_is_ever_selected():
    """
    No name ever scores 10 -- the book stays entirely in cash. Zero
    allocation means zero realised return every day, which is a
    well-defined 0% drawdown (never having been invested is not the same
    as an undefined result).
    """
    hold = 5
    closes = [100, 101, 102, 103, 104, 105]
    df = _panel({"A": closes}, hold)
    df["score"] = 5     # never a 10/10
    entries = np.array([df["Date"].iloc[0]])
    dd = _daily_tracked_drawdown(df, hold, entries)
    assert dd == pytest.approx(0.0, abs=1e-9)


def test_nan_when_price_data_is_entirely_missing():
    """A genuinely empty/unusable frame (no price data at all) is the one
    case that should be reported as undefined rather than as 0%."""
    df = pd.DataFrame({"Date": [], "ticker": [], "Close": [], "score": []})
    dd = _daily_tracked_drawdown(df, 5, np.array([]))
    assert np.isnan(dd)
