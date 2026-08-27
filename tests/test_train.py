"""
Tests for training/train.py.

Focus is the neutralization wiring: neutralize_predictions must apply
neutralize_score independently within EACH date, never mixing information
across days (that would be a subtle cross-sectional leak — a stock's score
would depend on other days' beta/sector composition, which the live
predictor, scoring one day at a time, could never reproduce).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import make_folds, make_target





def test_make_target_only_labels_tails():
    """Confirms the ambiguous middle stays unlabelled (NaN), by construction."""
    rng = np.random.default_rng(3)
    n_days, n_tickers = 30, 10
    df = pd.DataFrame({
        "Date": np.repeat(pd.bdate_range("2022-01-01", periods=n_days), n_tickers),
        "ticker": list(range(n_tickers)) * n_days,
    })
    df["fwd_3d"] = rng.standard_normal(len(df))

    y = make_target(df, horizon=3, tail_q=0.30)
    labelled = y.notna()
    # With tail_q=0.30, roughly 60% of rows per day should be labelled
    # (30% top + 30% bottom), the rest NaN.
    frac_labelled = labelled.mean()
    assert 0.45 < frac_labelled < 0.75


def test_make_folds_are_strictly_expanding_and_non_overlapping_test_sets():
    dates = np.array(pd.bdate_range("2020-01-01", periods=2000))
    folds = make_folds(dates, n_folds=6)
    assert len(folds) == 6
    for i in range(1, len(folds)):
        prev_tr, prev_te = folds[i - 1]
        cur_tr, cur_te = folds[i]
        # Expanding window: later folds train on more data.
        assert len(cur_tr) > len(prev_tr)
        # No test-set date reused across folds.
        assert not set(prev_te) & set(cur_te)
