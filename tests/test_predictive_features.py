"""
Tests for the prediction feature pipeline.

The property that matters most here is causality: a feature computed at date
T must not change when future data is appended. The previous model's failure
came from exactly this class of bug going unnoticed, so it is asserted
directly rather than assumed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.predictive import build_features, feature_columns


def _synthetic_panel(n_tickers: int = 6, n_days: int = 400, seed: int = 0):
    """Deterministic multi-stock panel with the columns build_features needs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    sectors = ["Alpha", "Beta", "Gamma"]

    frames = []
    for i in range(n_tickers):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n_days)))
        spread = np.abs(rng.normal(0.01, 0.004, n_days)) * close
        frames.append(pd.DataFrame({
            "Date": dates,
            "Close": close,
            "High": close + spread,
            "Low": close - spread,
            "Open": close * (1 + rng.normal(0, 0.005, n_days)),
            "Volume": rng.integers(1e5, 5e6, n_days).astype(float),
            "ticker": f"T{i}.NS",
            "symbol": f"T{i}",
            "sector": sectors[i % len(sectors)],
            "delivery_pct": rng.uniform(0.2, 0.8, n_days),
            "delivery_qty": rng.integers(1e4, 1e6, n_days).astype(float),
            "n_trades": rng.integers(1e3, 5e4, n_days).astype(float),
            "nse_traded_qty": rng.integers(1e5, 5e6, n_days).astype(float),
            "turnover_lacs": rng.uniform(100, 10000, n_days),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def panel():
    return _synthetic_panel()


@pytest.fixture(scope="module")
def features(panel):
    return build_features(panel)


def test_no_feature_is_entirely_nan(features):
    """Guards the index-misalignment bug that silently killed ADX before."""
    cols = feature_columns(features)
    dead = [c for c in cols if features[c].notna().sum() == 0]
    assert not dead, f"features computed as all-NaN: {dead}"


def test_features_are_deterministic(panel):
    a = build_features(panel)
    b = build_features(panel)
    cols = feature_columns(a)
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_features_do_not_use_future_data(panel):
    """
    Truncating the panel must not change features on the retained dates.

    If any feature peeks forward, values at the truncation boundary will
    differ between the full and truncated runs.
    """
    cutoff = panel["Date"].sort_values().unique()[-60]
    truncated = panel[panel["Date"] <= cutoff].copy()

    full_f = build_features(panel)
    trunc_f = build_features(truncated)

    cols = feature_columns(trunc_f)
    key = ["Date", "ticker"]

    merged = full_f[full_f["Date"] <= cutoff][key + cols].merge(
        trunc_f[key + cols], on=key, suffixes=("_full", "_trunc"))
    assert len(merged) > 0

    mismatched = []
    for c in cols:
        f, t = merged[f"{c}_full"], merged[f"{c}_trunc"]
        both = f.notna() & t.notna()
        if both.sum() == 0:
            continue
        if not np.allclose(f[both], t[both], rtol=1e-9, atol=1e-12):
            mismatched.append(c)

    assert not mismatched, f"features change when future data is added: {mismatched}"


def test_cross_sectional_ranks_are_bounded(features):
    rank_cols = [c for c in features.columns if c.startswith("xs_rank_")]
    assert rank_cols
    for c in rank_cols:
        v = features[c].dropna()
        assert v.between(0, 1).all(), f"{c} outside [0, 1]"


def test_market_return_matches_universe_mean(features):
    """market_ret_1d must equal the equal-weight mean of ret_1d that day."""
    day = features[features["Date"] == features["Date"].unique()[200]]
    assert np.isclose(day["market_ret_1d"].iloc[0], day["ret_1d"].mean(),
                      rtol=1e-9, equal_nan=True)


def test_excess_return_decomposition(features):
    """excess_ret_1d is ret_1d minus the market, by construction."""
    d = features.dropna(subset=["ret_1d", "market_ret_1d", "excess_ret_1d"])
    assert np.allclose(d["excess_ret_1d"], d["ret_1d"] - d["market_ret_1d"], atol=1e-12)


def test_feature_columns_excludes_identifiers_and_targets(features):
    f = features.copy()
    f["fwd_5d"] = 0.0
    f["target_5d"] = 1
    cols = feature_columns(f)
    for banned in ("Date", "ticker", "symbol", "sector", "Close", "fwd_5d", "target_5d"):
        assert banned not in cols
