"""
Tests for evaluation/stats.py.

The Monte Carlo expected-max function exists because a first, closed-form
attempt at it was wrong by 3-4x (a mis-transcribed Gumbel/Euler correction
that gave 7.24 instead of the true 1.87 for n=20) and would silently have
produced a deflated t-stat with the sign of the correction backwards. These
tests pin the function to known reference values so that class of error
cannot recur unnoticed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.stats import (
    _expected_max_of_n_normals,
    deflated_tstat,
    neutralize_score,
    newey_west_tstat,
    sample_uniqueness_weights,
)


# Reference E[max of n iid N(0,1)], accurate to 2 decimal places.
KNOWN_EXPECTED_MAX = {1: 0.0, 10: 1.539, 20: 1.867, 50: 2.249, 100: 2.508}


@pytest.mark.parametrize("n,expected", KNOWN_EXPECTED_MAX.items())
def test_expected_max_matches_known_reference_values(n, expected):
    got = _expected_max_of_n_normals(n, n_sims=300_000, seed=1)
    assert abs(got - expected) < 0.05, (
        f"n={n}: got {got:.3f}, reference is {expected:.3f} — "
        f"closed-form Gumbel approximations are easy to mis-transcribe here, "
        f"which is exactly the bug this test catches")


def test_expected_max_is_monotonically_increasing():
    vals = [_expected_max_of_n_normals(n, n_sims=100_000, seed=2)
            for n in (5, 10, 20, 50)]
    assert vals == sorted(vals)


def test_deflated_tstat_reduces_significance():
    raw = 4.78
    d1 = deflated_tstat(raw, n_trials=1)
    d20 = deflated_tstat(raw, n_trials=20)
    assert d1 == pytest.approx(raw)
    assert d20 < d1
    # Regression pin: this combination should stay clearly significant.
    assert d20 > 2.0, "known-good baseline stopped clearing the t>2 bar"


def test_deflated_tstat_never_increases_with_more_trials():
    raw = 5.0
    prev = deflated_tstat(raw, 1)
    for n in (2, 5, 10, 20, 50):
        cur = deflated_tstat(raw, n)
        assert cur <= prev
        prev = cur


def test_newey_west_matches_naive_when_uncorrelated():
    """With genuinely i.i.d. data, HAC and naive t-stats should be close."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000) * 0.02 + 0.01
    naive_t = x.mean() / x.std() * np.sqrt(len(x))
    _, nw_t = newey_west_tstat(x, lag=2)
    assert abs(naive_t - nw_t) / abs(naive_t) < 0.15


def test_newey_west_shrinks_tstat_under_positive_autocorrelation():
    """
    Construct a series with strong positive autocorrelation (as an overlapping
    forward-return IC series has) and confirm NW correctly reports lower
    effective significance than the naive i.i.d. formula.
    """
    rng = np.random.default_rng(1)
    n = 1000
    ar = np.zeros(n)
    ar[0] = rng.normal()
    phi = 0.5
    for i in range(1, n):
        ar[i] = phi * ar[i - 1] + rng.normal(scale=0.8)
    x = ar * 0.01 + 0.01  # small mean, autocorrelated noise

    naive_t = x.mean() / x.std() * np.sqrt(n)
    _, nw_t = newey_west_tstat(x, lag=2)
    assert nw_t < naive_t


def test_newey_west_handles_short_series():
    mean, t = newey_west_tstat([0.01, 0.02], lag=2)
    assert np.isfinite(mean)
    # Too few points for a meaningful t-stat: must not raise or divide by zero.
    mean0, t0 = newey_west_tstat([], lag=2)
    assert np.isnan(t0)


def test_uniqueness_weights_are_between_zero_and_one():
    df = pd.DataFrame({
        "Date": pd.bdate_range("2022-01-01", periods=40).tolist() * 2,
        "ticker": ["A"] * 40 + ["B"] * 40,
    })
    w = sample_uniqueness_weights(df, horizon=3)
    assert (w > 0).all()
    assert (w <= 1.0 + 1e-9).all()


def test_uniqueness_weights_lower_for_overlapping_labels():
    """
    Every row's label window overlaps its horizon-1 neighbours, so with a
    long enough series and horizon > 1, mean weight must be below 1 — the
    whole point of the correction.
    """
    df = pd.DataFrame({
        "Date": pd.bdate_range("2022-01-01", periods=100),
        "ticker": "A",
    })
    w = sample_uniqueness_weights(df, horizon=3)
    assert w.mean() < 1.0


def test_uniqueness_weights_equal_one_when_horizon_is_zero():
    """No overlap possible with a zero-length window."""
    df = pd.DataFrame({
        "Date": pd.bdate_range("2022-01-01", periods=20),
        "ticker": "A",
    })
    w = sample_uniqueness_weights(df, horizon=0)
    assert np.allclose(w, 1.0)


def test_neutralize_removes_pure_beta_signal():
    """
    A score built from beta plus a little independent noise should have its
    beta component almost entirely removed after neutralization.

    Deliberately not testing scores = beta exactly: that creates an exactly
    collinear regression whose residual sits at machine-epsilon noise, and
    correlating floating-point dust with anything is meaningless (confirmed
    separately: raw residual ~3e-15, fitted coefficient on beta == 1.0 to
    15 digits — the regression is correct, the exact-collinearity test of it
    was not). A small independent noise term keeps the residual on a real,
    measurable scale.
    """
    rng = np.random.default_rng(0)
    n = 400
    beta = rng.uniform(0.5, 1.5, n)
    sector = pd.Series(rng.choice(["A", "B", "C"], n))
    scores = beta + rng.normal(0, 1e-3, n)  # beta dominates, tiny own noise

    resid = neutralize_score(scores, beta, sector)
    corr = np.corrcoef(resid, beta)[0, 1]
    assert abs(corr) < 0.2, f"neutralization left beta correlation {corr:.3f}"


def test_neutralize_preserves_signal_orthogonal_to_beta():
    """A score with genuine information unrelated to beta must survive."""
    rng = np.random.default_rng(1)
    n = 300
    beta = rng.uniform(0.5, 1.5, n)
    sector = pd.Series(rng.choice(["A", "B", "C"], n))
    orthogonal_signal = rng.standard_normal(n)  # independent of beta/sector
    scores = orthogonal_signal + 0.001 * beta  # negligible beta contamination

    resid = neutralize_score(scores, beta, sector)
    corr = np.corrcoef(resid, orthogonal_signal)[0, 1]
    assert corr > 0.8, f"neutralization destroyed real signal: corr={corr:.3f}"


def test_neutralize_output_is_a_valid_percentile():
    rng = np.random.default_rng(2)
    n = 100
    scores = rng.uniform(0, 1, n)
    beta = rng.uniform(0.5, 1.5, n)
    sector = pd.Series(rng.choice(["A", "B"], n))
    resid = neutralize_score(scores, beta, sector)
    assert resid.min() >= 0 and resid.max() <= 1
    assert len(np.unique(resid)) > n * 0.9  # still a meaningful ranking


def test_neutralize_handles_missing_beta_gracefully():
    """Too much missing beta/sector data must not raise or corrupt output."""
    n = 20
    scores = np.linspace(0, 1, n)
    beta = np.full(n, np.nan)
    sector = pd.Series(["A"] * n)
    out = neutralize_score(scores, beta, sector)
    np.testing.assert_array_equal(out, scores)


def test_neutralize_single_sector_still_works():
    """No sector dummies at all (single sector) must not break the regression."""
    rng = np.random.default_rng(3)
    n = 50
    scores = rng.standard_normal(n)
    beta = rng.uniform(0.5, 1.5, n)
    sector = pd.Series(["Only"] * n)
    out = neutralize_score(scores, beta, sector)
    assert np.isfinite(out).all()
    assert out.min() >= 0 and out.max() <= 1


def test_uniqueness_weights_independent_across_tickers():
    """One ticker's overlap structure must not affect another's weights."""
    df = pd.DataFrame({
        "Date": pd.bdate_range("2022-01-01", periods=50).tolist() * 2,
        "ticker": ["A"] * 50 + ["B"] * 50,
    })
    w = sample_uniqueness_weights(df, horizon=5)
    wa = w[df["ticker"] == "A"].reset_index(drop=True)
    wb = w[df["ticker"] == "B"].reset_index(drop=True)
    pd.testing.assert_series_equal(wa, wb, check_names=False)
