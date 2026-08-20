"""
Tests for the SHAP reasoning engine.

The contract the UI depends on: every explanation names real features with
real values, stays inside the character budget, and never invents a signal
the model did not use.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from reasoning.engine import (
    MAX_REASON_CHARS,
    FEATURE_RENDER,
    ShapReasoner,
    Signal,
    build_reason,
    rank_signals,
    _generic,
    _render,
)

FEATURES = ["rsi_14", "delivery_pct_z20", "momentum_60d", "volume_zscore_20",
            "beta_60d", "xs_rank_ret_5d", "adx_14"]


@pytest.fixture(scope="module")
def fitted():
    import lightgbm as lgb
    rng = np.random.default_rng(0)
    n = 800
    X = pd.DataFrame({
        "rsi_14": rng.uniform(20, 80, n),
        "delivery_pct_z20": rng.normal(0, 1, n),
        "momentum_60d": rng.normal(0, 0.1, n),
        "volume_zscore_20": rng.normal(0, 1, n),
        "beta_60d": rng.uniform(0.5, 1.5, n),
        "xs_rank_ret_5d": rng.uniform(0, 1, n),
        "adx_14": rng.uniform(10, 50, n),
    })
    y = ((X["momentum_60d"] * 8 + X["delivery_pct_z20"] * 1.5
          + rng.normal(0, 0.5, n)) > 0).astype(int)
    model = lgb.LGBMClassifier(n_estimators=60, num_leaves=7, verbose=-1).fit(X, y)
    return model, X


def test_reason_respects_character_budget(fitted):
    model, X = fitted
    r = ShapReasoner(model, FEATURES)
    labels = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"] * 4
    out = r.explain(X.head(len(labels)), labels)
    assert len(out) == len(labels)
    for o in out:
        assert 0 < len(o["reason"]) <= MAX_REASON_CHARS


def test_signals_are_ranked_by_absolute_contribution():
    sigs = rank_signals(
        ["rsi_14", "beta_60d", "adx_14"],
        [50.0, 1.0, 25.0],
        [0.01, -0.9, 0.4],
        top_k=3,
    )
    # beta has the largest magnitude despite being negative.
    assert sigs[0].feature == "beta_60d"
    assert sigs[1].feature == "adx_14"


def test_india_specific_features_get_tie_break_priority():
    """Delivery features outrank a technical feature at equal magnitude."""
    sigs = rank_signals(
        ["rsi_14", "delivery_pct_z20"],
        [50.0, 1.2],
        [0.30, 0.30],
        top_k=2,
    )
    assert sigs[0].feature == "delivery_pct_z20"


def test_non_finite_values_are_skipped():
    sigs = rank_signals(
        ["rsi_14", "beta_60d", "adx_14"],
        [np.nan, 1.0, 25.0],
        [0.9, 0.5, np.inf],
        top_k=3,
    )
    assert [s.feature for s in sigs] == ["beta_60d"]


def test_signal_direction_follows_contribution_sign():
    assert Signal("rsi_14", 50.0, 0.3).direction == "bullish"
    assert Signal("rsi_14", 50.0, -0.3).direction == "bearish"


def test_empty_signals_produce_a_safe_message():
    text = build_reason([], "HOLD")
    assert text and len(text) <= MAX_REASON_CHARS


def test_opposing_signal_is_reported_as_an_offset():
    sigs = [
        Signal("momentum_60d", 0.08, 0.5),
        Signal("beta_60d", 1.3, -0.4),
    ]
    text = build_reason(sigs, "BUY")
    assert "Offset by" in text


def test_every_model_feature_has_a_renderer():
    """
    Guards against a new feature silently rendering as generic text.

    Skips when no trained model is present, so the suite still runs on a
    fresh checkout.
    """
    import joblib
    path = (Path(__file__).parent.parent / "results" / "models"
            / "prediction" / "feature_cols.pkl")
    if not path.exists():
        pytest.skip("no trained model available")

    missing = []
    for f in joblib.load(path):
        base = (f[len("xs_rank_"):] if f.startswith("xs_rank_")
                else f[len("xs_z_"):] if f.startswith("xs_z_") else f)
        if base not in FEATURE_RENDER:
            missing.append(f)
    assert not missing, f"features without a prose renderer: {missing}"


@pytest.mark.parametrize("value", [0.0001, 0.005, 0.5, 12.3, 45000.0])
def test_generic_renderer_never_collapses_to_zero(value):
    """Small feature values must stay legible, not print as '0.00'."""
    text = _generic("some_feature", value)
    assert "0.00 " not in text and not text.endswith("0.00")


def test_cross_sectional_features_render_as_percentiles():
    assert "%ile of universe" in _render("xs_rank_ret_5d", 0.87)
    assert "vs universe" in _render("xs_z_delivery_pct", 1.4)


def test_fallback_used_when_shap_unavailable(fitted, monkeypatch):
    """If SHAP raises, explanations still come back rather than blowing up."""
    model, X = fitted
    r = ShapReasoner(model, FEATURES)

    class Boom:
        def shap_values(self, *_a, **_k):
            raise RuntimeError("shap unavailable")

    r._explainer = Boom()
    out = r.explain(X.head(3), ["BUY", "HOLD", "SELL"])
    assert len(out) == 3
    for o in out:
        assert o["reason"]
