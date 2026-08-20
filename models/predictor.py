"""
Prediction service for the ATLAS terminal.

Scores the whole universe at once rather than one stock at a time. That is not
an optimisation detail — the model's target is *cross-sectional* (will this
stock out-perform its peers over the next 5 sessions), so a score only has
meaning relative to the other stocks scored on the same day.

The buy score is therefore a universe percentile: 10 means top of the
universe today, 0 means bottom. It is NOT a forecast of absolute return, and
the UI should not present it as one.
"""

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "results" / "models" / "prediction"

# Score bands. The signal is relative, so the bands are universe percentiles.
LABEL_BANDS = [
    (8, "STRONG BUY"),
    (6, "BUY"),
    (4, "HOLD"),
    (2, "SELL"),
    (0, "STRONG SELL"),
]


def score_to_label(score: int) -> str:
    for threshold, label in LABEL_BANDS:
        if score >= threshold:
            return label
    return "HOLD"


class AtlasPredictor:
    """Loads the trained ensemble and scores the universe."""

    def __init__(self):
        self.models = None
        self.feature_cols = None
        self.meta = None
        self.reasoner = None
        self.loaded = False
        self._cache = {}
        self._cache_date = None
        self._lock = threading.Lock()

    # -- loading ----------------------------------------------------------

    def load(self) -> bool:
        try:
            self.models = joblib.load(MODEL_DIR / "lgbm_ensemble.pkl")
            self.feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
            self.meta = joblib.load(MODEL_DIR / "model_meta.pkl")
        except FileNotFoundError as exc:
            logger.warning("Prediction model not found (%s). "
                           "Run: python -m training.train", exc)
            self.loaded = False
            return False
        except Exception as exc:
            logger.error("Failed to load prediction model: %s", exc)
            self.loaded = False
            return False

        # The superseded train_simple.py writes a feature_cols.pkl of its own.
        # If it ran after this model was trained, the column list and the model
        # disagree — fail clearly here rather than deep inside LightGBM.
        expected = getattr(self.models[0], "n_features_in_", None)
        if expected is not None and expected != len(self.feature_cols):
            logger.error(
                "Model expects %d features but feature_cols.pkl lists %d. "
                "Artifacts are out of sync — retrain with: python -m training.train",
                expected, len(self.feature_cols))
            self.loaded = False
            return False

        try:
            from reasoning.engine import ShapReasoner
            # SHAP explains a single tree model; the ensemble members differ
            # only by seed, so the first member is representative.
            self.reasoner = ShapReasoner(self.models[0], self.feature_cols)
        except Exception as exc:
            logger.warning("Reasoning engine unavailable: %s", exc)
            self.reasoner = None

        self.loaded = True
        logger.info("Prediction model loaded: %d ensemble members, %d features, "
                    "horizon=%dd, trained %s",
                    len(self.models), len(self.feature_cols),
                    self.meta.get("horizon"), self.meta.get("trained_at", "?")[:10])
        return True

    # -- scoring ----------------------------------------------------------

    def _raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Average ensemble members' cross-sectional ranks."""
        parts = [pd.Series(m.predict_proba(X)[:, 1]).rank(pct=True).values
                 for m in self.models]
        return np.mean(parts, axis=0)

    def predict_universe(self, features: pd.DataFrame,
                         as_of=None) -> dict[str, dict]:
        """
        Score every ticker present on the latest date of `features`.

        Parameters
        ----------
        features : the full feature panel from features.predictive.build_features
        as_of    : optional date to score; defaults to the latest available

        Returns
        -------
        dict keyed by ticker with buyScore / buyLabel / buyReason / signals.
        """
        if not self.loaded:
            return {}

        as_of = as_of or features["Date"].max()
        day = features[features["Date"] == as_of].copy()
        if day.empty:
            logger.warning("No feature rows for %s", as_of)
            return {}

        with self._lock:
            if self._cache_date == as_of and self._cache:
                return self._cache

        X = day[self.feature_cols]
        # A whole-column NaN would silently skew the cross-section; median-fill
        # is safe here because features are already stationary transforms.
        X = X.fillna(X.median(numeric_only=True)).fillna(0)

        raw = self._raw_scores(X)
        # Percentile within today's universe -> 0..10.
        pct = pd.Series(raw).rank(pct=True).values
        scores = np.clip(np.ceil(pct * 10).astype(int), 0, 10)
        labels = [score_to_label(s) for s in scores]

        reasons = [{"reason": "", "signals": []}] * len(day)
        if self.reasoner is not None:
            try:
                reasons = self.reasoner.explain(X, labels)
            except Exception as exc:
                logger.warning("Reasoning failed: %s", exc)

        out = {}
        for i, ticker in enumerate(day["ticker"].values):
            out[ticker] = {
                "buyScore": int(scores[i]),
                "buyLabel": labels[i],
                "buyReason": reasons[i]["reason"] or "Model signal.",
                "signals": reasons[i]["signals"],
                "modelRank": float(pct[i]),
                "asOf": str(pd.Timestamp(as_of).date()),
                "horizonDays": int(self.meta.get("horizon", 5)),
            }

        with self._lock:
            self._cache = out
            self._cache_date = as_of
        logger.info("Scored %d tickers as of %s", len(out), pd.Timestamp(as_of).date())
        return out


# --------------------------------------------------------------------------
# Module-level singleton + universe scoring helper
# --------------------------------------------------------------------------

_predictor: AtlasPredictor | None = None
_scores_cache: dict = {}
_scores_meta: dict = {}


def get_predictor() -> AtlasPredictor:
    global _predictor
    if _predictor is None:
        _predictor = AtlasPredictor()
        _predictor.load()
    return _predictor


def compute_universe_scores() -> tuple[dict, dict]:
    """
    Build features from the cached panel and score every ticker.

    Returns (scores_by_ticker, meta). Meta carries the as-of date and model
    provenance so the UI can state what the score is actually based on.
    """
    global _scores_cache, _scores_meta

    predictor = get_predictor()
    if not predictor.loaded:
        return {}, {"error": "model not loaded"}

    try:
        from data.build_panel import load_panel
        from features.predictive import build_features

        panel = load_panel()
        feats = build_features(panel)
        scores = predictor.predict_universe(feats)
    except Exception as exc:
        logger.error("Universe scoring failed: %s", exc, exc_info=True)
        return {}, {"error": str(exc)}

    _scores_cache = scores
    _scores_meta = {
        "asOf": next(iter(scores.values()))["asOf"] if scores else None,
        "horizonDays": predictor.meta.get("horizon"),
        "trainedAt": predictor.meta.get("trained_at"),
        "nTickers": len(scores),
        "computedAt": datetime.now(timezone.utc).isoformat(),
    }
    return _scores_cache, _scores_meta


def cached_scores() -> tuple[dict, dict]:
    """Last computed scores without recomputing."""
    return _scores_cache, _scores_meta
