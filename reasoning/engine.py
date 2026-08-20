"""
SHAP-based reasoning for price predictions.

Turns a model's per-feature SHAP contributions into a short, numeric,
human-readable explanation of *why* a stock scored the way it did.

The narrative rules are deliberately conservative: every clause names a real
feature and its actual value, so a claim in the UI can always be traced back
to a number in the panel. Nothing here invents a story the model did not use.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_REASON_CHARS = 200


@dataclass
class Signal:
    """One feature's contribution to a single prediction."""
    feature: str
    value: float
    shap: float

    @property
    def direction(self) -> str:
        return "bullish" if self.shap > 0 else "bearish"


# How to render each feature in prose: (label, formatter).
# Features absent from this map fall back to a humanised column name.
FEATURE_RENDER = {
    "rsi_14":            ("RSI", lambda v: f"RSI {v:.0f}"),
    "rsi_2":             ("short-term RSI", lambda v: f"2d RSI {v:.0f}"),
    "adx_14":            ("trend strength", lambda v: f"ADX {v:.0f}"),
    "bollinger_z":       ("Bollinger position", lambda v: f"{v:+.1f}σ vs 20d mean"),
    "momentum_12_1":     ("12-1 momentum", lambda v: f"12m momentum {v:+.0%}"),
    "momentum_60d":      ("3m momentum", lambda v: f"3m momentum {v:+.0%}"),
    "momentum_120d":     ("6m momentum", lambda v: f"6m momentum {v:+.0%}"),
    "ret_1d":            ("1d return", lambda v: f"1d {v:+.1%}"),
    "ret_5d":            ("5d return", lambda v: f"5d {v:+.1%}"),
    "ret_20d":           ("1m return", lambda v: f"1m {v:+.1%}"),
    "reversal_5d":       ("5d reversal", lambda v: f"5d reversal {-v:+.1%}"),
    "vol_20d":           ("volatility", lambda v: f"20d vol {v:.1%}"),
    "vol_ratio_5_20":    ("volatility regime", lambda v: f"vol regime {v:.2f}x"),
    "idio_vol_60d":      ("idiosyncratic vol", lambda v: f"idio vol {v:.1%}"),
    "beta_60d":          ("beta", lambda v: f"beta {v:.2f}"),
    "atr_pct":           ("ATR", lambda v: f"ATR {v:.1%}"),
    "px_to_sma_20":      ("price vs 20d SMA", lambda v: f"{v:+.1%} vs 20d SMA"),
    "px_to_sma_50":      ("price vs 50d SMA", lambda v: f"{v:+.1%} vs 50d SMA"),
    "px_to_sma_200":     ("price vs 200d SMA", lambda v: f"{v:+.1%} vs 200d SMA"),
    "range_position_52w": ("52w range position", lambda v: f"{v:.0%} of 52w range"),
    "pct_from_52w_high": ("distance from 52w high", lambda v: f"{v:+.1%} from 52w high"),
    "pct_from_52w_low":  ("distance from 52w low", lambda v: f"{v:+.1%} above 52w low"),
    "volume_zscore_20":  ("volume", lambda v: f"volume {v:+.1f}σ"),
    "volume_ratio_20":   ("volume ratio", lambda v: f"volume {v:.1f}x avg"),
    # Amihud values span orders of magnitude, so precision follows the value.
    "amihud_illiq":      ("illiquidity",
                          lambda v: f"illiquidity {v:.3g}" if abs(v) < 1
                          else f"illiquidity {v:.2f}"),
    "delivery_pct":      ("delivery", lambda v: f"delivery {v:.0%}"),
    "delivery_pct_z20":  ("delivery vs norm", lambda v: f"delivery {v:+.1f}σ"),
    "delivery_pct_ma20": ("avg delivery", lambda v: f"20d delivery {v:.0%}"),
    "delivery_pct_trend": ("delivery trend", lambda v: f"delivery trend {v:.2f}x"),
    "delivery_volume_conviction": ("delivery conviction",
                                   lambda v: f"delivery-volume conviction {v:.2f}"),
    "avg_trade_size":    ("trade size", lambda v: f"log trade size {v:.1f}"),
    "avg_trade_size_z20": ("trade size vs norm", lambda v: f"trade size {v:+.1f}σ"),
    "n_trades_z20":      ("trade count", lambda v: f"trade count {v:+.1f}σ"),
    "excess_ret_5d":     ("excess return", lambda v: f"{v:+.1%} vs market 5d"),
    "vs_sector_ret_5d":  ("sector-relative return", lambda v: f"{v:+.1%} vs sector 5d"),
    "market_breadth":    ("market breadth", lambda v: f"breadth {v:.0%}"),
    "days_to_expiry":    ("F&O expiry", lambda v: f"{v:.0f}d to expiry"),
    "overnight_gap":     ("overnight gap", lambda v: f"gap {v:+.1%}"),
    "close_position_in_range": ("close in range", lambda v: f"closed {v:.0%} of range"),

    # --- returns over other lookbacks ---
    "ret_2d":            ("2d return", lambda v: f"2d {v:+.1%}"),
    "ret_3d":            ("3d return", lambda v: f"3d {v:+.1%}"),
    "ret_10d":           ("2w return", lambda v: f"2w {v:+.1%}"),
    "ret_60d":           ("3m return", lambda v: f"3m {v:+.1%}"),
    "log_ret_1d":        ("1d log return", lambda v: f"1d {v:+.1%}"),
    "reversal_1d":       ("1d reversal", lambda v: f"1d reversal {-v:+.1%}"),
    "intraday_ret":      ("intraday move", lambda v: f"intraday {v:+.1%}"),

    # --- volatility ---
    "vol_5d":            ("5d volatility", lambda v: f"5d vol {v:.1%}"),
    "vol_60d":           ("3m volatility", lambda v: f"3m vol {v:.1%}"),
    "vol_ratio_20_60":   ("vol trend", lambda v: f"vol trend {v:.2f}x"),
    "vol_of_vol":        ("vol instability", lambda v: f"vol-of-vol {v:.1%}"),
    "parkinson_vol_20":  ("range volatility", lambda v: f"range vol {v:.1%}"),
    "atr_14":            ("ATR", lambda v: f"ATR {v:.2f}"),
    "daily_range_pct":   ("daily range", lambda v: f"range {v:.1%}"),
    "gap_vs_range":      ("gap vs range", lambda v: f"gap {v:.2f}x range"),

    # --- trend ---
    "px_to_sma_5":       ("price vs 5d SMA", lambda v: f"{v:+.1%} vs 5d SMA"),
    "sma_5_20_cross":    ("5/20 SMA cross", lambda v: f"5v20 SMA {v:+.1%}"),
    "sma_20_50_cross":   ("20/50 SMA cross", lambda v: f"20v50 SMA {v:+.1%}"),
    "macd_hist":         ("MACD", lambda v: f"MACD {v:+.2%}"),
    "di_plus_14":        ("upward pressure", lambda v: f"+DI {v:.0f}"),
    "di_minus_14":       ("downward pressure", lambda v: f"-DI {v:.0f}"),

    # --- volume / liquidity ---
    "volume_trend_5_20": ("volume trend", lambda v: f"volume trend {v:.2f}x"),
    "dollar_volume":     ("traded value", lambda v: f"log traded value {v:.1f}"),
    "signed_volume_20":  ("signed volume", lambda v: f"signed volume {v:+.2f}"),
    "turnover_z20":      ("turnover", lambda v: f"turnover {v:+.1f}σ"),
    "delivery_pct_chg":  ("delivery change", lambda v: f"delivery {v:+.1%} d/d"),

    # --- market / sector context ---
    "market_ret_1d":     ("market move", lambda v: f"market {v:+.1%} 1d"),
    "market_ret_5d":     ("market move", lambda v: f"market {v:+.1%} 5d"),
    "excess_ret_1d":     ("excess return", lambda v: f"{v:+.1%} vs market 1d"),
    "sector_ret_1d":     ("sector move", lambda v: f"sector {v:+.1%} 1d"),
    "sector_ret_5d":     ("sector move", lambda v: f"sector {v:+.1%} 5d"),
    "vs_sector_ret_1d":  ("sector-relative return", lambda v: f"{v:+.1%} vs sector 1d"),
    "sector_rank_ret_5d": ("rank in sector", lambda v: f"{v:.0%}ile in sector"),

    # --- calendar ---
    "is_expiry_week":    ("expiry week", lambda v: "in F&O expiry week"
                          if v >= 0.5 else "outside expiry week"),
    "day_of_week":       ("weekday", lambda v: ["Mon", "Tue", "Wed", "Thu",
                                                "Fri", "Sat", "Sun"][int(v) % 7]),
    "day_of_month":      ("day of month", lambda v: f"day {v:.0f} of month"),
    "month":             ("month", lambda v: ["", "Jan", "Feb", "Mar", "Apr", "May",
                                              "Jun", "Jul", "Aug", "Sep", "Oct",
                                              "Nov", "Dec"][int(v) % 13]),
}

# Features worth surfacing preferentially when contributions are close.
# Delivery and trade-size features are India-specific and carry information
# a generic technical read would miss, so they earn a small tie-break edge.
PRIORITY_PREFIXES = ("delivery", "avg_trade_size", "n_trades", "days_to_expiry")
PRIORITY_BONUS = 1.15


def _generic(feature: str, value: float) -> str:
    """
    Last-resort rendering for a feature with no entry in FEATURE_RENDER.

    Fixed 2-decimal formatting is wrong for this domain: most features are
    returns or ratios in the 1e-3 range and would all print as "0.00".
    Precision is chosen from magnitude so the number stays informative.
    """
    name = feature.replace("_", " ")
    a = abs(value)
    if a < 0.01:
        return f"{name} {value:+.3%}" if a > 0 else f"{name} ~0"
    if a < 1:
        return f"{name} {value:+.2f}"
    if a < 1000:
        return f"{name} {value:.1f}"
    return f"{name} {value:,.0f}"


def _label_for(base: str) -> str:
    if base in FEATURE_RENDER:
        return FEATURE_RENDER[base][0]
    return base.replace("_", " ")


def _render(feature: str, value: float) -> str:
    if feature.startswith("xs_rank_"):
        return f"{_label_for(feature[len('xs_rank_'):])} in {value:.0%}ile of universe"
    if feature.startswith("xs_z_"):
        return f"{_label_for(feature[len('xs_z_'):])} {value:+.1f}σ vs universe"
    if feature in FEATURE_RENDER:
        return FEATURE_RENDER[feature][1](value)
    return _generic(feature, value)


def rank_signals(feature_names, values, shap_values, top_k: int = 3) -> list[Signal]:
    """Rank features by absolute SHAP contribution, with a domain tie-break."""
    sigs = []
    for name, val, sh in zip(feature_names, values, shap_values):
        if not np.isfinite(val) or not np.isfinite(sh):
            continue
        weight = abs(sh)
        if name.startswith(PRIORITY_PREFIXES) or any(
                name.startswith(p) for p in ("xs_rank_delivery", "xs_z_delivery")):
            weight *= PRIORITY_BONUS
        sigs.append((weight, Signal(name, float(val), float(sh))))

    sigs.sort(key=lambda x: -x[0])
    return [s for _, s in sigs[:top_k]]


def build_reason(signals: list[Signal], label: str,
                 max_chars: int = MAX_REASON_CHARS) -> str:
    """Compose a short narrative from the ranked signals."""
    if not signals:
        return "No dominant signal; score near universe median."

    supporting = [s for s in signals if s.direction ==
                  ("bullish" if label in ("STRONG BUY", "BUY") else "bearish")]
    lead = supporting if supporting else signals

    parts = [_render(s.feature, s.value) for s in lead[:3]]
    text = "; ".join(parts)

    opposing = [s for s in signals if s not in lead]
    if opposing:
        counter = _render(opposing[0].feature, opposing[0].value)
        candidate = f"{text}. Offset by {counter}."
        if len(candidate) <= max_chars:
            text = candidate
        else:
            text += "."
    else:
        text += "."

    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip(" ;,") + "."
    return text


class ShapReasoner:
    """
    Wraps a tree model with a cached SHAP explainer.

    The explainer is built once and reused; constructing it per prediction is
    the single most expensive thing you can do in this path.
    """

    def __init__(self, model, feature_names: list[str]):
        self.model = model
        self.feature_names = list(feature_names)
        self._explainer = None

    @property
    def explainer(self):
        if self._explainer is None:
            import shap
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def explain(self, X: pd.DataFrame, labels: list[str]) -> list[dict]:
        """Return one {'reason', 'signals'} dict per row of X."""
        try:
            sv = self.explainer.shap_values(X[self.feature_names])
        except Exception as exc:                       # pragma: no cover
            logger.warning("SHAP failed (%s); falling back to gain importance.", exc)
            return self._fallback(X, labels)

        # Binary tree models return either (n, f) or a 2-element list/3-D array.
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]

        out = []
        for i, label in enumerate(labels):
            sigs = rank_signals(self.feature_names, X[self.feature_names].iloc[i].values, sv[i])
            out.append({
                "reason": build_reason(sigs, label),
                "signals": [
                    {"feature": s.feature, "value": s.value,
                     "contribution": s.shap, "direction": s.direction}
                    for s in sigs
                ],
            })
        return out

    def _fallback(self, X: pd.DataFrame, labels: list[str]) -> list[dict]:
        """Global importance instead of per-row attribution, if SHAP is absent."""
        imp = getattr(self.model, "feature_importances_", None)
        if imp is None:
            return [{"reason": "Model signal; per-feature attribution unavailable.",
                     "signals": []} for _ in labels]
        top = pd.Series(imp, index=self.feature_names).nlargest(3).index
        out = []
        for i, _ in enumerate(labels):
            parts = [_render(f, float(X[f].iloc[i])) for f in top
                     if f in X.columns and np.isfinite(X[f].iloc[i])]
            out.append({"reason": ("; ".join(parts) + ".") if parts
                        else "Model signal.", "signals": []})
        return out
