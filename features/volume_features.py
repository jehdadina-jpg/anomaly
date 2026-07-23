"""
Volume-based features for anomaly detection.

Features:
- Volume z-score vs. 20-day rolling average (catches sudden spikes)
- Volume-price divergence (price up but volume declining = classic pump)
- Delivery percentage analysis (India-specific — low delivery + price spike = pump)
- Volume spike binary flag
"""

import numpy as np
import pandas as pd


def compute_volume_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Volume z-score relative to rolling mean/std.

    A z-score > 3 means volume is 3 standard deviations above recent
    average — a strong signal of unusual activity.
    """
    out = df.copy()

    rolling_mean = out["Volume"].rolling(window).mean()
    rolling_std = out["Volume"].rolling(window).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    out["volume_zscore_20"] = (out["Volume"] - rolling_mean) / rolling_std

    # Also compute the raw volume ratio for interpretability
    out["volume_ratio_20"] = out["Volume"] / rolling_mean.replace(0, np.nan)

    return out


def compute_volume_price_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volume-price divergence detection.

    Classic pump-and-dump signatures:
    - Price going UP but volume DECLINING (pump losing momentum)
    - Price going DOWN but volume SPIKING (dump / panic selling)

    Returns a signed divergence score:
    - Positive = price and volume moving in opposite directions
    - Negative = price and volume moving together (normal)
    """
    out = df.copy()

    if "return_1d" not in out.columns:
        out["return_1d"] = out["Close"].pct_change(1)

    vol_change = out["Volume"].pct_change(1)

    # Sign-based divergence: +1 when signs differ, -1 when same
    price_sign = np.sign(out["return_1d"])
    vol_sign = np.sign(vol_change)
    out["vol_price_sign_divergence"] = -(price_sign * vol_sign)

    # Magnitude-based divergence: product of |return| and |vol_change|,
    # signed by whether they diverge
    magnitude = out["return_1d"].abs() * vol_change.abs()
    out["vol_price_divergence"] = out["vol_price_sign_divergence"] * magnitude

    # Rolling average divergence (smoothed signal)
    out["vol_price_div_5d"] = out["vol_price_divergence"].rolling(5).mean()

    return out


def compute_delivery_features(df: pd.DataFrame,
                               low_threshold: float = 0.30) -> pd.DataFrame:
    """
    Delivery percentage analysis — INDIA-SPECIFIC feature.

    In the Indian market, the delivery percentage (shares actually delivered
    to demat accounts / total traded) is publicly available via NSE bhavcopy.
    This feature does NOT exist in US market data.

    Key insight: Low delivery % + price spike is a well-known pump-and-dump
    tell that SEBI actually monitors. It means lots of speculative intraday
    trading without genuine buying interest.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'delivery_pct' column (from bhavcopy merge)
    low_threshold : float
        Percentile threshold for "low" delivery (default 0.30 = 30th percentile)
    """
    out = df.copy()

    if "delivery_pct" not in out.columns:
        # No delivery data available — return NaN features
        out["delivery_pct_zscore"] = np.nan
        out["low_delivery_pump_flag"] = 0
        out["delivery_pct_change"] = np.nan
        return out

    # Z-score of delivery percentage
    del_mean = out["delivery_pct"].rolling(20).mean()
    del_std = out["delivery_pct"].rolling(20).std().replace(0, np.nan)
    out["delivery_pct_zscore"] = (out["delivery_pct"] - del_mean) / del_std

    # Flag: low delivery + positive return (classic pump signal)
    # Use rolling quantile for adaptive threshold
    rolling_q30 = out["delivery_pct"].rolling(60).quantile(low_threshold)
    is_low_delivery = out["delivery_pct"] < rolling_q30

    if "return_1d" not in out.columns:
        out["return_1d"] = out["Close"].pct_change(1)

    is_positive_return = out["return_1d"] > 0.01  # > 1% return
    out["low_delivery_pump_flag"] = (is_low_delivery & is_positive_return).astype(int)

    # Delivery percentage change (sudden drops are suspicious)
    out["delivery_pct_change"] = out["delivery_pct"].pct_change(1)

    return out


def compute_volume_spike(df: pd.DataFrame,
                          threshold: float = 3.0,
                          window: int = 20) -> pd.DataFrame:
    """
    Binary flag for volume spikes.

    A volume spike is when current volume exceeds `threshold` times
    the rolling average. Used as both a feature and for filtering
    "interesting" days for detailed analysis.
    """
    out = df.copy()

    rolling_mean = out["Volume"].rolling(window).mean()
    out["volume_spike_flag"] = (
        out["Volume"] > threshold * rolling_mean
    ).astype(int)

    # Also compute how many spikes in the last N days (clustering detection)
    out["volume_spikes_5d"] = out["volume_spike_flag"].rolling(5).sum()
    out["volume_spikes_20d"] = out["volume_spike_flag"].rolling(20).sum()

    return out


def compute_all_volume_features(df: pd.DataFrame,
                                 zscore_window: int = 20,
                                 spike_threshold: float = 3.0,
                                 delivery_threshold: float = 0.30) -> pd.DataFrame:
    """
    Compute all volume-based features.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain OHLCV columns, optionally 'delivery_pct'.

    Returns
    -------
    pd.DataFrame
        Original df with all volume features added.
    """
    out = df.copy()
    out = compute_volume_zscore(out, window=zscore_window)
    out = compute_volume_price_divergence(out)
    out = compute_delivery_features(out, low_threshold=delivery_threshold)
    out = compute_volume_spike(out, threshold=spike_threshold, window=zscore_window)

    return out
