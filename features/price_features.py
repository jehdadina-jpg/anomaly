"""
Price-based features for anomaly detection.

Features:
- Returns (1-day, 5-day) and log-returns
- Rolling volatility (5-period, 20-period)
- Price acceleration (second derivative of price)
- Deviation from rolling VWAP
- Gap-up/gap-down magnitude at open
"""

import numpy as np
import pandas as pd


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple and log returns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Close' column with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Original df with added return columns.
    """
    out = df.copy()

    # Simple returns
    out["return_1d"] = out["Close"].pct_change(1)
    out["return_5d"] = out["Close"].pct_change(5)

    # Log returns (more symmetric, better for statistical tests)
    out["log_return_1d"] = np.log(out["Close"] / out["Close"].shift(1))

    return out


def compute_rolling_volatility(df: pd.DataFrame,
                                windows: list[int] = None) -> pd.DataFrame:
    """
    Rolling standard deviation of daily log-returns.

    Captures regime changes in volatility — sudden spikes often
    accompany manipulation activity.
    """
    windows = windows or [5, 20]
    out = df.copy()

    # Ensure log returns exist
    if "log_return_1d" not in out.columns:
        out["log_return_1d"] = np.log(out["Close"] / out["Close"].shift(1))

    for w in windows:
        out[f"rolling_vol_{w}"] = out["log_return_1d"].rolling(w).std()

    # Volatility ratio: short-term vs long-term (spikes = sudden regime change)
    if 5 in windows and 20 in windows:
        out["vol_ratio_5_20"] = out["rolling_vol_5"] / out["rolling_vol_20"]

    return out


def compute_price_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Second derivative of price — catches sudden spikes.

    First derivative = return. Second derivative = change in return.
    A large absolute value means price is accelerating (pump) or
    decelerating (dump) rapidly.
    """
    out = df.copy()

    if "return_1d" not in out.columns:
        out["return_1d"] = out["Close"].pct_change(1)

    # Second derivative: change in daily return
    out["price_acceleration"] = out["return_1d"].diff()

    # Absolute acceleration for magnitude detection
    out["abs_price_acceleration"] = out["price_acceleration"].abs()

    return out


def compute_vwap_deviation(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Deviation from rolling VWAP (Volume-Weighted Average Price).

    Since we don't have tick-level data, we approximate VWAP using
    typical price (H+L+C)/3 weighted by volume over a rolling window.

    Stocks trading far above VWAP with declining volume = potential pump.
    """
    out = df.copy()

    # Typical price * volume
    if "typical_price" not in out.columns:
        out["typical_price"] = (out["High"] + out["Low"] + out["Close"]) / 3

    tp_vol = out["typical_price"] * out["Volume"]

    # Rolling VWAP
    rolling_tp_vol = tp_vol.rolling(window).sum()
    rolling_vol = out["Volume"].rolling(window).sum()

    out["rolling_vwap"] = rolling_tp_vol / rolling_vol.replace(0, np.nan)

    # Deviation: how far current close is from rolling VWAP
    out["vwap_deviation"] = (out["Close"] - out["rolling_vwap"]) / out["rolling_vwap"]

    return out


def compute_gap_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gap-up/gap-down magnitude at open.

    gap = (Open - previous_Close) / previous_Close

    Large overnight gaps can signal news-based moves or pre-market
    manipulation (especially in illiquid small-caps).
    """
    out = df.copy()
    out["gap_magnitude"] = (out["Open"] - out["Close"].shift(1)) / out["Close"].shift(1)
    out["abs_gap_magnitude"] = out["gap_magnitude"].abs()

    return out


def compute_all_price_features(df: pd.DataFrame,
                                vol_windows: list[int] = None,
                                vwap_window: int = 20) -> pd.DataFrame:
    """
    Compute all price-based features.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain OHLCV columns with DatetimeIndex.
    vol_windows : list[int]
        Windows for rolling volatility.
    vwap_window : int
        Window for rolling VWAP.

    Returns
    -------
    pd.DataFrame
        Original df with all price features added.
    """
    out = df.copy()
    out = compute_returns(out)
    out = compute_rolling_volatility(out, windows=vol_windows)
    out = compute_price_acceleration(out)
    out = compute_vwap_deviation(out, window=vwap_window)
    out = compute_gap_magnitude(out)

    return out
