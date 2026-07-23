"""
Pattern-based features for anomaly detection — the India-specific ones.

These features capture manipulation patterns unique to or particularly
prevalent in the Indian market (retail-dominated, circuit-limit regime):

- Intraday reversal magnitude (pump-and-dump intraday pattern)
- Days-to-expiry proximity (F&O pinning hypothesis)
- Circuit-limit proximity (SEBI's own manipulation flag)
- Consecutive circuit days (repeated limit-hitting)
- Return autocorrelation (unnaturally smooth price walks)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def compute_intraday_reversal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intraday reversal magnitude.

    Measures how much of an intraday move reverses by close.
    Pump-and-dump schemes often pump the price intraday, then the
    price dumps back down before close.

    For up-days: reversal = (High - Close) / (High - Low)
        High reversal = price hit a high but couldn't hold it
    For down-days: reversal = (Close - Low) / (High - Low)
        High reversal = price hit a low but recovered

    Values near 1.0 = strong reversal; near 0.0 = no reversal.
    """
    out = df.copy()

    daily_range = out["High"] - out["Low"]
    # Avoid division by zero for zero-range days
    daily_range = daily_range.replace(0, np.nan)

    # Determine up-day vs down-day
    is_up_day = out["Close"] >= out["Open"]

    # Reversal for up-days: price went up intraday but how much came back?
    up_reversal = (out["High"] - out["Close"]) / daily_range
    # Reversal for down-days: price went down but how much recovered?
    down_reversal = (out["Close"] - out["Low"]) / daily_range

    out["intraday_reversal"] = np.where(is_up_day, up_reversal, down_reversal)

    # Upper wick ratio (regardless of up/down): how much of the range is wick
    out["upper_wick_ratio"] = (out["High"] - np.maximum(out["Open"], out["Close"])) / daily_range
    out["lower_wick_ratio"] = (np.minimum(out["Open"], out["Close"]) - out["Low"]) / daily_range

    # Rolling average reversal (persistent reversals = sustained manipulation)
    out["reversal_5d_avg"] = out["intraday_reversal"].rolling(5).mean()

    return out


def compute_expiry_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Days-to-expiry proximity for F&O-eligible stocks.

    Tests the "pinning" hypothesis: stocks with heavy derivative interest
    tend to get "pinned" near the strike price with highest open interest
    as expiry approaches.

    Feature: calendar days to next monthly F&O expiry.
    Values 0-3 indicate very close to expiry (highest pinning pressure).
    """
    out = df.copy()

    # Check if this stock is F&O eligible
    is_fno = False
    if "is_fno" in out.columns:
        is_fno = out["is_fno"].any()
    elif "ticker" in out.columns:
        ticker = out["ticker"].iloc[0] if len(out) > 0 else ""
        is_fno = ticker in config.FNO_ELIGIBLE

    if not is_fno:
        out["days_to_expiry"] = np.nan
        out["is_expiry_day"] = 0
        out["is_expiry_week"] = 0
        return out

    # For each date, find the next expiry
    expiry_dates_sorted = sorted(config.FNO_EXPIRY_DATES)

    def _days_to_next_expiry(dt):
        dt_date = dt.date() if hasattr(dt, 'date') else dt
        for exp in expiry_dates_sorted:
            if exp >= dt_date:
                return (exp - dt_date).days
        return np.nan  # No future expiry found

    out["days_to_expiry"] = out.index.map(_days_to_next_expiry)

    # Binary flags
    out["is_expiry_day"] = (out["days_to_expiry"] == 0).astype(int)
    out["is_expiry_week"] = (out["days_to_expiry"] <= 5).astype(int)

    # Proximity score (higher = closer to expiry, peaks at 1.0 on expiry day)
    out["expiry_proximity_score"] = np.exp(-out["days_to_expiry"] / 5.0)

    return out


def compute_circuit_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Circuit-limit proximity features.

    How close the closing price is to the daily upper/lower circuit limit.
    Repeated circuit-hitting is a known manipulation flag that SEBI
    actively monitors.

    Circuit limits in India:
    - Non-F&O stocks: 2%, 5%, 10%, or 20% bands
    - F&O stocks: dynamic 10% operating range (flexing)

    Since we don't have exact circuit limits per day, we approximate
    using the previous close and the stock's circuit band percentage.
    """
    out = df.copy()

    # Get circuit band for this ticker
    ticker = ""
    if "ticker" in out.columns and len(out) > 0:
        ticker = out["ticker"].iloc[0]

    circuit_pct = config.TICKER_CIRCUIT_BAND.get(
        ticker, config.CIRCUIT_BANDS["default"]
    )

    # Approximate circuit limits
    prev_close = out["Close"].shift(1)
    upper_circuit = prev_close * (1 + circuit_pct)
    lower_circuit = prev_close * (1 - circuit_pct)

    # Proximity to upper circuit (0 = far, 1 = at circuit)
    out["upper_circuit_proximity"] = (
        (out["Close"] - prev_close) / (upper_circuit - prev_close)
    ).clip(0, 1)

    # Proximity to lower circuit
    out["lower_circuit_proximity"] = (
        (prev_close - out["Close"]) / (prev_close - lower_circuit)
    ).clip(0, 1)

    # Maximum circuit proximity (either direction)
    out["circuit_proximity"] = np.maximum(
        out["upper_circuit_proximity"],
        out["lower_circuit_proximity"]
    )

    # Binary: close is within 1% of circuit
    out["near_circuit_flag"] = (out["circuit_proximity"] > 0.90).astype(int)

    return out


def compute_consecutive_circuit_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count consecutive days closing at or near the circuit limit.

    Repeated circuit-limit hitting is one of SEBI's primary
    manipulation surveillance triggers. A stock hitting upper circuit
    for 5+ consecutive days is almost certainly under manipulation
    (SMS/WhatsApp tip-driven pump).
    """
    out = df.copy()

    if "near_circuit_flag" not in out.columns:
        out = compute_circuit_proximity(out)

    # Count consecutive 1s in near_circuit_flag
    # Use cumsum trick: reset counter when flag goes to 0
    flag = out["near_circuit_flag"]
    groups = (flag != flag.shift()).cumsum()
    out["consecutive_circuit_days"] = flag.groupby(groups).cumsum()

    return out


def compute_return_autocorrelation(df: pd.DataFrame,
                                    windows: list[int] = None) -> pd.DataFrame:
    """
    Rolling autocorrelation of returns.

    Unnaturally smooth/monotonic price walks (very high positive
    autocorrelation) can indicate manipulation vs. natural random walk.
    Natural stock returns should have near-zero autocorrelation.

    Significant positive autocorrelation = returns are trending
    (could be natural momentum or artificial price support).
    Significant negative autocorrelation = mean-reverting
    (could be natural or pump-dump-pump cycles).
    """
    windows = windows or [5, 20]
    out = df.copy()

    if "return_1d" not in out.columns:
        out["return_1d"] = out["Close"].pct_change(1)

    for w in windows:
        out[f"return_autocorr_{w}"] = out["return_1d"].rolling(w).apply(
            lambda x: x.autocorr(lag=1) if len(x.dropna()) > 2 else np.nan,
            raw=False,
        )

    return out


def compute_all_pattern_features(df: pd.DataFrame,
                                  autocorr_windows: list[int] = None) -> pd.DataFrame:
    """
    Compute all pattern-based features.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain OHLCV columns with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Original df with all pattern features added.
    """
    out = df.copy()
    out = compute_intraday_reversal(out)
    out = compute_expiry_proximity(out)
    out = compute_circuit_proximity(out)
    out = compute_consecutive_circuit_days(out)
    out = compute_return_autocorrelation(out, windows=autocorr_windows)

    return out
