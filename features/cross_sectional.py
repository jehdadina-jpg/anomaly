"""
Cross-sectional features — compare each stock against its sector peers.

Features:
- Beta-adjusted residual return (isolate stock-specific anomalies)
- Volume z-score rank within sector (is this stock's activity abnormal vs peers?)
- Return rank within sector

These features answer: "Is this stock behaving oddly relative to its peers,
or is the whole sector moving?" — critical for separating genuine
stock-specific anomalies from market-wide moves.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def compute_beta_adjusted_residual(stock_df: pd.DataFrame,
                                    index_df: pd.DataFrame,
                                    window: int = 60) -> pd.DataFrame:
    """
    Beta-adjusted residual return using rolling OLS.

    residual = stock_return - beta * index_return - alpha

    A large positive residual means the stock moved up more than its
    beta-adjusted fair share — possible stock-specific catalyst or manipulation.
    A large negative residual means unexplained underperformance.

    Parameters
    ----------
    stock_df : pd.DataFrame
        Single stock's data with 'return_1d' column
    index_df : pd.DataFrame
        Market index data (Nifty 50) with 'Close' column
    window : int
        Rolling OLS window (default 60 trading days ≈ 3 months)

    Returns
    -------
    pd.DataFrame
        stock_df with added beta, alpha, residual columns
    """
    out = stock_df.copy()

    # Compute index returns
    idx_returns = index_df["Close"].pct_change(1)
    idx_returns.name = "index_return"

    # Align dates
    if "return_1d" not in out.columns:
        out["return_1d"] = out["Close"].pct_change(1)

    # Join on index
    combined = out[["return_1d"]].join(idx_returns, how="inner")
    combined = combined.dropna()

    if len(combined) < window:
        out["beta"] = np.nan
        out["alpha"] = np.nan
        out["beta_adjusted_residual"] = np.nan
        return out

    # Rolling OLS: stock_return = alpha + beta * index_return + epsilon
    betas = []
    alphas = []
    residuals = []
    dates = []

    for i in range(window, len(combined)):
        y = combined["return_1d"].iloc[i - window:i].values
        x = combined["index_return"].iloc[i - window:i].values

        # Simple OLS via numpy
        try:
            slope, intercept, _, _, _ = stats.linregress(x, y)
            betas.append(slope)
            alphas.append(intercept)

            # Current-day residual
            curr_y = combined["return_1d"].iloc[i]
            curr_x = combined["index_return"].iloc[i]
            residual = curr_y - (intercept + slope * curr_x)
            residuals.append(residual)
            dates.append(combined.index[i])
        except Exception:
            betas.append(np.nan)
            alphas.append(np.nan)
            residuals.append(np.nan)
            dates.append(combined.index[i])

    # Create series and join back
    beta_series = pd.Series(betas, index=dates, name="beta")
    alpha_series = pd.Series(alphas, index=dates, name="alpha")
    residual_series = pd.Series(residuals, index=dates, name="beta_adjusted_residual")

    out = out.join(beta_series, how="left")
    out = out.join(alpha_series, how="left")
    out = out.join(residual_series, how="left")

    # Absolute residual for anomaly magnitude
    out["abs_residual"] = out["beta_adjusted_residual"].abs()

    return out


def compute_sector_ranks(panel_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute within-sector percentile ranks for volume and returns.

    For each stock-day, rank it against all stocks in the same sector
    on the same day. High rank = more extreme behavior relative to peers.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Combined panel with all stocks (must have 'ticker', 'sector',
        'volume_zscore_20', 'return_1d' columns)

    Returns
    -------
    pd.DataFrame
        Panel with added rank columns
    """
    out = panel_df.copy()

    if "sector" not in out.columns:
        # Can't compute sector ranks without sector assignment
        out["volume_zscore_sector_rank"] = np.nan
        out["return_sector_rank"] = np.nan
        return out

    # Group by (date, sector) and compute percentile ranks
    # Reset index to make Date a column for groupby
    out_reset = out.reset_index()

    if "volume_zscore_20" in out.columns:
        out_reset["volume_zscore_sector_rank"] = out_reset.groupby(
            [out_reset.index.name or "Date", "sector"]
        )["volume_zscore_20"].rank(pct=True)
    else:
        out_reset["volume_zscore_sector_rank"] = np.nan

    if "return_1d" in out.columns:
        out_reset["return_sector_rank"] = out_reset.groupby(
            [out_reset.index.name or "Date", "sector"]
        )["return_1d"].transform(lambda x: x.abs().rank(pct=True))
    else:
        out_reset["return_sector_rank"] = np.nan

    # Restore index
    if "Date" in out_reset.columns:
        out_reset = out_reset.set_index("Date")

    return out_reset


def compute_cross_sectional_single(stock_df: pd.DataFrame,
                                     index_data: dict[str, pd.DataFrame],
                                     beta_window: int = 60) -> pd.DataFrame:
    """
    Compute cross-sectional features for a single stock.

    Parameters
    ----------
    stock_df : pd.DataFrame
        Single stock's processed data
    index_data : dict
        Mapping of index ticker -> DataFrame
    beta_window : int
        Rolling OLS window for beta calculation

    Returns
    -------
    pd.DataFrame
        stock_df with added cross-sectional features
    """
    out = stock_df.copy()

    # Use Nifty 50 as primary benchmark
    nifty_df = index_data.get("^NSEI")
    if nifty_df is None:
        # Try Bank Nifty as fallback for banking stocks
        sector = out["sector"].iloc[0] if "sector" in out.columns else ""
        if sector == "Banking":
            nifty_df = index_data.get("^NSEBANK")

    if nifty_df is not None:
        out = compute_beta_adjusted_residual(out, nifty_df, window=beta_window)
    else:
        out["beta"] = np.nan
        out["alpha"] = np.nan
        out["beta_adjusted_residual"] = np.nan
        out["abs_residual"] = np.nan

    return out
