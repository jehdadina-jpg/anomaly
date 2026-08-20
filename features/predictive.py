"""
Feature engineering for price prediction.

Operates on the tidy panel from `data.build_panel` and returns the same panel
with predictive features appended. Two families:

* Per-stock time-series features — momentum, reversal, volatility, trend,
  volume, and NSE delivery/microstructure features.
* Cross-sectional features — where a stock sits *relative to the universe*
  on a given day. These carry most of the signal: relative rank is far more
  predictable than absolute direction.

Every feature is computed strictly from information available at or before
time t. Forward-looking columns live in `training.targets`, never here.

Implementation note: all rolling/grouped operations preserve the panel's
index. Building a `pd.Series` from a raw numpy array inside a grouped apply
silently reindexes and yields all-NaN columns — the bug that killed the ADX
family in `prediction_features.py`.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

EPS = 1e-12


# ---------------------------------------------------------------------------
# Per-stock helpers (operate on a single ticker's frame, date-ascending)
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / (loss + EPS)
    return 100 - (100 / (1 + rs))


def _adx(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.DataFrame:
    """ADX / DI. All intermediates stay pandas so the index stays aligned."""
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr + EPS)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr + EPS)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return pd.DataFrame({
        "di_plus_14": plus_di,
        "di_minus_14": minus_di,
        "adx_14": adx,
        "atr_14": atr,
    }, index=close.index)


def _per_stock_features(g: pd.DataFrame) -> pd.DataFrame:
    """Compute time-series features for one ticker."""
    g = g.sort_values("Date").copy()
    close, high, low, open_ = g["Close"], g["High"], g["Low"], g["Open"]
    volume = g["Volume"]

    # --- Returns over multiple lookbacks ---
    for n in (1, 2, 3, 5, 10, 20, 60):
        g[f"ret_{n}d"] = close.pct_change(n)

    log_ret = np.log(close / close.shift(1))
    g["log_ret_1d"] = log_ret

    # --- Momentum ---
    # 12-1 momentum: the classic cross-sectional factor. Skipping the most
    # recent month avoids contaminating momentum with short-term reversal.
    g["momentum_12_1"] = close.shift(21) / (close.shift(252) + EPS) - 1
    g["momentum_60d"] = close.pct_change(60)
    g["momentum_120d"] = close.pct_change(120)

    # --- Short-horizon reversal (opposite sign to momentum, both matter) ---
    g["reversal_1d"] = -g["ret_1d"]
    g["reversal_5d"] = -g["ret_5d"]

    # --- Volatility ---
    for n in (5, 20, 60):
        g[f"vol_{n}d"] = log_ret.rolling(n).std()
    g["vol_ratio_5_20"] = g["vol_5d"] / (g["vol_20d"] + EPS)
    g["vol_ratio_20_60"] = g["vol_20d"] / (g["vol_60d"] + EPS)
    g["vol_of_vol"] = g["vol_20d"].rolling(60).std()

    # Parkinson volatility uses the high-low range: less noisy than close-close.
    hl = np.log((high + EPS) / (low + EPS))
    g["parkinson_vol_20"] = np.sqrt((hl ** 2).rolling(20).mean() / (4 * np.log(2)))

    # --- Trend / moving averages ---
    for n in (5, 20, 50, 200):
        sma = close.rolling(n).mean()
        g[f"px_to_sma_{n}"] = close / (sma + EPS) - 1
    g["sma_5_20_cross"] = (close.rolling(5).mean()
                           / (close.rolling(20).mean() + EPS) - 1)
    g["sma_20_50_cross"] = (close.rolling(20).mean()
                            / (close.rolling(50).mean() + EPS) - 1)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    g["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / (close + EPS)

    trend = _adx(high, low, close, 14)
    for c in trend.columns:
        g[c] = trend[c]
    g["atr_pct"] = g["atr_14"] / (close + EPS)

    # --- Oscillators ---
    g["rsi_14"] = _rsi(close, 14)
    g["rsi_2"] = _rsi(close, 2)          # very short-term mean reversion
    sma20, std20 = close.rolling(20).mean(), close.rolling(20).std()
    g["bollinger_z"] = (close - sma20) / (std20 + EPS)

    # --- 52-week range position (documented anomaly) ---
    hi_52 = close.rolling(252, min_periods=60).max()
    lo_52 = close.rolling(252, min_periods=60).min()
    g["pct_from_52w_high"] = close / (hi_52 + EPS) - 1
    g["pct_from_52w_low"] = close / (lo_52 + EPS) - 1
    g["range_position_52w"] = (close - lo_52) / (hi_52 - lo_52 + EPS)

    # --- Intraday structure ---
    g["overnight_gap"] = open_ / (close.shift(1) + EPS) - 1
    g["intraday_ret"] = close / (open_ + EPS) - 1
    g["close_position_in_range"] = (close - low) / (high - low + EPS)
    g["daily_range_pct"] = (high - low) / (close + EPS)
    g["gap_vs_range"] = g["overnight_gap"] / (g["daily_range_pct"] + EPS)

    # --- Volume ---
    vol_ma20 = volume.rolling(20).mean()
    g["volume_ratio_20"] = volume / (vol_ma20 + EPS)
    g["volume_zscore_20"] = ((volume - vol_ma20)
                             / (volume.rolling(20).std() + EPS))
    g["volume_trend_5_20"] = volume.rolling(5).mean() / (vol_ma20 + EPS)
    g["dollar_volume"] = np.log1p(close * volume)

    # Amihud illiquidity: price impact per unit traded. Rises under stress.
    g["amihud_illiq"] = (g["ret_1d"].abs()
                         / (close * volume + EPS)).rolling(20).mean() * 1e9

    # Signed volume: volume carries direction only when paired with return sign.
    g["signed_volume_20"] = (np.sign(g["ret_1d"]) * g["volume_ratio_20"]).rolling(20).mean()

    # --- NSE delivery / microstructure (the India-specific edge) ---
    if "delivery_pct" in g.columns:
        dp = g["delivery_pct"]
        g["delivery_pct_ma20"] = dp.rolling(20).mean()
        g["delivery_pct_z20"] = (dp - g["delivery_pct_ma20"]) / (dp.rolling(20).std() + EPS)
        g["delivery_pct_chg"] = dp - dp.shift(1)
        g["delivery_pct_trend"] = dp.rolling(5).mean() / (g["delivery_pct_ma20"] + EPS)
        # High delivery on rising volume = conviction buying, not day trading.
        g["delivery_volume_conviction"] = dp * g["volume_ratio_20"]

    if "n_trades" in g.columns and "nse_traded_qty" in g.columns:
        # Average trade size proxies institutional vs retail participation.
        avg_trade = g["nse_traded_qty"] / (g["n_trades"] + EPS)
        g["avg_trade_size"] = np.log1p(avg_trade)
        g["avg_trade_size_z20"] = ((avg_trade - avg_trade.rolling(20).mean())
                                   / (avg_trade.rolling(20).std() + EPS))
        g["n_trades_z20"] = ((g["n_trades"] - g["n_trades"].rolling(20).mean())
                             / (g["n_trades"].rolling(20).std() + EPS))

    if "turnover_lacs" in g.columns:
        to = g["turnover_lacs"]
        g["turnover_z20"] = (to - to.rolling(20).mean()) / (to.rolling(20).std() + EPS)

    return g


# ---------------------------------------------------------------------------
# Cross-sectional features (operate across all tickers on a given date)
# ---------------------------------------------------------------------------

CROSS_SECTIONAL_BASE = [
    "ret_1d", "ret_5d", "ret_20d", "momentum_12_1", "momentum_60d",
    "vol_20d", "rsi_14", "volume_ratio_20", "delivery_pct",
    "delivery_pct_z20", "range_position_52w", "amihud_illiq",
]


def _add_cross_sectional(panel: pd.DataFrame) -> pd.DataFrame:
    """Rank and demean features within each date across the universe."""
    out = panel.copy()
    by_date = out.groupby("Date")

    for col in CROSS_SECTIONAL_BASE:
        if col not in out.columns:
            continue
        grp = by_date[col]
        # Percentile rank in [0, 1]: scale-free and robust to outliers.
        out[f"xs_rank_{col}"] = grp.rank(pct=True)
        # Distance from the day's cross-section, in cross-sectional sigmas.
        out[f"xs_z_{col}"] = (out[col] - grp.transform("mean")) / (grp.transform("std") + EPS)

    # Market (equal-weight universe) return and each stock's excess over it.
    out["market_ret_1d"] = by_date["ret_1d"].transform("mean")
    out["market_ret_5d"] = by_date["ret_5d"].transform("mean")
    out["excess_ret_1d"] = out["ret_1d"] - out["market_ret_1d"]
    out["excess_ret_5d"] = out["ret_5d"] - out["market_ret_5d"]

    # Sector-relative: strips out sector rotation to isolate stock selection.
    by_ds = out.groupby(["Date", "sector"])
    out["sector_ret_1d"] = by_ds["ret_1d"].transform("mean")
    out["sector_ret_5d"] = by_ds["ret_5d"].transform("mean")
    out["vs_sector_ret_1d"] = out["ret_1d"] - out["sector_ret_1d"]
    out["vs_sector_ret_5d"] = out["ret_5d"] - out["sector_ret_5d"]
    out["sector_rank_ret_5d"] = by_ds["ret_5d"].rank(pct=True)

    # Market breadth: fraction of the universe up today.
    out["market_breadth"] = by_date["ret_1d"].transform(lambda s: (s > 0).mean())

    return out


def _add_beta(panel: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Rolling beta and idiosyncratic volatility vs the equal-weight universe."""
    out = panel.sort_values(["ticker", "Date"]).copy()

    def _beta(g):
        g = g.sort_values("Date")
        cov = g["ret_1d"].rolling(window).cov(g["market_ret_1d"])
        var = g["market_ret_1d"].rolling(window).var()
        beta = cov / (var + EPS)
        resid = g["ret_1d"] - beta * g["market_ret_1d"]
        return pd.DataFrame({
            "beta_60d": beta,
            "idio_vol_60d": resid.rolling(window).std(),
        }, index=g.index)

    parts = [_beta(g) for _, g in out.groupby("ticker", sort=False)]
    betas = pd.concat(parts).reindex(out.index)
    out["beta_60d"] = betas["beta_60d"]
    out["idio_vol_60d"] = betas["idio_vol_60d"]
    return out


def _add_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    """Calendar effects, including F&O expiry proximity."""
    out = panel.copy()
    dates = pd.to_datetime(out["Date"])
    out["day_of_week"] = dates.dt.dayofweek
    out["day_of_month"] = dates.dt.day
    out["month"] = dates.dt.month

    expiry = pd.to_datetime(sorted(config.FNO_EXPIRY_DATES))
    # Trading days until the next monthly expiry; expiry weeks behave oddly
    # in Indian markets thanks to derivative roll pressure.
    idx = np.searchsorted(expiry.values, dates.values, side="left")
    idx = np.clip(idx, 0, len(expiry) - 1)
    days_to_expiry = (expiry.values[idx] - dates.values) / np.timedelta64(1, "D")
    out["days_to_expiry"] = days_to_expiry
    out["is_expiry_week"] = (days_to_expiry <= 5).astype(int)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Columns that are identifiers/raw inputs rather than model features.
NON_FEATURE_COLS = {
    "Date", "Close", "High", "Low", "Open", "Volume", "ticker", "symbol",
    "sector", "delivery_qty", "nse_traded_qty", "turnover_lacs", "n_trades",
}


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add all predictive features to the panel."""
    logger.info("Computing per-stock features for %d tickers...",
                panel["ticker"].nunique())
    parts = [_per_stock_features(g) for _, g in panel.groupby("ticker", sort=False)]
    out = pd.concat(parts, ignore_index=True)

    logger.info("Computing cross-sectional features...")
    out = _add_cross_sectional(out)
    out = _add_beta(out)
    out = _add_calendar(out)

    out = out.sort_values(["Date", "ticker"]).reset_index(drop=True)

    # Guard against the all-NaN class of bug that motivated this rewrite.
    feats = feature_columns(out)
    dead = [c for c in feats if out[c].notna().sum() == 0]
    if dead:
        raise ValueError(f"Features computed as entirely NaN: {dead}")

    logger.info("Built %d features over %d rows.", len(feats), len(out))
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model feature columns: numeric, not identifiers, not targets."""
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and not c.startswith(("target_", "fwd_"))
        and pd.api.types.is_numeric_dtype(df[c])
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    from data.build_panel import load_panel

    p = load_panel()
    f = build_features(p)
    cols = feature_columns(f)
    print(f"\n{len(cols)} features over {len(f)} rows")
    print("\nCoverage (non-null %) of the 15 sparsest features:")
    cov = f[cols].notna().mean().sort_values()
    print((cov.head(15) * 100).round(1).to_string())
