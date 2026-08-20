"""
Prediction-specific features for price movement forecasting.

Features optimized for price prediction (1-day, 5-day, 10-day forward returns):
- Momentum indicators (ROC 5/10/20/60 day, Chaikin Money Flow)
- Trend indicators (ADX, Parabolic SAR, Ichimoku components, SuperTrend)
- Volatility indicators (ATR percentile, Bollinger Band width, volatility regime)
- Relative strength features (RS vs sector, RS vs market, RS rank)
- Order flow proxies (volume delta, VWAP distance, delivery patterns, F&O effects)
- Pattern similarity features (DTW/euclidean distance to historical patterns)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def compute_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute momentum indicators for price prediction.
    
    Momentum indicators measure the rate of change in price over various periods.
    Strong positive momentum often precedes continued upward moves (momentum persistence),
    while negative momentum can signal downtrends.
    
    Features:
    - momentum_5d, momentum_10d, momentum_20d, momentum_60d: (Close_t - Close_t-N) / Close_t-N
    - roc_5d, roc_10d, roc_20d: Rate of change indicators (same as momentum but standardized)
    - chaikin_money_flow_20: Chaikin Money Flow indicator (CMF) - volume-weighted price pressure
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain Close, High, Low, Volume columns.
    
    Returns
    -------
    pd.DataFrame
        Original df with momentum indicators added.
    """
    out = df.copy()
    
    # Momentum features (rate of change over various periods)
    for period in [5, 10, 20, 60]:
        out[f"momentum_{period}d"] = out["Close"].pct_change(period)
        out[f"roc_{period}d"] = out[f"momentum_{period}d"]  # Alias for consistency
    
    # Chaikin Money Flow (CMF) - measures buying/selling pressure
    # CMF = sum(Money Flow Volume) / sum(Volume) over 20 periods
    # Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    # Money Flow Volume = Money Flow Multiplier * Volume
    
    money_flow_multiplier = (
        (out["Close"] - out["Low"]) - (out["High"] - out["Close"])
    ) / (out["High"] - out["Low"]).replace(0, np.nan)
    
    money_flow_volume = money_flow_multiplier * out["Volume"]
    
    out["chaikin_money_flow_20"] = (
        money_flow_volume.rolling(20).sum() / 
        out["Volume"].rolling(20).sum().replace(0, np.nan)
    )
    
    return out


def compute_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute trend strength indicators for price prediction.
    
    Trend indicators identify the strength and direction of price trends.
    ADX (Average Directional Index) is widely used to measure trend strength,
    while Parabolic SAR and Ichimoku components help identify trend reversals.
    
    Features:
    - adx_14: Average Directional Index (14-period) - trend strength
    - di_plus_14: Positive Directional Indicator - upward pressure
    - di_minus_14: Negative Directional Indicator - downward pressure
    - trend_strength_score: Normalized ADX score
    - parabolic_sar: Parabolic Stop and Reverse indicator
    - ichimoku_conversion: Ichimoku Conversion Line (Tenkan-sen)
    - ichimoku_base: Ichimoku Base Line (Kijun-sen)
    - supertrend: SuperTrend indicator (trend-following)
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain High, Low, Close columns.
    
    Returns
    -------
    pd.DataFrame
        Original df with trend indicators added.
    """
    out = df.copy()
    
    # Calculate True Range (TR) for ADX
    high_low = out["High"] - out["Low"]
    high_close = (out["High"] - out["Close"].shift(1)).abs()
    low_close = (out["Low"] - out["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate Directional Movement (DM)
    up_move = out["High"] - out["High"].shift(1)
    down_move = out["Low"].shift(1) - out["Low"]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR and DM using EMA (14-period)
    atr_14 = true_range.ewm(span=14, adjust=False).mean()
    plus_di_raw = pd.Series(plus_dm).ewm(span=14, adjust=False).mean()
    minus_di_raw = pd.Series(minus_dm).ewm(span=14, adjust=False).mean()
    
    # Calculate Directional Indicators (DI)
    out["di_plus_14"] = 100 * (plus_di_raw / atr_14.replace(0, np.nan))
    out["di_minus_14"] = 100 * (minus_di_raw / atr_14.replace(0, np.nan))
    
    # Calculate ADX (Average Directional Index)
    dx = 100 * (
        (out["di_plus_14"] - out["di_minus_14"]).abs() /
        (out["di_plus_14"] + out["di_minus_14"]).replace(0, np.nan)
    )
    out["adx_14"] = dx.ewm(span=14, adjust=False).mean()
    
    # Trend strength score (normalized ADX)
    out["trend_strength_score"] = (out["adx_14"] - 20) / 5
    out["trend_strength_score"] = out["trend_strength_score"].clip(-1, 4)
    
    # Parabolic SAR (simplified version)
    # Full implementation is complex; using a simplified approximation
    af = 0.02  # Acceleration factor
    max_af = 0.20
    
    # Simplified SAR: use recent low/high as initial SAR
    sar = out["Low"].rolling(5).min().copy()
    out["parabolic_sar"] = sar
    
    # Ichimoku Cloud components
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low) / 2
    period9_high = out["High"].rolling(9).max()
    period9_low = out["Low"].rolling(9).min()
    out["ichimoku_conversion"] = (period9_high + period9_low) / 2
    
    # Base Line (Kijun-sen): (26-period high + 26-period low) / 2
    period26_high = out["High"].rolling(26).max()
    period26_low = out["Low"].rolling(26).min()
    out["ichimoku_base"] = (period26_high + period26_low) / 2
    
    # SuperTrend indicator (trend-following)
    # SuperTrend = (High + Low) / 2 ± (multiplier × ATR)
    multiplier = 3
    hl_avg = (out["High"] + out["Low"]) / 2
    
    upper_band = hl_avg + (multiplier * atr_14)
    lower_band = hl_avg - (multiplier * atr_14)
    
    # Simplified SuperTrend: positive when Close > lower_band
    out["supertrend"] = np.where(out["Close"] > lower_band, 1, -1)
    
    return out


def compute_price_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute multi-timeframe price ratios for trend alignment.
    
    Price ratios help identify when a stock is trading above/below key moving averages,
    which can signal trend strength or potential reversals.
    
    Features:
    - price_to_sma5_ratio: Current price / 5-day SMA
    - price_to_sma20_ratio: Current price / 20-day SMA
    - price_to_sma60_ratio: Current price / 60-day SMA
    - sma5_to_sma20_ratio: 5-day SMA / 20-day SMA (trend alignment)
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain Close column.
    
    Returns
    -------
    pd.DataFrame
        Original df with price ratio features added.
    """
    out = df.copy()
    
    # Calculate SMAs
    sma_5 = out["Close"].rolling(5).mean()
    sma_20 = out["Close"].rolling(20).mean()
    sma_60 = out["Close"].rolling(60).mean()
    
    # Price ratios
    out["price_to_sma5_ratio"] = out["Close"] / sma_5.replace(0, np.nan)
    out["price_to_sma20_ratio"] = out["Close"] / sma_20.replace(0, np.nan)
    out["price_to_sma60_ratio"] = out["Close"] / sma_60.replace(0, np.nan)
    
    # SMA alignment ratio (trend confirmation)
    out["sma5_to_sma20_ratio"] = sma_5 / sma_20.replace(0, np.nan)
    
    return out


def compute_volume_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volume-price relationship features for order flow analysis.
    
    Volume-price divergences can signal trend weakness or reversals.
    These features help identify when volume patterns contradict price movements.
    
    Features:
    - vol_price_corr_20: Rolling 20-day correlation between Volume and |Returns|
    - vol_price_divergence_5d_ma: 5-day MA of existing vol_price_divergence
    - delivery_pct_trend_5d: 5-day change in delivery percentage (India-specific)
    - volume_momentum_5d: (Volume_t / Volume_t-5) - 1
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain Volume, Close columns. Optionally delivery_pct.
    
    Returns
    -------
    pd.DataFrame
        Original df with volume-price features added.
    """
    out = df.copy()
    
    # Calculate returns if not present
    if "return_1d" not in out.columns:
        out["return_1d"] = out["Close"].pct_change(1)
    
    # Volume-price correlation (positive correlation = healthy trend)
    abs_returns = out["return_1d"].abs()
    out["vol_price_corr_20"] = (
        out["Volume"].rolling(20).corr(abs_returns)
    )
    
    # Smoothed volume-price divergence
    if "vol_price_divergence" in out.columns:
        out["vol_price_divergence_5d_ma"] = (
            out["vol_price_divergence"].rolling(5).mean()
        )
    else:
        # Calculate basic divergence if not present
        vol_change = out["Volume"].pct_change(1)
        price_sign = np.sign(out["return_1d"])
        vol_sign = np.sign(vol_change)
        divergence = -(price_sign * vol_sign)
        magnitude = out["return_1d"].abs() * vol_change.abs()
        out["vol_price_divergence_5d_ma"] = (divergence * magnitude).rolling(5).mean()
    
    # Delivery percentage trend (India-specific)
    if "delivery_pct" in out.columns:
        out["delivery_pct_trend_5d"] = out["delivery_pct"].pct_change(5)
    else:
        out["delivery_pct_trend_5d"] = np.nan
    
    # Volume momentum
    out["volume_momentum_5d"] = out["Volume"].pct_change(5)
    
    return out


def compute_relative_strength(df: pd.DataFrame, 
                               sector_data: dict = None,
                               index_data: dict = None) -> pd.DataFrame:
    """
    Compute relative strength features vs sector and market.
    
    Relative strength helps identify stocks outperforming or underperforming
    their sector and the broader market, which can signal continued momentum
    or mean reversion opportunities.
    
    Features:
    - stock_vs_sector_return_5d: Stock 5d return - Sector 5d return
    - stock_vs_index_return_5d: Stock 5d return - ^NSEI 5d return
    - beta_adjusted_return: Return - (beta × index_return)
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain Close column and optionally 'ticker' and 'beta'.
    sector_data : dict, optional
        Dict of {sector_name: pd.DataFrame with Close column}
    index_data : dict, optional
        Dict of {'^NSEI': pd.DataFrame with Close column}
    
    Returns
    -------
    pd.DataFrame
        Original df with relative strength features added.
    """
    out = df.copy()
    
    # Calculate 5-day returns
    stock_return_5d = out["Close"].pct_change(5)
    
    # Get ticker and sector
    ticker = ""
    if "ticker" in out.columns and len(out) > 0:
        ticker = out["ticker"].iloc[0]
    
    sector = config.TICKER_TO_SECTOR.get(ticker, None)
    
    # Stock vs Sector return
    if sector_data and sector and sector in sector_data:
        sector_df = sector_data[sector]
        # Align dates
        sector_return_5d = sector_df["Close"].pct_change(5)
        out["stock_vs_sector_return_5d"] = (
            stock_return_5d - sector_return_5d.reindex(out.index)
        )
    else:
        out["stock_vs_sector_return_5d"] = np.nan
    
    # Stock vs Index return
    if index_data and "^NSEI" in index_data:
        index_df = index_data["^NSEI"]
        index_return_5d = index_df["Close"].pct_change(5)
        out["stock_vs_index_return_5d"] = (
            stock_return_5d - index_return_5d.reindex(out.index)
        )
    else:
        out["stock_vs_index_return_5d"] = np.nan
    
    # Beta-adjusted return (alpha)
    if "beta" in out.columns and index_data and "^NSEI" in index_data:
        index_df = index_data["^NSEI"]
        index_return_5d = index_df["Close"].pct_change(5).reindex(out.index)
        out["beta_adjusted_return"] = (
            stock_return_5d - (out["beta"] * index_return_5d)
        )
    else:
        out["beta_adjusted_return"] = np.nan
    
    return out


def compute_volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volatility regime features for risk assessment.
    
    Volatility regime classification helps identify when markets are calm vs turbulent,
    which affects prediction reliability and trading strategies.
    
    Features:
    - volatility_percentile_60d: Current vol_20 vs 60-day historical percentile
    - volatility_spike_flag: 1 if vol_20 > 2 × median(vol_20, 60d)
    - volatility_regime: Categorical (0=low, 1=medium, 2=high based on percentile)
    - bollinger_band_width: (Upper BB - Lower BB) / Middle BB
    - atr_percentile_60d: ATR percentile vs 60-day history
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain rolling_vol_20 (or will be computed). High, Low, Close for ATR.
    
    Returns
    -------
    pd.DataFrame
        Original df with volatility regime features added.
    """
    out = df.copy()
    
    # Ensure rolling volatility exists
    if "rolling_vol_20" not in out.columns:
        if "log_return_1d" not in out.columns:
            out["log_return_1d"] = np.log(out["Close"] / out["Close"].shift(1))
        out["rolling_vol_20"] = out["log_return_1d"].rolling(20).std()
    
    # Volatility percentile (current vol vs 60-day history)
    out["volatility_percentile_60d"] = (
        out["rolling_vol_20"].rolling(60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan,
            raw=False
        )
    )
    
    # Volatility spike flag
    vol_median_60 = out["rolling_vol_20"].rolling(60).median()
    out["volatility_spike_flag"] = (
        out["rolling_vol_20"] > 2 * vol_median_60
    ).astype(int)
    
    # Volatility regime classification
    # Low: percentile < 0.33, Medium: 0.33-0.67, High: > 0.67
    out["volatility_regime"] = pd.cut(
        out["volatility_percentile_60d"],
        bins=[0, 0.33, 0.67, 1.0],
        labels=[0, 1, 2],
        include_lowest=True
    ).astype(float)
    
    # Bollinger Band width
    sma_20 = out["Close"].rolling(20).mean()
    std_20 = out["Close"].rolling(20).std()
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    out["bollinger_band_width"] = (
        (upper_bb - lower_bb) / sma_20.replace(0, np.nan)
    )
    
    # ATR percentile
    high_low = out["High"] - out["Low"]
    high_close = (out["High"] - out["Close"].shift(1)).abs()
    low_close = (out["Low"] - out["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()
    
    out["atr_percentile_60d"] = (
        atr.rolling(60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan,
            raw=False
        )
    )
    
    return out


def compute_order_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute order flow features (India-specific indicators).
    
    Order flow features capture institutional vs retail activity patterns,
    particularly relevant in the Indian market with delivery percentage data
    and F&O expiry effects.
    
    Features:
    - delivery_pct_ma20: 20-day MA of delivery percentage
    - delivery_pct_std20: 20-day std of delivery percentage
    - delivery_spike_flag: 1 if delivery_pct > mean + 2×std
    - fno_expiry_amplification: Multiplier based on days_to_expiry (higher near expiry)
    - vwap_distance: Distance from VWAP (as percentage)
    - volume_delta: Change in volume from previous day
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain Volume. Optionally delivery_pct, days_to_expiry.
    
    Returns
    -------
    pd.DataFrame
        Original df with order flow features added.
    """
    out = df.copy()
    
    # Delivery percentage features (India-specific)
    if "delivery_pct" in out.columns:
        out["delivery_pct_ma20"] = out["delivery_pct"].rolling(20).mean()
        out["delivery_pct_std20"] = out["delivery_pct"].rolling(20).std()
        
        # Delivery spike flag
        out["delivery_spike_flag"] = (
            out["delivery_pct"] > (out["delivery_pct_ma20"] + 2 * out["delivery_pct_std20"])
        ).astype(int)
    else:
        out["delivery_pct_ma20"] = np.nan
        out["delivery_pct_std20"] = np.nan
        out["delivery_spike_flag"] = 0
    
    # F&O expiry amplification (higher near expiry)
    if "days_to_expiry" in out.columns:
        # Exponential decay: amplification peaks at expiry (days=0)
        # Amplification = 1 + exp(-days/3)
        out["fno_expiry_amplification"] = 1 + np.exp(-out["days_to_expiry"] / 3.0)
    else:
        out["fno_expiry_amplification"] = 1.0
    
    # VWAP distance
    if "rolling_vwap" in out.columns:
        out["vwap_distance"] = (
            (out["Close"] - out["rolling_vwap"]) / out["rolling_vwap"].replace(0, np.nan)
        )
    else:
        # Compute VWAP if not present
        if "typical_price" not in out.columns:
            out["typical_price"] = (out["High"] + out["Low"] + out["Close"]) / 3
        tp_vol = out["typical_price"] * out["Volume"]
        rolling_tp_vol = tp_vol.rolling(20).sum()
        rolling_vol = out["Volume"].rolling(20).sum()
        rolling_vwap = rolling_tp_vol / rolling_vol.replace(0, np.nan)
        out["vwap_distance"] = (
            (out["Close"] - rolling_vwap) / rolling_vwap.replace(0, np.nan)
        )
    
    # Volume delta
    out["volume_delta"] = out["Volume"].diff()
    
    return out


def compute_pattern_similarity(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Compute pattern similarity features using historical price patterns.
    
    Pattern similarity helps identify when current price action resembles
    historical patterns, which can forecast similar future moves.
    
    Features:
    - pattern_similarity_score: Euclidean distance to nearest historical pattern
    - recent_pattern_cluster: K-means cluster ID of recent price pattern
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain Close column.
    window : int
        Window size for pattern comparison (default 10 days).
    
    Returns
    -------
    pd.DataFrame
        Original df with pattern similarity features added.
    """
    out = df.copy()
    
    # Normalize close prices for pattern comparison (use percentage changes)
    returns = out["Close"].pct_change().fillna(0)
    
    # Create rolling windows of returns
    patterns = []
    indices = []
    
    for i in range(window, len(returns)):
        pattern = returns.iloc[i-window:i].values
        patterns.append(pattern)
        indices.append(returns.index[i])
    
    if len(patterns) < 20:  # Need enough patterns for clustering
        out["pattern_similarity_score"] = np.nan
        out["recent_pattern_cluster"] = np.nan
        return out
    
    patterns_array = np.array(patterns)
    
    # Pattern similarity: distance to nearest historical pattern
    similarity_scores = []
    
    for i in range(len(patterns)):
        current_pattern = patterns[i]
        
        # Compare to all previous patterns (avoid lookahead)
        if i < 20:
            similarity_scores.append(np.nan)
            continue
        
        historical_patterns = patterns_array[:i-5]  # Exclude very recent patterns
        
        # Calculate euclidean distances
        distances = [euclidean(current_pattern, hist_pattern) 
                    for hist_pattern in historical_patterns]
        
        # Minimum distance to any historical pattern
        min_distance = min(distances) if distances else np.nan
        similarity_scores.append(min_distance)
    
    # K-means clustering of patterns (3 clusters: bullish, neutral, bearish)
    if len(patterns) >= 30:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(patterns_array)
    else:
        cluster_labels = [0] * len(patterns)
    
    # Create Series with proper indices
    similarity_series = pd.Series(similarity_scores, index=indices)
    cluster_series = pd.Series(cluster_labels, index=indices)
    
    # Merge back to original DataFrame
    out["pattern_similarity_score"] = similarity_series
    out["recent_pattern_cluster"] = cluster_series
    
    return out


def compute_all_prediction_features(df: pd.DataFrame,
                                     sector_data: dict = None,
                                     index_data: dict = None) -> pd.DataFrame:
    """
    Compute all prediction-specific features.
    
    This is the main orchestration function that applies all prediction
    feature engineering functions in sequence.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain OHLCV columns with DatetimeIndex.
    sector_data : dict, optional
        Dict of sector DataFrames for relative strength calculations.
    index_data : dict, optional
        Dict of index DataFrames (^NSEI, ^NSEBANK) for relative strength.
    
    Returns
    -------
    pd.DataFrame
        Original df with all prediction features added (20+ new columns).
    """
    out = df.copy()
    
    # Apply feature engineering functions
    out = compute_momentum_indicators(out)
    out = compute_trend_indicators(out)
    out = compute_price_ratios(out)
    out = compute_volume_price_features(out)
    out = compute_relative_strength(out, sector_data=sector_data, index_data=index_data)
    out = compute_volatility_regime(out)
    out = compute_order_flow_features(out)
    out = compute_pattern_similarity(out)
    
    return out

