"""
Feature engineering orchestrator.

Calls all feature modules, concatenates into a single feature matrix
per stock-day, handles NaN warmup, standardizes features, and saves
the final feature matrices.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from features.price_features import compute_all_price_features
from features.volume_features import compute_all_volume_features
from features.pattern_features import compute_all_pattern_features
from features.cross_sectional import compute_cross_sectional_single, compute_sector_ranks

logger = logging.getLogger(__name__)

# List of feature columns used as model input (excludes metadata)
FEATURE_COLUMNS = [
    # Price features
    "return_1d", "return_5d", "log_return_1d",
    "rolling_vol_5", "rolling_vol_20", "vol_ratio_5_20",
    "price_acceleration", "abs_price_acceleration",
    "vwap_deviation",
    "gap_magnitude", "abs_gap_magnitude",
    # Volume features
    "volume_zscore_20", "volume_ratio_20",
    "vol_price_divergence", "vol_price_div_5d",
    "delivery_pct_zscore", "low_delivery_pump_flag",
    "volume_spike_flag", "volume_spikes_5d",
    # Pattern features
    "intraday_reversal", "reversal_5d_avg",
    "upper_wick_ratio", "lower_wick_ratio",
    "days_to_expiry", "is_expiry_week", "expiry_proximity_score",
    "circuit_proximity", "upper_circuit_proximity", "lower_circuit_proximity",
    "near_circuit_flag", "consecutive_circuit_days",
    "return_autocorr_5", "return_autocorr_20",
    # Cross-sectional features
    "beta", "beta_adjusted_residual", "abs_residual",
]


def build_features_single(stock_df: pd.DataFrame,
                            index_data: dict[str, pd.DataFrame] = None,
                            params: dict = None) -> pd.DataFrame:
    """
    Build all features for a single stock.

    Parameters
    ----------
    stock_df : pd.DataFrame
        Preprocessed OHLCV data for one stock
    index_data : dict
        Index DataFrames for beta calculation
    params : dict
        Feature engineering parameters (from config)

    Returns
    -------
    pd.DataFrame
        Stock data with all features added
    """
    params = params or config.FEATURE_PARAMS

    logger.debug(f"Building features for {stock_df.get('ticker', ['?'])[0] if 'ticker' in stock_df.columns else '?'}")

    # 1. Price features
    df = compute_all_price_features(
        stock_df,
        vol_windows=params["rolling_vol_windows"],
        vwap_window=params["volume_zscore_window"],
    )

    # 2. Volume features
    df = compute_all_volume_features(
        df,
        zscore_window=params["volume_zscore_window"],
        spike_threshold=params["volume_spike_threshold"],
        delivery_threshold=params["delivery_pct_low_threshold"],
    )

    # 3. Pattern features
    df = compute_all_pattern_features(
        df,
        autocorr_windows=params["autocorrelation_windows"],
    )

    # 4. Cross-sectional features
    if index_data:
        df = compute_cross_sectional_single(
            df, index_data,
            beta_window=params["beta_window"],
        )

    return df


def build_all(processed_data: dict[str, pd.DataFrame] = None,
              test_mode: bool = False,
              save: bool = True) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Build features for all stocks, standardize, and save.

    Parameters
    ----------
    processed_data : dict
        Ticker -> processed DataFrame mapping. If None, loads from disk.
    test_mode : bool
        If True, process only test tickers with limited date range.
    save : bool
        Whether to save feature matrices and scaler to disk.

    Returns
    -------
    tuple[pd.DataFrame, StandardScaler]
        (feature_matrix, fitted_scaler)
    """
    if processed_data is None:
        from data.preprocess import load_processed
        processed_data = load_processed()

    if test_mode:
        processed_data = {
            k: v for k, v in processed_data.items()
            if k in config.TEST_MODE_TICKERS
        }

    # Separate index data from stock data
    index_data = {k: v for k, v in processed_data.items() if k.startswith("^")}
    stock_data = {k: v for k, v in processed_data.items() if not k.startswith("^")}

    logger.info(f"Building features for {len(stock_data)} stocks")
    logger.info(f"Index data available: {list(index_data.keys())}")

    # Build features per stock
    all_features = []
    for ticker, df in stock_data.items():
        logger.info(f"  Processing {ticker} ({len(df)} rows)")

        try:
            featured_df = build_features_single(df, index_data)

            # Drop warmup rows (NaN from rolling windows)
            warmup = config.FEATURE_PARAMS["warmup_period"]
            featured_df = featured_df.iloc[warmup:]

            all_features.append(featured_df)
        except Exception as e:
            logger.error(f"  Failed to build features for {ticker}: {e}")

    if not all_features:
        logger.error("No features built")
        return pd.DataFrame(), None

    # Combine into panel
    panel = pd.concat(all_features)
    logger.info(f"Combined feature panel: {panel.shape}")

    # ---- Standardize features ----
    # Get the feature columns that actually exist in the panel
    available_features = [c for c in FEATURE_COLUMNS if c in panel.columns]
    missing_features = [c for c in FEATURE_COLUMNS if c not in panel.columns]

    if missing_features:
        logger.warning(f"Missing feature columns: {missing_features}")

    logger.info(f"Standardizing {len(available_features)} features")

    # Handle infinities before scaling
    feature_matrix = panel[available_features].copy()
    feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with column median (robust to outliers)
    for col in available_features:
        median_val = feature_matrix[col].median()
        feature_matrix[col] = feature_matrix[col].fillna(median_val)

    # Fit scaler
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(feature_matrix)
    scaled_df = pd.DataFrame(
        scaled_values,
        index=feature_matrix.index,
        columns=[f"{c}_scaled" for c in available_features],
    )

    # Add scaled features back to panel
    panel = panel.join(scaled_df)

    # ---- Save ----
    if save:
        # Save full panel (unscaled + scaled)
        panel_path = config.FEATURES_DIR / "feature_panel.parquet"
        try:
            panel.to_parquet(panel_path)
        except Exception:
            # Fallback to CSV if parquet fails
            panel_path = config.FEATURES_DIR / "feature_panel.csv"
            panel.to_csv(panel_path)
        logger.info(f"Feature panel saved to {panel_path}")

        # Save scaler for reproducibility
        scaler_path = config.FEATURES_DIR / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

        # Save feature column list
        cols_path = config.FEATURES_DIR / "feature_columns.txt"
        with open(cols_path, "w") as f:
            for col in available_features:
                f.write(col + "\n")

        # Generate feature statistics report
        stats_df = feature_matrix.describe().T
        stats_df["n_missing_before_fill"] = panel[available_features].isna().sum()
        stats_df["pct_missing"] = (
            panel[available_features].isna().mean() * 100
        )
        stats_path = config.FEATURES_DIR / "feature_statistics.csv"
        stats_df.to_csv(stats_path)
        logger.info(f"Feature statistics saved to {stats_path}")

    return panel, scaler


def load_feature_panel() -> pd.DataFrame:
    """Load the saved feature panel from disk."""
    parquet_path = config.FEATURES_DIR / "feature_panel.parquet"
    csv_path = config.FEATURES_DIR / "feature_panel.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        logger.error("No feature panel found on disk")
        return pd.DataFrame()


def get_scaled_feature_names() -> list[str]:
    """Get the list of scaled feature column names."""
    return [f"{c}_scaled" for c in FEATURE_COLUMNS]


def get_model_input(panel: pd.DataFrame) -> tuple[np.ndarray, pd.Index, list[str]]:
    """
    Extract the model input matrix from the feature panel.

    Returns
    -------
    tuple[np.ndarray, pd.Index, list[str]]
        (X, index, feature_names) where X is the scaled feature matrix,
        index is the DatetimeIndex, and feature_names are the column names.
    """
    scaled_cols = [c for c in panel.columns if c.endswith("_scaled")]
    X = panel[scaled_cols].values
    return X, panel.index, scaled_cols


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    import argparse
    parser = argparse.ArgumentParser(description="Build feature matrices")
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()

    panel, scaler = build_all(test_mode=args.test_mode)
    if not panel.empty:
        print(f"\nFeature panel shape: {panel.shape}")
        print(f"Feature columns: {len(FEATURE_COLUMNS)}")
        print(f"Tickers: {panel['ticker'].nunique() if 'ticker' in panel.columns else 'N/A'}")
