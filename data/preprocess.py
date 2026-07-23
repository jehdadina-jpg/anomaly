"""
Preprocess and merge raw data sources into clean, analysis-ready DataFrames.

Merges:
1. yfinance daily OHLCV (adjusted prices)
2. NSE bhavcopy delivery percentage
3. Index data for beta calculation

Handles stock splits, fills minor gaps, and generates data quality reports.
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from data.fetch_daily import load_cached
from data.fetch_bhavcopy import load_delivery_data

logger = logging.getLogger(__name__)


def preprocess_single(ticker: str, daily_df: pd.DataFrame,
                      delivery_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Clean and enrich a single ticker's daily OHLCV data.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g., 'RELIANCE.NS')
    daily_df : pd.DataFrame
        Raw OHLCV data from yfinance
    delivery_df : pd.DataFrame
        Combined bhavcopy data (optional)

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with OHLCV + delivery_pct + metadata columns
    """
    df = daily_df.copy()

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Sort by date
    df = df.sort_index()

    # Remove exact duplicates
    df = df[~df.index.duplicated(keep="first")]

    # ---- Handle missing data ----
    # Forward-fill small gaps (max 2 consecutive days)
    price_cols = ["Open", "High", "Low", "Close"]
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].ffill(limit=2)

    # Volume: fill with 0 if missing (but flag it)
    if "Volume" in df.columns:
        df["volume_missing"] = df["Volume"].isna().astype(int)
        df["Volume"] = df["Volume"].fillna(0)

    # ---- Merge delivery data ----
    symbol = ticker.replace(".NS", "")
    if delivery_df is not None and not delivery_df.empty:
        del_data = delivery_df[
            delivery_df["SYMBOL"].str.strip() == symbol
        ].copy()

        if not del_data.empty:
            del_data["DATE"] = pd.to_datetime(del_data["DATE"])
            del_data = del_data.set_index("DATE")

            # Get delivery percentage
            if "DELIVERY_PCT" in del_data.columns:
                df = df.join(
                    del_data[["DELIVERY_PCT"]].rename(
                        columns={"DELIVERY_PCT": "delivery_pct"}
                    ),
                    how="left",
                )
            # Also grab traded quantity if available
            if "TTL_TRD_QNTY" in del_data.columns:
                df = df.join(
                    del_data[["TTL_TRD_QNTY"]].rename(
                        columns={"TTL_TRD_QNTY": "nse_traded_qty"}
                    ),
                    how="left",
                )
        else:
            df["delivery_pct"] = np.nan
    else:
        df["delivery_pct"] = np.nan

    # ---- Add metadata columns ----
    df["ticker"] = ticker
    df["sector"] = config.TICKER_TO_SECTOR.get(ticker, "Other")
    df["is_fno"] = ticker in config.FNO_ELIGIBLE
    df["trading_day"] = df.index.dayofweek  # 0=Monday
    df["month"] = df.index.month
    df["year"] = df.index.year

    # ---- Basic derived columns ----
    df["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["daily_range"] = df["High"] - df["Low"]
    df["daily_range_pct"] = df["daily_range"] / df["Close"]

    return df


def preprocess_all(daily_data: dict[str, pd.DataFrame] = None,
                   delivery_df: pd.DataFrame = None,
                   save: bool = True) -> dict[str, pd.DataFrame]:
    """
    Preprocess all tickers and merge with delivery data.

    Parameters
    ----------
    daily_data : dict
        Mapping of ticker -> OHLCV DataFrame. If None, loads from cache.
    delivery_df : pd.DataFrame
        Combined bhavcopy data. If None, loads from cache.
    save : bool
        Whether to save processed DataFrames to disk.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker -> processed DataFrame
    """
    if daily_data is None:
        daily_data = load_cached()

    if delivery_df is None:
        try:
            delivery_df = load_delivery_data()
        except Exception as e:
            logger.warning(f"Could not load delivery data: {e}")
            delivery_df = pd.DataFrame()

    results = {}
    quality_report = []

    for ticker, raw_df in daily_data.items():
        if ticker.startswith("^"):
            # Index tickers — simpler processing
            df = raw_df.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df["ticker"] = ticker
            results[ticker] = df

            if save:
                safe_name = ticker.replace("^", "IDX_")
                df.to_csv(config.PROCESSED_DIR / f"{safe_name}.csv")
            continue

        logger.info(f"Preprocessing {ticker}")
        df = preprocess_single(ticker, raw_df, delivery_df)
        results[ticker] = df

        # Quality report
        quality_report.append({
            "ticker": ticker,
            "start_date": df.index.min(),
            "end_date": df.index.max(),
            "n_rows": len(df),
            "pct_delivery_available": df["delivery_pct"].notna().mean() * 100,
            "pct_zero_volume": (df["Volume"] == 0).mean() * 100,
            "mean_delivery_pct": df["delivery_pct"].mean(),
        })

        if save:
            safe_name = ticker.replace("^", "IDX_").replace(".", "_")
            df.to_csv(config.PROCESSED_DIR / f"{safe_name}.csv")

    # Save quality report
    if quality_report:
        qr_df = pd.DataFrame(quality_report)
        qr_path = config.PROCESSED_DIR / "data_quality_report.csv"
        qr_df.to_csv(qr_path, index=False)
        logger.info(f"Data quality report saved to {qr_path}")

        # Print summary
        logger.info("\n=== Data Quality Summary ===")
        logger.info(f"Tickers processed: {len(qr_df)}")
        logger.info(
            f"Avg delivery data coverage: "
            f"{qr_df['pct_delivery_available'].mean():.1f}%"
        )
        logger.info(
            f"Avg zero-volume days: "
            f"{qr_df['pct_zero_volume'].mean():.1f}%"
        )

    return results


def build_combined_panel(processed_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Stack all processed tickers into a single panel DataFrame.

    Returns a DataFrame with (Date, ticker) as a MultiIndex,
    containing all OHLCV + delivery + metadata columns.
    """
    frames = []
    for ticker, df in processed_data.items():
        if ticker.startswith("^"):
            continue  # Skip indices
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames)
    panel.index.name = "Date"

    logger.info(
        f"Combined panel: {len(panel)} rows, "
        f"{panel['ticker'].nunique()} tickers, "
        f"{panel.index.nunique()} unique dates"
    )

    # Save
    panel_path = config.PROCESSED_DIR / "panel.csv"
    panel.to_csv(panel_path)
    logger.info(f"Panel saved to {panel_path}")

    return panel


def load_processed(ticker: str = None) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Load processed data from disk.

    If ticker is specified, returns a single DataFrame.
    Otherwise returns a dict of all processed tickers.
    """
    if ticker:
        safe_name = ticker.replace("^", "IDX_").replace(".", "_")
        path = config.PROCESSED_DIR / f"{safe_name}.csv"
        if path.exists():
            return pd.read_csv(path, index_col=0, parse_dates=True)
        logger.warning(f"No processed data for {ticker}")
        return pd.DataFrame()

    # Load all
    results = {}
    for f in config.PROCESSED_DIR.glob("*.csv"):
        if f.name in ("data_quality_report.csv", "panel.csv",
                       "bhavcopy_combined.csv"):
            continue
        ticker_name = f.stem.replace("IDX_", "^").replace("_NS", ".NS")
        try:
            results[ticker_name] = pd.read_csv(f, index_col=0, parse_dates=True)
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    processed = preprocess_all()
    if processed:
        panel = build_combined_panel(processed)
        print(f"\nPanel shape: {panel.shape}")
        print(f"Columns: {list(panel.columns)}")
