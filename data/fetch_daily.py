"""
Fetch daily OHLCV data from Yahoo Finance for all tickers in the universe.

Uses yfinance library with rate limiting and error handling.
Saves raw CSVs to data/raw/daily/{TICKER}.csv
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def fetch_single_ticker(ticker: str, start: str = None, end: str = None,
                        save_dir: Path = None) -> pd.DataFrame | None:
    """
    Fetch daily OHLCV data for a single ticker from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g., 'RELIANCE.NS')
    start : str
        Start date in YYYY-MM-DD format
    end : str
        End date in YYYY-MM-DD format
    save_dir : Path
        Directory to save the CSV file

    Returns
    -------
    pd.DataFrame or None
        OHLCV DataFrame with DatetimeIndex, or None if fetch failed
    """
    start = start or config.START_DATE
    end = end or config.END_DATE
    save_dir = save_dir or config.RAW_DIR / "daily"

    logger.info(f"Fetching daily data for {ticker} ({start} to {end})")

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,  # Use adjusted prices
        )

        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return None

        # yfinance may return MultiIndex columns when downloading single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel("Ticker")

        # Validate we have the expected columns
        expected_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not expected_cols.issubset(set(df.columns)):
            logger.warning(
                f"Missing columns for {ticker}: "
                f"expected {expected_cols}, got {set(df.columns)}"
            )
            return None

        # Basic data quality checks
        n_rows = len(df)
        n_missing = df[["Open", "High", "Low", "Close"]].isna().any(axis=1).sum()
        n_zero_vol = (df["Volume"] == 0).sum()

        logger.info(
            f"  {ticker}: {n_rows} rows, {n_missing} missing price rows, "
            f"{n_zero_vol} zero-volume days"
        )

        # Check for suspiciously large gaps
        if n_rows > 1:
            date_diffs = df.index.to_series().diff().dt.days
            max_gap = date_diffs.max()
            if max_gap > config.FETCH_PARAMS["max_gap_days"] * 2:
                logger.warning(
                    f"  {ticker}: Large gap detected — {max_gap} calendar days"
                )

        # Save to CSV
        safe_ticker = ticker.replace("^", "IDX_")
        filepath = save_dir / f"{safe_ticker}.csv"
        df.to_csv(filepath)
        logger.info(f"  Saved to {filepath}")

        return df

    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        return None


def fetch_all(tickers: list[str] = None, start: str = None, end: str = None,
              dry_run: bool = False) -> dict[str, pd.DataFrame]:
    """
    Fetch daily data for all tickers in the universe.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols. Defaults to config.ALL_TICKERS + INDEX_TICKERS
    start : str
        Start date
    end : str
        End date
    dry_run : bool
        If True, only validate config without downloading

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from ticker to its OHLCV DataFrame
    """
    tickers = tickers or (config.ALL_TICKERS + config.INDEX_TICKERS)
    start = start or config.START_DATE
    end = end or config.END_DATE

    if dry_run:
        logger.info(f"DRY RUN: Would fetch {len(tickers)} tickers: {tickers}")
        logger.info(f"  Date range: {start} to {end}")
        return {}

    logger.info(f"Fetching daily data for {len(tickers)} tickers")
    results = {}
    failed = []

    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] {ticker}")
        df = fetch_single_ticker(ticker, start, end)

        if df is not None:
            results[ticker] = df
        else:
            failed.append(ticker)

        # Rate limiting
        if i < len(tickers) - 1:
            time.sleep(config.FETCH_PARAMS["yfinance_sleep"])

    # Summary
    logger.info(f"\nFetch complete: {len(results)}/{len(tickers)} succeeded")
    if failed:
        logger.warning(f"Failed tickers: {failed}")

    return results


def load_cached(tickers: list[str] = None) -> dict[str, pd.DataFrame]:
    """
    Load previously downloaded daily data from CSV files.

    Parameters
    ----------
    tickers : list[str]
        Tickers to load. Defaults to all available files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from ticker to OHLCV DataFrame
    """
    data_dir = config.RAW_DIR / "daily"
    results = {}

    if tickers is None:
        # Load all available files
        for f in data_dir.glob("*.csv"):
            ticker = f.stem.replace("IDX_", "^")
            try:
                df = pd.read_csv(f, index_col=0, parse_dates=True)
                results[ticker] = df
            except Exception as e:
                logger.error(f"Failed to load {f}: {e}")
    else:
        for ticker in tickers:
            safe_ticker = ticker.replace("^", "IDX_")
            filepath = data_dir / f"{safe_ticker}.csv"
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    results[ticker] = df
                except Exception as e:
                    logger.error(f"Failed to load {filepath}: {e}")
            else:
                logger.warning(f"No cached data for {ticker}")

    logger.info(f"Loaded {len(results)} tickers from cache")
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    import argparse
    parser = argparse.ArgumentParser(description="Fetch daily OHLCV data")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config without downloading")
    parser.add_argument("--test-mode", action="store_true",
                        help="Fetch only test tickers")
    args = parser.parse_args()

    tickers = config.TEST_MODE_TICKERS if args.test_mode else None
    fetch_all(tickers=tickers, dry_run=args.dry_run)
