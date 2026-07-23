"""
Fetch intraday OHLCV data from Yahoo Finance.

- 5-min bars: ~60 days lookback
- 1-min bars: ~7 days lookback

Used for intraday signature analysis (pump-and-dump intraday patterns,
spoofing-style reversals).
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def fetch_intraday_single(ticker: str, interval: str = "5m",
                          period: str = None) -> pd.DataFrame | None:
    """
    Fetch intraday data for a single ticker.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol
    interval : str
        Candle interval: '1m', '5m', '15m', '30m', '1h'
    period : str
        Lookback period. If None, uses max available for the interval.

    Returns
    -------
    pd.DataFrame or None
    """
    if period is None:
        if interval == "1m":
            period = "7d"
        elif interval in ("5m", "15m"):
            period = "60d"
        elif interval in ("30m", "1h"):
            period = "60d"
        else:
            period = "60d"

    logger.info(f"Fetching {interval} data for {ticker} (period={period})")

    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )

        if df.empty:
            logger.warning(f"No intraday data returned for {ticker} ({interval})")
            return None

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel("Ticker")

        n_rows = len(df)
        n_days = df.index.normalize().nunique()
        logger.info(f"  {ticker} ({interval}): {n_rows} bars over {n_days} days")

        # Save
        save_dir = config.RAW_DIR / "intraday"
        safe_ticker = ticker.replace("^", "IDX_")
        filepath = save_dir / f"{safe_ticker}_{interval}.csv"
        df.to_csv(filepath)
        logger.info(f"  Saved to {filepath}")

        return df

    except Exception as e:
        logger.error(f"Failed to fetch intraday {ticker} ({interval}): {e}")
        return None


def fetch_all_intraday(tickers: list[str] = None,
                       intervals: list[str] = None) -> dict[str, pd.DataFrame]:
    """
    Fetch intraday data for all tickers at specified intervals.

    Parameters
    ----------
    tickers : list[str]
        Tickers to fetch. Defaults to config.ALL_TICKERS.
    intervals : list[str]
        Intervals to fetch. Defaults to ['5m'].

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are '{ticker}_{interval}', values are DataFrames
    """
    tickers = tickers or config.ALL_TICKERS
    intervals = intervals or ["5m"]
    results = {}

    total = len(tickers) * len(intervals)
    count = 0

    for interval in intervals:
        for ticker in tickers:
            count += 1
            logger.info(f"[{count}/{total}] {ticker} @ {interval}")

            df = fetch_intraday_single(ticker, interval)
            if df is not None:
                key = f"{ticker}_{interval}"
                results[key] = df

            time.sleep(config.FETCH_PARAMS["yfinance_sleep"])

    logger.info(f"Intraday fetch complete: {len(results)}/{total} succeeded")
    return results


def load_cached_intraday(ticker: str, interval: str = "5m") -> pd.DataFrame | None:
    """Load cached intraday data from CSV."""
    safe_ticker = ticker.replace("^", "IDX_")
    filepath = config.RAW_DIR / "intraday" / f"{safe_ticker}_{interval}.csv"
    if filepath.exists():
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
    return None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    import argparse
    parser = argparse.ArgumentParser(description="Fetch intraday data")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--interval", default="5m",
                        choices=["1m", "5m", "15m"])
    args = parser.parse_args()

    tickers = config.TEST_MODE_TICKERS if args.test_mode else None
    fetch_all_intraday(tickers=tickers, intervals=[args.interval])
