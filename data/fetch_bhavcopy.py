"""
Fetch NSE Bhavcopy data — specifically the "Security-wise Price Volume
and Deliverable Position" report, which includes delivery percentage.

Delivery percentage (deliverable qty / traded qty) is a key India-specific
feature: low delivery % + price spike is a classic pump-and-dump tell
that US datasets don't even provide.
"""

import sys
import time
import logging
import io
from pathlib import Path
from datetime import datetime, timedelta, date

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# NSE archives URL patterns for security-wise delivery data
BHAVCOPY_URL_TEMPLATES = [
    # Primary: Full bhavcopy with delivery data
    "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv",
    # Alternate format
    "https://nsearchives.nseindia.com/archives/equities/bhavcopy/pr/PR{date2}.zip",
]


def _get_session() -> requests.Session:
    """Create a requests session with proper headers to avoid NSE blocks."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": config.FETCH_PARAMS["bhavcopy_user_agent"],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Referer": "https://www.nseindia.com/",
    })

    # First visit the main page to establish a session cookie
    try:
        session.get("https://www.nseindia.com/", timeout=10)
    except Exception:
        pass  # Not critical if this fails

    return session


def fetch_bhavcopy_single(trade_date: date, session: requests.Session = None,
                          save_dir: Path = None) -> pd.DataFrame | None:
    """
    Fetch bhavcopy for a single trading day.

    Parameters
    ----------
    trade_date : date
        The trading date to fetch
    session : requests.Session
        Reusable session with proper headers
    save_dir : Path
        Directory to save the CSV

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns including SYMBOL, DELIVERY_QTY, TRADED_QTY, etc.
    """
    save_dir = save_dir or config.RAW_DIR / "bhavcopy"

    # Check cache first
    cache_path = save_dir / f"bhavcopy_{trade_date.strftime('%Y%m%d')}.csv"
    if cache_path.exists():
        try:
            return pd.read_csv(cache_path)
        except Exception:
            pass  # Re-download if cache is corrupted

    if session is None:
        session = _get_session()

    # Try the primary URL format
    date_str = trade_date.strftime("%d%m%Y")
    url = BHAVCOPY_URL_TEMPLATES[0].format(date=date_str)

    try:
        response = session.get(url, timeout=15)

        if response.status_code == 200:
            # Parse the CSV
            df = pd.read_csv(io.StringIO(response.text))

            # Clean column names (NSE CSVs have trailing spaces sometimes)
            df.columns = df.columns.str.strip()

            # Filter for equity segment only
            if "SERIES" in df.columns:
                df = df[df["SERIES"].str.strip().isin(["EQ", "BE", "BZ"])]

            # Compute delivery percentage if raw columns exist
            if "DELIV_QTY" in df.columns and "TTL_TRD_QNTY" in df.columns:
                df["DELIVERY_PCT"] = (
                    pd.to_numeric(df["DELIV_QTY"], errors="coerce") /
                    pd.to_numeric(df["TTL_TRD_QNTY"], errors="coerce")
                )
            elif "DELIV_PER" in df.columns:
                df["DELIVERY_PCT"] = pd.to_numeric(
                    df["DELIV_PER"], errors="coerce"
                ) / 100.0

            # Add date column
            df["DATE"] = trade_date

            # Save to cache
            df.to_csv(cache_path, index=False)
            logger.info(
                f"  Bhavcopy {trade_date}: {len(df)} securities"
            )
            return df

        elif response.status_code == 404:
            # Probably a holiday / non-trading day
            logger.debug(f"  No bhavcopy for {trade_date} (404)")
            return None
        else:
            logger.warning(
                f"  Bhavcopy {trade_date}: HTTP {response.status_code}"
            )
            return None

    except requests.exceptions.Timeout:
        logger.warning(f"  Bhavcopy {trade_date}: timeout")
        return None
    except Exception as e:
        logger.error(f"  Bhavcopy {trade_date}: {e}")
        return None


def fetch_bhavcopy_range(start_date: date = None, end_date: date = None,
                         tickers: list[str] = None) -> pd.DataFrame:
    """
    Fetch bhavcopy data for a date range and filter for our ticker universe.

    Parameters
    ----------
    start_date : date
        Start of range
    end_date : date
        End of range
    tickers : list[str]
        If provided, filter to these tickers (without .NS suffix)

    Returns
    -------
    pd.DataFrame
        Combined bhavcopy data with delivery percentages
    """
    start_date = start_date or datetime.strptime(config.START_DATE, "%Y-%m-%d").date()
    end_date = end_date or datetime.strptime(config.END_DATE, "%Y-%m-%d").date()

    # Strip .NS suffix for matching against bhavcopy SYMBOL column
    if tickers:
        symbols = {t.replace(".NS", "") for t in tickers}
    else:
        symbols = {t.replace(".NS", "") for t in config.ALL_TICKERS
                    if not t.startswith("^")}

    session = _get_session()
    all_data = []
    current = start_date
    trading_days = 0
    fetch_days = 0

    logger.info(
        f"Fetching bhavcopy from {start_date} to {end_date} "
        f"for {len(symbols)} symbols"
    )

    while current <= end_date:
        # Skip weekends
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        fetch_days += 1
        df = fetch_bhavcopy_single(current, session)

        if df is not None:
            trading_days += 1
            # Filter for our symbols
            if "SYMBOL" in df.columns:
                df_filtered = df[df["SYMBOL"].str.strip().isin(symbols)]
                if not df_filtered.empty:
                    all_data.append(df_filtered)

        current += timedelta(days=1)
        time.sleep(config.FETCH_PARAMS["bhavcopy_sleep"])

        # Log progress every 50 days
        if fetch_days % 50 == 0:
            logger.info(
                f"  Progress: {fetch_days} weekdays checked, "
                f"{trading_days} trading days found, "
                f"{len(all_data)} data chunks collected"
            )

    if not all_data:
        logger.warning("No bhavcopy data collected")
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    logger.info(
        f"Bhavcopy fetch complete: {len(result)} rows "
        f"over {trading_days} trading days"
    )

    # Save combined result
    output_path = config.PROCESSED_DIR / "bhavcopy_combined.csv"
    result.to_csv(output_path, index=False)
    logger.info(f"Saved combined bhavcopy to {output_path}")

    return result


def load_delivery_data() -> pd.DataFrame:
    """
    Load the combined bhavcopy data with delivery percentages.

    Returns a DataFrame with columns: SYMBOL, DATE, DELIVERY_PCT, ...
    """
    path = config.PROCESSED_DIR / "bhavcopy_combined.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["DATE"])
        logger.info(f"Loaded {len(df)} bhavcopy rows from cache")
        return df

    logger.info("No cached bhavcopy data; fetching...")
    return fetch_bhavcopy_range()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    import argparse
    parser = argparse.ArgumentParser(description="Fetch NSE bhavcopy data")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--test-mode", action="store_true",
                        help="Fetch only last 30 days")
    args = parser.parse_args()

    if args.test_mode:
        end = date.today()
        start = end - timedelta(days=30)
        fetch_bhavcopy_range(start, end, config.TEST_MODE_TICKERS)
    else:
        start = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None
        end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None
        fetch_bhavcopy_range(start, end)
