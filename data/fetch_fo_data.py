"""
Fetch F&O (Futures & Options) data from NSE for expiry-day pinning analysis.

Downloads F&O bhavcopy to extract open interest data near expiry dates.
Used to test the hypothesis that stocks get "pinned" near strike prices
with highest open interest before F&O expiry.
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

# NSE F&O bhavcopy URL pattern
FO_BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/content/historical/"
    "DERIVATIVES/fo/fo{date}bhav.csv.zip"
)


def _get_session() -> requests.Session:
    """Create a session with proper NSE headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": config.FETCH_PARAMS["bhavcopy_user_agent"],
        "Accept": "*/*",
        "Referer": "https://www.nseindia.com/",
    })
    try:
        session.get("https://www.nseindia.com/", timeout=10)
    except Exception:
        pass
    return session


def fetch_fo_bhavcopy_single(trade_date: date,
                              session: requests.Session = None,
                              save_dir: Path = None) -> pd.DataFrame | None:
    """
    Fetch F&O bhavcopy for a single trading day.

    Parameters
    ----------
    trade_date : date
        Trading date
    session : requests.Session
        Reusable session
    save_dir : Path
        Directory to save data

    Returns
    -------
    pd.DataFrame or None
        F&O data with open interest, strike prices, etc.
    """
    save_dir = save_dir or config.RAW_DIR / "fo"
    cache_path = save_dir / f"fo_{trade_date.strftime('%Y%m%d')}.csv"

    if cache_path.exists():
        try:
            return pd.read_csv(cache_path)
        except Exception:
            pass

    if session is None:
        session = _get_session()

    date_str = trade_date.strftime("%d%b%Y").upper()
    url = FO_BHAVCOPY_URL.format(date=date_str)

    try:
        response = session.get(url, timeout=15)

        if response.status_code == 200:
            # It's a zip file containing a CSV
            import zipfile
            z = zipfile.ZipFile(io.BytesIO(response.content))
            csv_name = z.namelist()[0]
            df = pd.read_csv(z.open(csv_name))
            df.columns = df.columns.str.strip()

            # Filter for stock options (not index options, not futures)
            if "INSTRUMENT" in df.columns:
                df = df[df["INSTRUMENT"].isin(["OPTSTK", "FUTSTK"])]

            df["DATE"] = trade_date
            df.to_csv(cache_path, index=False)
            logger.info(f"  F&O bhavcopy {trade_date}: {len(df)} rows")
            return df

        elif response.status_code == 404:
            return None
        else:
            logger.warning(f"  F&O bhavcopy {trade_date}: HTTP {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"  F&O bhavcopy {trade_date}: {e}")
        return None


def get_oi_near_expiry(expiry_date: date, lookback_days: int = 5,
                       tickers: list[str] = None) -> pd.DataFrame:
    """
    Get open interest data for days leading up to an expiry date.

    This is used to identify the "max pain" strike price — the strike
    with highest combined OI — to test the pinning hypothesis.

    Parameters
    ----------
    expiry_date : date
        The F&O expiry date
    lookback_days : int
        Number of trading days before expiry to fetch
    tickers : list[str]
        Filter to these tickers (without .NS suffix)

    Returns
    -------
    pd.DataFrame
        OI data near expiry
    """
    symbols = None
    if tickers:
        symbols = {t.replace(".NS", "") for t in tickers}

    session = _get_session()
    all_data = []
    current = expiry_date - timedelta(days=lookback_days * 2)  # Account for weekends

    while current <= expiry_date:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        df = fetch_fo_bhavcopy_single(current, session)
        if df is not None and symbols:
            if "SYMBOL" in df.columns:
                df = df[df["SYMBOL"].str.strip().isin(symbols)]
            if not df.empty:
                all_data.append(df)

        current += timedelta(days=1)
        time.sleep(config.FETCH_PARAMS["bhavcopy_sleep"])

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def compute_max_pain(oi_data: pd.DataFrame, symbol: str,
                     expiry_date: date) -> float | None:
    """
    Compute the max pain strike price for a symbol on expiry day.

    Max pain = the strike price where option writers have minimum loss,
    i.e., total OI (calls + puts) is maximized at that strike.

    Parameters
    ----------
    oi_data : pd.DataFrame
        F&O data containing SYMBOL, OPTION_TYP, STRIKE_PR, OPEN_INT
    symbol : str
        Stock symbol (without .NS)
    expiry_date : date
        The expiry date

    Returns
    -------
    float or None
        Max pain strike price
    """
    df = oi_data[
        (oi_data["SYMBOL"].str.strip() == symbol) &
        (oi_data["DATE"] == expiry_date)
    ]

    if df.empty or "STRIKE_PR" not in df.columns:
        return None

    # Group by strike price, sum OI for calls and puts
    if "OPTION_TYP" in df.columns:
        oi_by_strike = df.groupby("STRIKE_PR")["OPEN_INT"].sum()
        if oi_by_strike.empty:
            return None
        return oi_by_strike.idxmax()

    return None


def fetch_all_expiry_oi(tickers: list[str] = None,
                        expiry_dates: list[date] = None) -> dict:
    """
    Fetch OI data near all expiry dates for analysis.

    Returns a dict mapping expiry_date -> DataFrame of OI data.
    """
    tickers = tickers or list(config.FNO_ELIGIBLE)
    expiry_dates = expiry_dates or config.FNO_EXPIRY_DATES

    results = {}
    for i, exp_date in enumerate(expiry_dates):
        logger.info(f"[{i+1}/{len(expiry_dates)}] Fetching OI near {exp_date}")
        oi_data = get_oi_near_expiry(exp_date, lookback_days=3, tickers=tickers)
        if not oi_data.empty:
            results[exp_date] = oi_data

    logger.info(f"Fetched OI data for {len(results)}/{len(expiry_dates)} expiry dates")
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # Quick test: fetch OI near the most recent expiry
    recent_expiries = [d for d in config.FNO_EXPIRY_DATES if d <= date.today()]
    if recent_expiries:
        last_expiry = recent_expiries[-1]
        print(f"Fetching OI data near {last_expiry}")
        oi = get_oi_near_expiry(last_expiry, tickers=config.TEST_MODE_TICKERS)
        print(f"Got {len(oi)} rows")
        if not oi.empty:
            print(oi.head())
