"""
Build the full multi-stock panel used for price prediction.

Joins daily OHLCV (yfinance) with NSE bhavcopy microstructure (delivery %,
trade counts, turnover) into a single tidy panel indexed by (Date, ticker).

The bhavcopy join is the reason this module exists: delivery percentage and
trades-per-share are genuinely informative in Indian equities and are not
available from the OHLCV feed.

Usage
-----
    python -m data.build_panel            # writes data/features/panel.parquet
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

PANEL_PATH = config.FEATURES_DIR / "panel.csv"

# Sector map for the 48 tickers actually present in data/raw/daily.
# Sector membership drives the relative-strength features, so it needs to
# reflect real GICS-style groupings rather than a single "Nifty 50" bucket.
SECTOR_MAP = {
    "ABB.NS": "Capital Goods", "CUMMINSIND.NS": "Capital Goods",
    "BHARATFORG.NS": "Capital Goods", "BOSCHLTD.NS": "Auto",
    "ACC.NS": "Cement", "AMBUJACEM.NS": "Cement", "GRASIM.NS": "Cement",
    "ASIANPAINT.NS": "Consumer", "BRITANNIA.NS": "Consumer",
    "COLPAL.NS": "Consumer", "GODREJCP.NS": "Consumer",
    "MARICO.NS": "Consumer", "TATACONSUM.NS": "Consumer", "ITC.NS": "Consumer",
    "AUBANK.NS": "Banking", "BANKBARODA.NS": "Banking",
    "HDFCBANK.NS": "Banking", "SBIN.NS": "Banking",
    "HDFCAMC.NS": "Financials", "ICICIGI.NS": "Financials",
    "ICICIPRULI.NS": "Financials",
    "BAJAJ-AUTO.NS": "Auto", "MOTHERSON.NS": "Auto", "TVSMOTOR.NS": "Auto",
    "BHARTIARTL.NS": "Telecom", "NAUKRI.NS": "Tech",
    "COFORGE.NS": "IT", "HCLTECH.NS": "IT", "INFY.NS": "IT",
    "TATAELXSI.NS": "IT", "TCS.NS": "IT", "WIPRO.NS": "IT",
    "CIPLA.NS": "Pharma", "DRREDDY.NS": "Pharma",
    "SUNPHARMA.NS": "Pharma", "ZYDUSLIFE.NS": "Pharma",
    "COALINDIA.NS": "Energy", "GAIL.NS": "Energy", "HINDPETRO.NS": "Energy",
    "NTPC.NS": "Utilities", "POWERGRID.NS": "Utilities",
    "HINDALCO.NS": "Metals", "VEDL.NS": "Metals",
    "PIDILITIND.NS": "Chemicals", "PIIND.NS": "Chemicals",
    "CONCOR.NS": "Logistics", "INDIGO.NS": "Logistics",
    "TITAN.NS": "Consumer",
}


def load_bhavcopy() -> pd.DataFrame:
    """Load combined bhavcopy and return per-(symbol, date) microstructure."""
    path = config.PROCESSED_DIR / "bhavcopy_combined.csv"
    bc = pd.read_csv(path, low_memory=False)
    bc.columns = [c.strip() for c in bc.columns]

    bc["SYMBOL"] = bc["SYMBOL"].astype(str).str.strip()
    bc["SERIES"] = bc["SERIES"].astype(str).str.strip()
    bc = bc[bc["SERIES"] == "EQ"]
    bc["Date"] = pd.to_datetime(bc["DATE"])

    # DELIV_QTY / DELIV_PER arrive with padding and occasional '-' placeholders.
    for col in ["DELIVERY_PCT", "DELIV_QTY", "NO_OF_TRADES", "TTL_TRD_QNTY",
                "TURNOVER_LACS"]:
        bc[col] = pd.to_numeric(
            bc[col].astype(str).str.strip().replace({"-": None, "": None}),
            errors="coerce",
        )

    out = bc[["SYMBOL", "Date", "DELIVERY_PCT", "DELIV_QTY", "NO_OF_TRADES",
              "TTL_TRD_QNTY", "TURNOVER_LACS"]].rename(columns={
        "SYMBOL": "symbol",
        "DELIVERY_PCT": "delivery_pct",
        "DELIV_QTY": "delivery_qty",
        "NO_OF_TRADES": "n_trades",
        "TTL_TRD_QNTY": "nse_traded_qty",
        "TURNOVER_LACS": "turnover_lacs",
    })
    # A symbol can appear twice on a date across series edge cases; keep one.
    out = out.drop_duplicates(subset=["symbol", "Date"], keep="first")
    logger.info("Bhavcopy: %d rows, %d symbols, %s -> %s",
                len(out), out["symbol"].nunique(),
                out["Date"].min().date(), out["Date"].max().date())
    return out


def build_panel(save: bool = True) -> pd.DataFrame:
    """Build the (Date, ticker) panel of OHLCV + microstructure."""
    daily_dir = config.RAW_DIR / "daily"
    bhav = load_bhavcopy()

    frames = []
    for csv_path in sorted(daily_dir.glob("*.csv")):
        ticker = csv_path.stem
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        if df.empty:
            logger.warning("Skipping empty file: %s", csv_path.name)
            continue
        df["ticker"] = ticker
        df["symbol"] = ticker.replace(".NS", "")
        df["sector"] = SECTOR_MAP.get(ticker, "Other")
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.merge(bhav, on=["symbol", "Date"], how="left")

    # Drop rows with no usable price, then sort into strict panel order.
    panel = panel.dropna(subset=["Close"])
    panel = panel.sort_values(["ticker", "Date"]).reset_index(drop=True)

    coverage = panel["delivery_pct"].notna().mean()
    logger.info("Panel: %d rows, %d tickers, %d sectors | delivery_pct coverage %.1f%%",
                len(panel), panel["ticker"].nunique(),
                panel["sector"].nunique(), coverage * 100)

    if save:
        config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        panel.to_csv(PANEL_PATH, index=False)
        logger.info("Wrote %s", PANEL_PATH)

    return panel


def load_panel() -> pd.DataFrame:
    """Load the cached panel, building it if absent."""
    if PANEL_PATH.exists():
        return pd.read_csv(PANEL_PATH, parse_dates=["Date"])
    return build_panel(save=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    p = build_panel()
    print(p.head())
    print("\nColumns:", list(p.columns))
    print("\nPer-ticker rows:")
    print(p.groupby("ticker").size().describe())
