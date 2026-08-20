"""
FastAPI Backend for NSE Anomaly Detection Terminal.

Provides live/last prices (Yahoo Finance), anomaly results, OHLCV history,
and feature explanations for the web frontend.
"""

import sys
import logging
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))
import config

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

IST = ZoneInfo("Asia/Kolkata")

STATE = {
    "universe": [],
    "anomalies": {},
    "quotes": {},
    "predictions": {},      # ticker -> buyScore / buyLabel / buyReason / signals
    "prediction_meta": {},  # as-of date, horizon, model provenance
    "last_updated": None,
}


def _format_ist(dt: datetime) -> str:
    """Format a datetime to IST string."""
    if dt is None:
        return datetime.now(IST).strftime("%d-%b-%Y %H:%M:%S IST")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST).strftime("%d-%b-%Y %H:%M:%S IST")


def _safe_float(val) -> float | None:
    """Safely convert to float, returning None on failure."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_attr(obj, *attrs) -> float | None:
    """Try to get a float value from multiple attribute names on an object."""
    for attr in attrs:
        try:
            val = getattr(obj, attr, None)
            if val is not None:
                result = _safe_float(val)
                if result is not None:
                    return result
        except Exception:
            continue
    return None


def _load_from_cached_daily() -> tuple[list, dict]:
    """Build universe from cached daily CSVs when pipeline results are missing."""
    daily_dir = config.RAW_DIR / "daily"
    universe = []
    for csv_path in sorted(daily_dir.glob("*.csv")):
        ticker = csv_path.stem
        try:
            df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
        except Exception:
            df = pd.read_csv(csv_path)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
        if df.empty:
            continue
        row = df.iloc[-1]
        close = _safe_float(row.get("Close"))
        open_px = _safe_float(row.get("Open", close))
        if close is None:
            continue
        chg_pct = ((close / open_px) - 1) * 100 if open_px else 0.0
        as_of = df.index[-1]
        if hasattr(as_of, "to_pydatetime"):
            as_of = as_of.to_pydatetime()
        universe.append({
            "ticker": ticker,
            "name": ticker.replace(".NS", ""),
            "sector": config.TICKER_TO_SECTOR.get(ticker, "Nifty 50"),
            "price": close,
            "priorClose": close,
            "chg": f"{chg_pct:+.2f}%",
            "deliveryPct": 0.0,
            "isAnomaly": False,
            "sebiOrder": next(
                (desc for start, end, desc in config.SEBI_CASES.get(ticker, [])),
                None,
            ),
            "agreementCount": 0,
            "combinedScore": 0.0,
            "priceAsOf": _format_ist(as_of),
            "isLive": False,
            "dataDate": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else None,
        })
    logger.info("Built fallback universe from %d cached daily files.", len(universe))
    return universe, {}


def load_pipeline_results():
    """Load latest results from the pipeline (assuming main.py was run)."""
    results_path = config.RESULTS_DIR / "anomaly_results.csv"
    if not results_path.exists():
        logger.warning("No results at %s — using cached daily data.", results_path)
        universe, anomalies = _load_from_cached_daily()
        STATE["universe"] = universe
        STATE["anomalies"] = anomalies
        return

    df = pd.read_csv(results_path, parse_dates=["Date"])
    latest_idx = df.groupby("ticker")["Date"].idxmax()
    latest_df = df.loc[latest_idx]

    universe = []
    anomalies = {}

    for _, row in latest_df.iterrows():
        ticker = row["ticker"]
        is_anomaly = bool(row.get("consensus_anomaly", 0))
        close = _safe_float(row["Close"]) or 0.0
        open_px = _safe_float(row.get("Open", close)) or close
        chg_pct = ((close / open_px) - 1) * 100 if open_px else 0.0
        as_of = row["Date"]
        if hasattr(as_of, "to_pydatetime"):
            as_of = as_of.to_pydatetime()

        stock_info = {
            "ticker": ticker,
            "name": ticker.replace(".NS", ""),
            "sector": config.TICKER_TO_SECTOR.get(ticker, "Nifty 50"),
            "price": close,
            "priorClose": close,
            "chg": f"{chg_pct:+.2f}%",
            "deliveryPct": round(float(row.get("delivery_pct", 0) or 0) * 100, 1),
            "isAnomaly": is_anomaly,
            "sebiOrder": next(
                (desc for start, end, desc in config.SEBI_CASES.get(ticker, [])),
                None,
            ),
            "agreementCount": int(row.get("agreement_count", 0)),
            "combinedScore": float(row.get("combined_score", 0)),
            "priceAsOf": _format_ist(as_of),
            "isLive": False,
            "dataDate": str(row["Date"].date()) if hasattr(row["Date"], "date") else None,
        }
        universe.append(stock_info)

        if is_anomaly:
            explanations = []
            vol_z = float(row.get("volume_zscore_20", 0) or 0)
            if vol_z > 2.0:
                explanations.append(
                    f"Volume surged {vol_z:.1f} standard deviations above 20-day average."
                )
            del_pct = float(row.get("delivery_pct", 1.0) or 1.0)
            if del_pct < 0.3:
                explanations.append(
                    f"Delivery percentage dropped to critically low levels ({del_pct * 100:.1f}%)."
                )
            rev = float(row.get("intraday_reversal", 0) or 0)
            if rev > 0.6:
                explanations.append(f"Extreme intraday price reversal ({rev * 100:.0f}% from high).")
            circ = float(row.get("circuit_proximity", 0) or 0)
            if circ > 0.9:
                explanations.append("Price closed extremely close to the regulatory circuit limit.")
            if not explanations:
                explanations.append(
                    "Detected complex multi-dimensional anomaly pattern by Autoencoder/LOF."
                )
            anomalies[ticker] = {
                "explanations": explanations,
                "feature_values": {
                    "Volume Z-Score": f"{vol_z:.2f} σ",
                    "Delivery %": f"{del_pct * 100:.1f}%",
                    "Circuit Proximity": f"{circ * 100:.1f}%",
                    "Intraday Reversal": f"{rev:.2f}",
                },
            }

    STATE["universe"] = universe
    STATE["anomalies"] = anomalies
    logger.info("Loaded pipeline results for %d stocks.", len(universe))


def _quote_from_ticker_obj(ticker: str, t: yf.Ticker) -> dict | None:
    """
    Extract last/live price and metadata from a yfinance Ticker.

    fast_info uses ATTRIBUTE access (fi.last_price), NOT dict access (fi.get()).
    This was the root cause of silent failures in the original code.
    """
    price = None
    as_of = None
    is_live = False
    prior_close = None

    # --- Attempt 1: fast_info (attribute-based access) ---
    try:
        fi = t.fast_info
        # Attribute access — NOT fi.get("lastPrice")
        price = _safe_attr(fi, "last_price", "lastPrice",
                           "regular_market_price", "regularMarketPrice")
        prior_close = _safe_attr(fi, "previous_close", "previousClose",
                                 "regularMarketPreviousClose")

        # Get market timestamp
        market_time = None
        for attr in ("regular_market_time", "regularMarketTime",
                     "last_market_time", "lastMarketTime"):
            try:
                mt = getattr(fi, attr, None)
                if mt is not None:
                    market_time = mt
                    break
            except Exception:
                continue

        if market_time is not None:
            if isinstance(market_time, (int, float)):
                as_of = datetime.fromtimestamp(market_time, tz=timezone.utc)
            elif isinstance(market_time, datetime):
                as_of = market_time
            is_live = True
    except Exception as exc:
        logger.debug("fast_info failed for %s: %s", ticker, exc)

    # --- Attempt 2: info dict fallback ---
    if price is None:
        try:
            info = t.info
            if isinstance(info, dict):
                price = _safe_float(info.get("regularMarketPrice")) or \
                        _safe_float(info.get("currentPrice"))
                prior_close = prior_close or _safe_float(info.get("previousClose"))
                raw_time = info.get("regularMarketTime")
                if raw_time and as_of is None:
                    if isinstance(raw_time, (int, float)):
                        as_of = datetime.fromtimestamp(raw_time, tz=timezone.utc)
                    is_live = True
        except Exception as exc:
            logger.debug("info dict failed for %s: %s", ticker, exc)

    # --- Attempt 3: recent history, purely as a price fallback ---
    # Buy scores are NOT computed here. They come from the cross-sectional
    # model, which scores the whole universe at once (see models.predictor);
    # a per-stock score would have no meaning for a relative target.
    if price is None or prior_close is None or as_of is None:
        try:
            hist = t.history(period="5d", interval="1d", auto_adjust=True)
        except Exception:
            hist = None

        if hist is not None and not hist.empty:
            last_row = hist.iloc[-1]
            if price is None:
                price = _safe_float(last_row["Close"])
            if as_of is None:
                as_of = hist.index[-1].to_pydatetime()
            if prior_close is None:
                prior_close = (_safe_float(hist.iloc[-2]["Close"]) if len(hist) >= 2
                               else _safe_float(last_row.get("Open", last_row["Close"])))

    if price is None:
        return None

    if as_of is None:
        as_of = datetime.now(timezone.utc)
        is_live = False

    chg_pct = ((price / prior_close) - 1) * 100 if prior_close else 0.0

    return {
        "price": round(price, 2),
        "priorClose": round(prior_close, 2) if prior_close else None,
        "chg": f"{chg_pct:+.2f}%",
        "priceAsOf": _format_ist(as_of),
        "isLive": is_live,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }


def fetch_quotes_batch(tickers: list[str]) -> dict[str, dict]:
    """Fetch live/last prices for all tickers in batches."""
    quotes = {}
    batch_size = 25

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_str = " ".join(batch)
        try:
            yf_batch = yf.Tickers(batch_str)

            # yf.Tickers stores keys in different formats depending on version.
            # Build a case-insensitive lookup map.
            ticker_map = {}
            if hasattr(yf_batch, 'tickers') and yf_batch.tickers:
                for key, val in yf_batch.tickers.items():
                    ticker_map[key.upper()] = val

            for ticker in batch:
                try:
                    # Try exact key, then uppercase key
                    t_obj = ticker_map.get(ticker) or \
                            ticker_map.get(ticker.upper()) or \
                            ticker_map.get(ticker.replace(".", "").upper())
                    if t_obj is None:
                        # Fall back to individual Ticker
                        t_obj = yf.Ticker(ticker)
                    q = _quote_from_ticker_obj(ticker, t_obj)
                    if q:
                        quotes[ticker] = q
                except Exception as exc:
                    logger.debug("Quote failed for %s: %s", ticker, exc)
        except Exception as exc:
            logger.warning("Batch quote fetch failed: %s", exc)
            # Fallback: fetch individually
            for ticker in batch:
                try:
                    q = _quote_from_ticker_obj(ticker, yf.Ticker(ticker))
                    if q:
                        quotes[ticker] = q
                except Exception:
                    pass

    return quotes


def refresh_quotes():
    """Sync fetch and merge quotes into STATE."""
    if not STATE["universe"]:
        return
    tickers = [s["ticker"] for s in STATE["universe"]]
    quotes = fetch_quotes_batch(tickers)
    STATE["quotes"] = quotes
    STATE["last_updated"] = datetime.now(timezone.utc).isoformat()
    logger.info("Updated quotes for %d / %d tickers.", len(quotes), len(tickers))


def refresh_predictions():
    """Score the whole universe with the cross-sectional model."""
    try:
        from models.predictor import compute_universe_scores
    except Exception as exc:
        logger.warning("Predictor unavailable: %s", exc)
        return

    scores, meta = compute_universe_scores()
    if scores:
        STATE["predictions"] = scores
        STATE["prediction_meta"] = meta
        logger.info("Predictions ready for %d tickers (as of %s, %dd horizon).",
                    len(scores), meta.get("asOf"), meta.get("horizonDays") or 0)
    else:
        logger.warning("No predictions produced: %s", meta.get("error", "unknown"))


def _merge_quotes_into_stock(stock: dict) -> dict:
    out = stock.copy()
    q = STATE["quotes"].get(stock["ticker"])
    if q:
        out["price"] = q["price"]
        out["priorClose"] = q.get("priorClose", stock.get("priorClose"))
        out["chg"] = q["chg"]
        out["priceAsOf"] = q["priceAsOf"]
        out["isLive"] = q["isLive"]

    pred = STATE["predictions"].get(stock["ticker"])
    if pred:
        out["buyScore"] = pred["buyScore"]
        out["buyLabel"] = pred["buyLabel"]
        out["buyReason"] = pred["buyReason"]
        out["signals"] = pred["signals"]
        out["predictionAsOf"] = pred["asOf"]
        out["horizonDays"] = pred["horizonDays"]
    return out


async def fetch_live_prices_loop():
    """Background task to refresh Yahoo Finance quotes every 60s."""
    while True:
        try:
            refresh_quotes()
        except Exception as e:
            logger.error("Error fetching live prices: %s", e)
        await asyncio.sleep(60)


import webbrowser
import threading

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load pipeline results
    load_pipeline_results()

    # Score the universe once at startup. Feature construction needs the whole
    # panel, so this runs in a worker thread to keep startup non-blocking.
    asyncio.get_running_loop().run_in_executor(None, refresh_predictions)

    # Start background task for live price updates (don't block startup)
    task = asyncio.create_task(fetch_live_prices_loop())
    
    # Automatically open the default web browser after a short delay
    threading.Timer(2.0, lambda: webbrowser.open("http://127.0.0.1:8001")).start()
    
    logger.info("Server startup complete. Live price updates running in background.")
    yield
    task.cancel()


app = FastAPI(title="NSE Anomaly Terminal API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
def get_status():
    return {
        "status": "ok",
        "stocks": len(STATE["universe"]),
        "quotesLoaded": len(STATE["quotes"]),
        "anomalies": sum(1 for s in STATE["universe"] if s.get("isAnomaly")),
        "lastUpdated": STATE["last_updated"],
        "hasPipelineResults": (config.RESULTS_DIR / "anomaly_results.csv").exists(),
        "predictionsLoaded": len(STATE["predictions"]),
        "predictionMeta": STATE["prediction_meta"],
    }


@app.get("/api/universe")
def get_universe():
    """Return all tracked stocks with anomaly status and live/last prices."""
    universe_data = [_merge_quotes_into_stock(s) for s in STATE["universe"]]
    return {
        "status": "success",
        "data": universe_data,
        "lastUpdated": STATE["last_updated"],
        "count": len(universe_data),
    }


@app.get("/api/anomalies/{ticker}")
def get_anomaly_details(ticker: str):
    if ticker in STATE["anomalies"]:
        return {"status": "success", "data": STATE["anomalies"][ticker]}
    return {"status": "not_found", "message": "No anomaly details found."}


@app.get("/api/history/{ticker}")
def get_price_history(ticker: str, days: int = 90):
    """Return OHLCV history for charting — cached CSV first, then Yahoo."""
    days = min(max(days, 7), 365)
    csv_path = config.RAW_DIR / "daily" / f"{ticker}.csv"

    df = None
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            df = df.tail(days)
        except Exception:
            df = None

    if df is None or df.empty:
        try:
            raw = yf.download(
                ticker,
                period=f"{days}d",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if raw.empty:
                raise HTTPException(status_code=404, detail=f"No data for {ticker}")
            raw = raw.reset_index()
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
            df = raw.rename(columns={"Date": "Date", "Close": "Close", "Open": "Open",
                                     "High": "High", "Low": "Low", "Volume": "Volume"})
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    dates = []
    closes = []
    for _, row in df.iterrows():
        d = row.get("Date", row.name)
        if hasattr(d, "strftime"):
            dates.append(d.strftime("%Y-%m-%d"))
        else:
            dates.append(str(d)[:10])
        closes.append(round(float(row["Close"]), 2))

    return {
        "status": "success",
        "ticker": ticker,
        "dates": dates,
        "closes": closes,
        "count": len(dates),
    }


@app.get("/api/prediction/{ticker}")
def get_prediction_detail(ticker: str):
    """Score plus the ranked SHAP signals behind it."""
    pred = STATE["predictions"].get(ticker)
    if not pred:
        return {"status": "not_found",
                "message": f"No prediction for {ticker}."}
    return {"status": "success", "data": pred, "meta": STATE["prediction_meta"]}


@app.get("/api/model/performance")
def get_model_performance():
    """
    Out-of-sample validation metrics for the live model.

    Served straight from the walk-forward report so the UI can state measured
    performance rather than a remembered number.
    """
    report = config.RESULTS_DIR / "metrics" / "validation_report.json"
    if not report.exists():
        return {"status": "not_found",
                "message": "No validation report. Run: python -m training.train"}
    import json
    data = json.loads(report.read_text())
    return {"status": "success", "data": data, "meta": STATE["prediction_meta"]}


@app.post("/api/refresh")
def force_refresh():
    refresh_quotes()
    return {"status": "success", "quotesLoaded": len(STATE["quotes"])}


@app.post("/api/predictions/refresh")
def force_prediction_refresh():
    """Recompute universe scores (use after new daily data lands)."""
    refresh_predictions()
    return {"status": "success",
            "predictions": len(STATE["predictions"]),
            "meta": STATE["prediction_meta"]}


app.mount("/static", StaticFiles(directory="web", html=True), name="web")


@app.get("/")
def read_root():
    return FileResponse("web/index.html")


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8001, reload=False)
