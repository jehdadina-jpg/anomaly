"""
Central configuration for the NSE Anomaly Detection project.
Contains stock universe, SEBI ground-truth cases, F&O expiry dates,
circuit limit definitions, model hyperparameters, and path constants.
"""

import os
from datetime import date
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"

# Create directories if they don't exist
for d in [RAW_DIR / "daily", RAW_DIR / "intraday", RAW_DIR / "bhavcopy",
          RAW_DIR / "fo", PROCESSED_DIR, FEATURES_DIR, FIGURES_DIR,
          TABLES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATE RANGE
# ============================================================================

# Primary analysis window (daily data)
START_DATE = "2021-01-01"
END_DATE = "2026-07-23"

# Intraday data (limited by yfinance)
INTRADAY_5M_LOOKBACK_DAYS = 59   # yfinance limit ~60 days
INTRADAY_1M_LOOKBACK_DAYS = 7    # yfinance limit ~7 days

# ============================================================================
# STOCK UNIVERSE
# ============================================================================

# Sector definitions for cross-sectional analysis
SECTORS = {
    "Nifty 50": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS", "SBIN.NS", 
        "BHARTIARTL.NS", "LT.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "AXISBANK.NS", "HCLTECH.NS", 
        "ADANIENT.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "TATAMOTORS.NS", "ULTRACEMCO.NS", 
        "TATASTEEL.NS", "POWERGRID.NS", "NTPC.NS", "BAJAJFINSV.NS", "INDUSINDBK.NS", "TITAN.NS", 
        "NESTLEIND.NS", "ONGC.NS", "JSWSTEEL.NS", "TECHM.NS", "HINDUNILVR.NS", "M&M.NS", "GRASIM.NS", 
        "WIPRO.NS", "HINDALCO.NS", "COALINDIA.NS", "CIPLA.NS", "SBILIFE.NS", "DRREDDY.NS", "BRITANNIA.NS", 
        "BAJAJ-AUTO.NS", "APOLLOHOSP.NS", "TATACONSUM.NS", "EICHERMOT.NS", "DIVISLAB.NS", "HDFCLIFE.NS", 
        "BPCL.NS", "HEROMOTOCO.NS", "LTIM.NS", "UPL.NS", "ADANIPORTS.NS"
    ]
}

NIFTY_50 = SECTORS["Nifty 50"]

# Full ticker list
ALL_TICKERS = list(set(NIFTY_50))

# Index tickers for beta calculation
INDEX_TICKERS = ["^NSEI", "^NSEBANK"]  # Nifty 50, Bank Nifty

# F&O eligible tickers (large-cap + select mid-cap with F&O contracts)
FNO_ELIGIBLE = set(NIFTY_50)

# Ticker → Sector lookup (built from SECTORS dict)
TICKER_TO_SECTOR = {}
for sector, tickers in SECTORS.items():
    for t in tickers:
        TICKER_TO_SECTOR[t] = sector

# ============================================================================
# SEBI GROUND TRUTH CASES
# ============================================================================
# Format: ticker -> list of (start_date, end_date, description)
# These are the documented manipulation periods from SEBI orders.
# Used in evaluation/sebi_validation.py to measure model hit rate.

SEBI_CASES = {
    # Pump-and-dump scheme, SEBI interim order June 19, 2023
    # Manipulation period: 2017-2020 (coordinated trades + SMS campaigns)
    "VISHALFAB.NS": [
        (date(2017, 6, 1), date(2018, 3, 31),
         "Pump-and-dump scheme; SEBI interim order June 2023; "
         "coordinated trades and bulk SMS campaigns"),
        (date(2019, 1, 1), date(2020, 6, 30),
         "Second wave of manipulation in Vishal Fabrics"),
    ],

    # SEBI interim order June 3, 2026
    "RAJESHEXPS.NS": [
        (date(2025, 9, 1), date(2026, 5, 31),
         "SEBI interim order June 2026; suspected manipulation"),
    ],
}

# ============================================================================
# F&O EXPIRY DATES (2021-2026)
# ============================================================================
# Pre-Sept 2025: Last Thursday of month
# Post-Sept 2025: Last Tuesday of month
# Holiday adjustments included for known holidays

# Auto-generate monthly expiry dates
def _generate_expiry_dates():
    """Generate F&O monthly expiry dates from 2021 to 2026."""
    import calendar
    from datetime import timedelta

    expiry_dates = []

    for year in range(2021, 2027):
        for month in range(1, 13):
            if year == 2026 and month > 7:
                break

            # Determine the target weekday
            # Pre-Sept 2025: Thursday (3), Post-Sept 2025: Tuesday (1)
            if year < 2025 or (year == 2025 and month < 9):
                target_weekday = 3  # Thursday
            else:
                target_weekday = 1  # Tuesday

            # Find last occurrence of target weekday in the month
            last_day = calendar.monthrange(year, month)[1]
            d = date(year, month, last_day)

            while d.weekday() != target_weekday:
                d -= timedelta(days=1)

            expiry_dates.append(d)

    return expiry_dates


FNO_EXPIRY_DATES = _generate_expiry_dates()

# Convert to set for O(1) lookup
FNO_EXPIRY_DATE_SET = set(FNO_EXPIRY_DATES)

# ============================================================================
# CIRCUIT LIMIT BANDS
# ============================================================================
# Most non-F&O stocks: 2%, 5%, 10%, or 20% daily price bands
# F&O stocks: no fixed circuit (dynamic flexing around 10%)
# We use these to compute circuit_proximity features

CIRCUIT_BANDS = {
    "default": 0.20,          # 20% for most stocks
    "small_cap": 0.05,        # 5% for small-cap
    "fno": 0.10,              # 10% dynamic operating range for F&O
}

# Override per ticker if needed
TICKER_CIRCUIT_BAND = {}
for t in NIFTY_50:
    TICKER_CIRCUIT_BAND[t] = CIRCUIT_BANDS["fno"]

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

MODEL_PARAMS = {
    "isolation_forest": {
        "contamination_values": [0.01, 0.02, 0.05, 0.10],
        "n_estimators": 200,
        "max_samples": "auto",
        "random_state": 42,
    },
    "ocsvm": {
        "nu_values": [0.01, 0.05, 0.10],
        "kernel": "rbf",
        "gamma": "scale",
        "max_train_samples": 10000,  # subsample for scalability
    },
    "autoencoder": {
        "hidden_dims": [64, 32, 16],  # encoder: input->64->32->16->32->64->input
        "learning_rate": 1e-3,
        "batch_size": 256,
        "epochs": 100,
        "patience": 10,           # early stopping patience
        "threshold_sigmas": [2, 3],  # mean + Nσ for anomaly threshold
        "train_fraction": 0.9,    # fraction of "normal" data for training
        "random_state": 42,
    },
    "lof": {
        "n_neighbors_values": [10, 20, 50],
        "contamination": 0.05,
        "metric": "euclidean",
    },
}

# Ensemble: minimum number of models that must agree for consensus anomaly
ENSEMBLE_MIN_AGREEMENT = 2  # out of 4 models

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================

FEATURE_PARAMS = {
    "rolling_vol_windows": [5, 20],
    "volume_zscore_window": 20,
    "volume_spike_threshold": 3.0,     # volume > 3x rolling mean
    "delivery_pct_low_threshold": 0.30, # 30th percentile
    "beta_window": 60,                  # rolling OLS window for beta
    "autocorrelation_windows": [5, 20],
    "warmup_period": 30,               # days to drop at start (NaN from rolling)
}

# ============================================================================
# DATA FETCH PARAMETERS
# ============================================================================

FETCH_PARAMS = {
    "yfinance_sleep": 1.0,          # seconds between yfinance requests
    "max_gap_days": 5,              # max consecutive missing trading days
    "bhavcopy_user_agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "bhavcopy_sleep": 0.5,         # seconds between bhavcopy downloads
}

# ============================================================================
# TEST MODE (for smoke testing)
# ============================================================================

TEST_MODE_TICKERS = ["RELIANCE.NS", "SBIN.NS", "TCS.NS", "INFY.NS", "ITC.NS"]
TEST_MODE_DAYS = 90

