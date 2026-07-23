"""
End-to-end Pipeline Orchestrator for NSE Anomaly Detection research.

Workflow:
1. Fetch or load daily OHLCV, intraday, and NSE bhavcopy data.
2. Preprocess, clean, and align data.
3. Engineer price, volume, pattern, and cross-sectional features.
4. Fit Isolation Forest, OC-SVM, Autoencoder, and LOF.
5. Compute cross-model agreement and consensus anomalies.
6. Evaluate against SEBI ground truth cases and F&O expiry pinning.
7. Generate visualization artifacts and summary tables.
"""

import sys
import logging
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config
from data.fetch_daily import fetch_all, load_cached
from data.fetch_bhavcopy import load_delivery_data
from data.preprocess import preprocess_all, build_combined_panel
from features.build_features import build_all, get_model_input, load_feature_panel
from models.ensemble import run_all_models, build_results_dataframe, save_results
from evaluation.sebi_validation import generate_sebi_validation_report, compute_hit_rate
from evaluation.expiry_validation import generate_expiry_validation_report
from evaluation.cross_model import generate_cross_model_report
from evaluation.metrics import (
    compute_anomaly_rates_by_ticker,
    compute_anomaly_rates_by_category,
    compute_feature_contribution,
)
from viz.plots import (
    plot_price_with_anomalies,
    plot_sebi_case_study,
    plot_expiry_comparison,
)

logger = logging.getLogger("main")


def run_pipeline(test_mode: bool = False, fetch_fresh: bool = False):
    """Run the complete benchmark pipeline."""
    logger.info("=" * 60)
    logger.info("Starting NSE Equities Anomaly Detection Benchmark")
    logger.info("=" * 60)

    # 1. Data Acquisition
    logger.info("Step 1: Data Acquisition")
    tickers = config.TEST_MODE_TICKERS if test_mode else None

    if fetch_fresh:
        daily_data = fetch_all(tickers=tickers)
    else:
        daily_data = load_cached(tickers=tickers)
        if not daily_data:
            logger.info("No cached data found. Fetching daily data...")
            daily_data = fetch_all(tickers=tickers)

    # Load delivery data if available
    try:
        delivery_df = load_delivery_data()
    except Exception as e:
        logger.warning(f"Failed to load delivery data: {e}")
        delivery_df = pd.DataFrame()

    # 2. Preprocessing
    logger.info("\nStep 2: Preprocessing")
    processed_data = preprocess_all(daily_data=daily_data, delivery_df=delivery_df, save=True)

    # 3. Feature Engineering
    logger.info("\nStep 3: Feature Engineering")
    panel, scaler = build_all(processed_data=processed_data, test_mode=test_mode, save=True)

    if panel.empty:
        logger.error("Feature building produced empty DataFrame. Aborting.")
        return

    # Extract inputs for models
    X, index, scaled_feature_names = get_model_input(panel)

    # 4. Model Benchmarking & Ensemble
    logger.info("\nStep 4: Model Benchmarking & Consensus Ensemble")
    model_results = run_all_models(X, feature_names=scaled_feature_names, test_mode=test_mode)

    results_df = build_results_dataframe(panel, model_results)
    save_results(results_df, model_results)

    # 5. Evaluation & Validation
    logger.info("\nStep 5: SEBI & Expiry Validation")
    sebi_report = generate_sebi_validation_report(results_df)
    hit_rates = compute_hit_rate(results_df)

    expiry_report = generate_expiry_validation_report(results_df)
    cross_model_report = generate_cross_model_report(results_df)

    # Metrics
    rates_by_ticker = compute_anomaly_rates_by_ticker(results_df)
    rates_by_category = compute_anomaly_rates_by_category(results_df)
    feature_contrib = compute_feature_contribution(results_df)

    # 6. Visualization Artifacts
    logger.info("\nStep 6: Generating Publication Figures")

    # Plot sample stocks
    plot_tickers = config.TEST_MODE_TICKERS if test_mode else ["RELIANCE.NS", "VISHALFAB.NS"]
    for t in plot_tickers:
        t_data = results_df[results_df["ticker"] == t]
        if not t_data.empty:
            safe_t = t.replace(".", "_")
            plot_price_with_anomalies(
                t_data, t,
                save_path=config.FIGURES_DIR / f"price_anomalies_{safe_t}.png"
            )

    # Plot SEBI case study if data present
    if not test_mode and "VISHALFAB.NS" in results_df["ticker"].values:
        plot_sebi_case_study(
            results_df, "VISHALFAB.NS",
            start_date="2017-06-01", end_date="2018-03-31",
            save_path=config.FIGURES_DIR / "sebi_case_study_vishalfab.png"
        )

    # Plot Expiry Comparison
    fno_mask = results_df.get("is_fno", pd.Series(False, index=results_df.index))
    if fno_mask.dtype != bool:
        fno_mask = fno_mask.astype(bool)
    if fno_mask.any():
        plot_expiry_comparison(
            results_df[fno_mask],
            save_path=config.FIGURES_DIR / "expiry_anomaly_scores_boxplot.png"
        )

    logger.info("=" * 60)
    logger.info("Pipeline Execution Complete!")
    logger.info(f"Results saved to: {config.RESULTS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="NSE Anomaly Detection Benchmark Pipeline")
    parser.add_argument("--test-mode", action="store_true", help="Run rapid smoke test")
    parser.add_argument("--fetch-fresh", action="store_true", help="Force fresh download from yfinance/NSE")
    args = parser.parse_args()

    run_pipeline(test_mode=args.test_mode, fetch_fresh=args.fetch_fresh)
