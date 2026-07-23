"""
General anomaly detection metrics and analysis utilities.

Computes:
- Per-model, per-stock anomaly rates
- Feature contribution analysis
- Temporal distribution of anomalies
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def compute_anomaly_rates_by_ticker(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute anomaly rate per model per stock.

    Returns a pivot table: rows=tickers, columns=models, values=anomaly rates.
    """
    model_cols = [c for c in results_df.columns if c.startswith("anomaly_")]

    if not model_cols or "ticker" not in results_df.columns:
        return pd.DataFrame()

    rates = results_df.groupby("ticker")[model_cols].mean()
    rates.columns = [c.replace("anomaly_", "") for c in rates.columns]

    # Add consensus
    if "consensus_anomaly" in results_df.columns:
        rates["consensus"] = results_df.groupby("ticker")["consensus_anomaly"].mean()

    # Sort by consensus anomaly rate
    sort_col = "consensus" if "consensus" in rates.columns else rates.columns[0]
    rates = rates.sort_values(sort_col, ascending=False)

    # Save
    rates_path = config.TABLES_DIR / "anomaly_rates_by_ticker.csv"
    rates.to_csv(rates_path)
    logger.info(f"Anomaly rates by ticker saved to {rates_path}")

    return rates


def compute_anomaly_rates_by_category(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare anomaly rates across stock categories
    (large-cap F&O, mid-cap, SEBI case stocks).
    """
    if "ticker" not in results_df.columns:
        return pd.DataFrame()

    def _categorize(ticker):
        if ticker in config.SEBI_CASE_STOCKS:
            return "SEBI Cases"
        elif ticker in config.LARGE_CAP_FNO:
            return "Large-cap F&O"
        elif ticker in config.MID_CAP:
            return "Mid-cap"
        else:
            return "Other"

    results_df = results_df.copy()
    results_df["category"] = results_df["ticker"].apply(_categorize)

    model_cols = [c for c in results_df.columns if c.startswith("anomaly_")]
    rates = results_df.groupby("category")[model_cols].mean()
    rates.columns = [c.replace("anomaly_", "") for c in rates.columns]

    if "consensus_anomaly" in results_df.columns:
        rates["consensus"] = results_df.groupby("category")["consensus_anomaly"].mean()

    logger.info(f"Anomaly rates by category:\n{rates.to_string()}")

    rates_path = config.TABLES_DIR / "anomaly_rates_by_category.csv"
    rates.to_csv(rates_path)

    return rates


def compute_feature_contribution(results_df: pd.DataFrame,
                                  feature_columns: list[str] = None) -> pd.DataFrame:
    """
    Analyze which features are most different on anomalous vs. normal days.

    For each feature, compute the mean value on anomaly days vs. normal days.
    Large differences indicate the features driving the anomaly flag.
    """
    if "consensus_anomaly" not in results_df.columns:
        logger.warning("No consensus_anomaly column")
        return pd.DataFrame()

    if feature_columns is None:
        # Use unscaled feature columns
        from features.build_features import FEATURE_COLUMNS
        feature_columns = [c for c in FEATURE_COLUMNS if c in results_df.columns]

    anomalous = results_df[results_df["consensus_anomaly"] == 1]
    normal = results_df[results_df["consensus_anomaly"] == 0]

    if len(anomalous) == 0 or len(normal) == 0:
        return pd.DataFrame()

    rows = []
    for col in feature_columns:
        if col not in results_df.columns:
            continue

        a_mean = anomalous[col].mean()
        n_mean = normal[col].mean()
        a_std = anomalous[col].std()
        n_std = normal[col].std()

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((a_std**2 + n_std**2) / 2)
        cohens_d = (a_mean - n_mean) / pooled_std if pooled_std > 0 else 0

        rows.append({
            "feature": col,
            "mean_anomalous": a_mean,
            "mean_normal": n_mean,
            "diff": a_mean - n_mean,
            "cohens_d": cohens_d,
            "abs_cohens_d": abs(cohens_d),
        })

    contribution = pd.DataFrame(rows).sort_values("abs_cohens_d", ascending=False)

    path = config.TABLES_DIR / "feature_contribution.csv"
    contribution.to_csv(path, index=False)
    logger.info(f"Feature contribution analysis saved to {path}")

    # Log top features
    logger.info("Top 10 features driving anomaly flags:")
    for _, row in contribution.head(10).iterrows():
        direction = "↑" if row["diff"] > 0 else "↓"
        logger.info(
            f"  {row['feature']}: d={row['cohens_d']:.2f} {direction}"
        )

    return contribution


def compute_monthly_anomaly_distribution(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Distribution of anomalies by month — are they clustered seasonally?

    F&O expiry effects, quarterly results seasons, and budget/policy
    events may create seasonal patterns.
    """
    if "consensus_anomaly" not in results_df.columns:
        return pd.DataFrame()

    results_df = results_df.copy()
    if "month" not in results_df.columns:
        results_df["month"] = results_df.index.month

    monthly = results_df.groupby("month")["consensus_anomaly"].agg(
        ["mean", "sum", "count"]
    )
    monthly.columns = ["anomaly_rate", "n_anomalies", "n_total_days"]

    path = config.TABLES_DIR / "monthly_anomaly_distribution.csv"
    monthly.to_csv(path)

    return monthly
