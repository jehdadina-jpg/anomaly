"""
Cross-model agreement analysis.

Reports how often the 4 models agree on anomalies.
High agreement + match with a known SEBI case is the strongest
evidence that the approach works.
"""

import sys
import logging
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def compute_pairwise_agreement(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Jaccard similarity between each pair of models' anomaly sets.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    Higher = more agreement between two models.
    """
    model_names = [c.replace("anomaly_", "") for c in results_df.columns
                   if c.startswith("anomaly_")]

    jaccard_matrix = pd.DataFrame(
        np.zeros((len(model_names), len(model_names))),
        index=model_names,
        columns=model_names,
    )

    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i == j:
                jaccard_matrix.loc[m1, m2] = 1.0
                continue

            a = results_df[f"anomaly_{m1}"].astype(bool)
            b = results_df[f"anomaly_{m2}"].astype(bool)

            intersection = (a & b).sum()
            union = (a | b).sum()

            jaccard = intersection / union if union > 0 else 0
            jaccard_matrix.loc[m1, m2] = jaccard

    logger.info("Pairwise Jaccard similarity:")
    logger.info(f"\n{jaccard_matrix.to_string()}")

    return jaccard_matrix


def compute_score_correlations(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman rank correlation of anomaly scores across models.

    Spearman is robust to the different scale/distribution of scores
    across models (IF scores vs. LOF scores vs. reconstruction error).
    """
    score_cols = [c for c in results_df.columns if c.startswith("score_")]

    if len(score_cols) < 2:
        return pd.DataFrame()

    scores = results_df[score_cols].dropna()
    model_names = [c.replace("score_", "") for c in score_cols]

    corr_matrix = pd.DataFrame(
        np.zeros((len(model_names), len(model_names))),
        index=model_names,
        columns=model_names,
    )

    for i, (n1, c1) in enumerate(zip(model_names, score_cols)):
        for j, (n2, c2) in enumerate(zip(model_names, score_cols)):
            if i == j:
                corr_matrix.loc[n1, n2] = 1.0
                continue
            rho, _ = stats.spearmanr(scores[c1], scores[c2])
            corr_matrix.loc[n1, n2] = rho

    logger.info("Spearman score correlations:")
    logger.info(f"\n{corr_matrix.to_string()}")

    return corr_matrix


def analyze_high_confidence_anomalies(results_df: pd.DataFrame,
                                       min_agreement: int = 3) -> pd.DataFrame:
    """
    Analyze anomalies flagged by multiple models (high confidence).

    These are the most interesting findings — stock-days that multiple
    independent methods independently identify as anomalous.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must have 'agreement_count' column
    min_agreement : int
        Minimum number of models that must agree (default 3/4)

    Returns
    -------
    pd.DataFrame
        High-confidence anomalies with feature values for investigation
    """
    if "agreement_count" not in results_df.columns:
        logger.warning("No agreement_count column — run ensemble first")
        return pd.DataFrame()

    high_conf = results_df[results_df["agreement_count"] >= min_agreement].copy()

    logger.info(
        f"High-confidence anomalies (≥{min_agreement}/4 models): "
        f"{len(high_conf)} stock-days"
    )

    if len(high_conf) == 0:
        return pd.DataFrame()

    # Summarize by ticker
    by_ticker = high_conf.groupby("ticker").agg(
        n_anomaly_days=("agreement_count", "count"),
        mean_agreement=("agreement_count", "mean"),
        date_range_start=("ticker", lambda x: x.index.min()),
        date_range_end=("ticker", lambda x: x.index.max()),
    ).sort_values("n_anomaly_days", ascending=False)

    logger.info(f"\nHigh-confidence anomalies by ticker:")
    logger.info(f"\n{by_ticker.to_string()}")

    # Check overlap with SEBI ground truth
    if "sebi_ground_truth" in high_conf.columns:
        n_gt = high_conf["sebi_ground_truth"].sum()
        logger.info(
            f"\nOf {len(high_conf)} high-confidence anomalies, "
            f"{n_gt} ({n_gt/len(high_conf):.1%}) are in SEBI ground truth"
        )

    return high_conf


def compute_temporal_clustering(results_df: pd.DataFrame) -> dict:
    """
    Analyze whether anomalies cluster in time or are uniformly distributed.

    Temporal clustering (bursts of anomalies) supports the manipulation
    hypothesis — manipulation events are discrete episodes, not random.
    Uniform distribution would suggest the model is just capturing noise.
    """
    model_names = [c.replace("anomaly_", "") for c in results_df.columns
                   if c.startswith("anomaly_")]

    clustering_results = {}
    for name in model_names:
        col = f"anomaly_{name}"
        if col not in results_df.columns:
            continue

        anomaly_days = results_df[results_df[col] == 1]
        if len(anomaly_days) < 10:
            continue

        # Per-ticker temporal clustering
        ticker_clustering = []
        for ticker in anomaly_days["ticker"].unique():
            ticker_anomalies = anomaly_days[anomaly_days["ticker"] == ticker]
            if len(ticker_anomalies) < 3:
                continue

            # Compute gaps between consecutive anomaly days
            gaps = ticker_anomalies.index.to_series().diff().dt.days.dropna()

            if len(gaps) > 0:
                ticker_clustering.append({
                    "ticker": ticker,
                    "n_anomalies": len(ticker_anomalies),
                    "mean_gap_days": gaps.mean(),
                    "median_gap_days": gaps.median(),
                    "min_gap_days": gaps.min(),
                    "max_gap_days": gaps.max(),
                    "std_gap_days": gaps.std(),
                    # If gaps are small and clustered, CoV will be high
                    "gap_cov": gaps.std() / gaps.mean() if gaps.mean() > 0 else 0,
                })

        if ticker_clustering:
            tc_df = pd.DataFrame(ticker_clustering)
            clustering_results[name] = {
                "per_ticker": tc_df,
                "mean_gap_across_tickers": tc_df["mean_gap_days"].mean(),
                "mean_cluster_cov": tc_df["gap_cov"].mean(),
            }

    return clustering_results


def generate_cross_model_report(results_df: pd.DataFrame) -> dict:
    """
    Generate comprehensive cross-model agreement report.

    Returns dict of DataFrames suitable for paper tables.
    """
    report = {}

    # 1. Pairwise Jaccard similarity
    report["jaccard_similarity"] = compute_pairwise_agreement(results_df)

    # 2. Score correlations
    report["score_correlations"] = compute_score_correlations(results_df)

    # 3. High-confidence anomalies
    report["high_confidence_3of4"] = analyze_high_confidence_anomalies(
        results_df, min_agreement=3
    )
    report["high_confidence_4of4"] = analyze_high_confidence_anomalies(
        results_df, min_agreement=4
    )

    # 4. Temporal clustering
    report["temporal_clustering"] = compute_temporal_clustering(results_df)

    # Save key tables
    for name, data in report.items():
        if isinstance(data, pd.DataFrame) and not data.empty:
            path = config.TABLES_DIR / f"cross_model_{name}.csv"
            data.to_csv(path)
            logger.info(f"Saved {name} to {path}")

    return report
