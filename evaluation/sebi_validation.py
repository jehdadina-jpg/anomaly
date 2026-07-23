"""
SEBI Case Validation — Known-event ground truth evaluation.

Since no labeled fraud data exists publicly, we validate against
documented SEBI manipulation orders (the regulator publishes these
with specific entity names and time periods).

This is the part that makes this a real paper: we check if our
unsupervised models flag the exact stock-days that SEBI later
confirmed as manipulated.
"""

import sys
import logging
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def identify_ground_truth_days(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark stock-days that fall within known SEBI manipulation periods.

    Parameters
    ----------
    results_df : pd.DataFrame
        Anomaly results panel with 'ticker' column and DatetimeIndex

    Returns
    -------
    pd.DataFrame
        Results with 'sebi_ground_truth' column added (1 = known manipulation)
    """
    out = results_df.copy()
    out["sebi_ground_truth"] = 0
    out["sebi_case_description"] = ""

    for ticker, cases in config.SEBI_CASES.items():
        for start_date, end_date, description in cases:
            mask = (
                (out["ticker"] == ticker) &
                (out.index >= pd.Timestamp(start_date)) &
                (out.index <= pd.Timestamp(end_date))
            )
            out.loc[mask, "sebi_ground_truth"] = 1
            out.loc[mask, "sebi_case_description"] = description

    n_gt = out["sebi_ground_truth"].sum()
    n_tickers = out.loc[out["sebi_ground_truth"] == 1, "ticker"].nunique()
    logger.info(
        f"Ground truth: {n_gt} stock-days across {n_tickers} tickers "
        f"marked as known manipulation"
    )

    return out


def compute_hit_rate(results_df: pd.DataFrame,
                     model_name: str = None,
                     window_days: int = 0) -> dict:
    """
    Compute hit rate: what fraction of known manipulation days
    were flagged as anomalous by each model?

    Parameters
    ----------
    results_df : pd.DataFrame
        Must have 'sebi_ground_truth' and 'anomaly_{model}' columns
    model_name : str
        Specific model to evaluate. If None, evaluates all models.
    window_days : int
        Tolerance window: count a hit if the model flags any day
        within ±window_days of a ground truth day. 0 = exact match.

    Returns
    -------
    dict of model_name -> hit rate metrics
    """
    if "sebi_ground_truth" not in results_df.columns:
        results_df = identify_ground_truth_days(results_df)

    gt_mask = results_df["sebi_ground_truth"] == 1
    gt_days = results_df[gt_mask]

    if len(gt_days) == 0:
        logger.warning("No ground truth days found in results")
        return {}

    model_names = ([model_name] if model_name
                   else [c.replace("anomaly_", "") for c in results_df.columns
                         if c.startswith("anomaly_")])

    hit_rates = {}
    for name in model_names:
        col = f"anomaly_{name}"
        if col not in results_df.columns:
            continue

        if window_days == 0:
            # Exact match
            hits = gt_days[col].sum()
            total = len(gt_days)
            hit_rate = hits / total if total > 0 else 0
        else:
            # Window-based matching
            hits = 0
            total = len(gt_days)
            for idx in gt_days.index:
                # Check if any day within ±window is flagged
                window_start = idx - pd.Timedelta(days=window_days)
                window_end = idx + pd.Timedelta(days=window_days)
                window_mask = (
                    (results_df.index >= window_start) &
                    (results_df.index <= window_end) &
                    (results_df["ticker"] == gt_days.loc[idx, "ticker"])
                )
                if results_df.loc[window_mask, col].any():
                    hits += 1
            hit_rate = hits / total if total > 0 else 0

        hit_rates[name] = {
            "hit_rate": hit_rate,
            "hits": int(hits),
            "total_gt_days": int(total),
            "window_days": window_days,
        }
        logger.info(
            f"  {name}: hit rate = {hit_rate:.2%} "
            f"({hits}/{total} ground truth days flagged, "
            f"window=±{window_days}d)"
        )

    return hit_rates


def compute_precision_at_ground_truth(results_df: pd.DataFrame) -> dict:
    """
    Among the days flagged as anomalous, how many are in ground truth periods?

    This is the "flip side" of hit rate — measures false positive rate
    near known manipulation events.
    """
    if "sebi_ground_truth" not in results_df.columns:
        results_df = identify_ground_truth_days(results_df)

    model_names = [c.replace("anomaly_", "") for c in results_df.columns
                   if c.startswith("anomaly_")]

    precision_results = {}
    for name in model_names:
        col = f"anomaly_{name}"
        flagged = results_df[results_df[col] == 1]

        if len(flagged) == 0:
            precision_results[name] = {"precision": 0, "flagged": 0, "true_positives": 0}
            continue

        # Among flagged days for SEBI case tickers only
        sebi_tickers = set(config.SEBI_CASES.keys())
        flagged_sebi = flagged[flagged["ticker"].isin(sebi_tickers)]

        if len(flagged_sebi) == 0:
            precision_results[name] = {
                "precision": 0,
                "flagged_sebi_ticker_days": 0,
                "true_positives": 0,
                "total_flagged": len(flagged),
            }
            continue

        true_positives = flagged_sebi["sebi_ground_truth"].sum()
        precision = true_positives / len(flagged_sebi) if len(flagged_sebi) > 0 else 0

        precision_results[name] = {
            "precision": precision,
            "flagged_sebi_ticker_days": len(flagged_sebi),
            "true_positives": int(true_positives),
            "total_flagged": len(flagged),
        }
        logger.info(
            f"  {name}: precision on SEBI tickers = {precision:.2%} "
            f"({true_positives}/{len(flagged_sebi)})"
        )

    return precision_results


def generate_sebi_validation_report(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a comprehensive SEBI validation report.

    Returns a DataFrame with per-case, per-model results suitable
    for inclusion in a paper.
    """
    if "sebi_ground_truth" not in results_df.columns:
        results_df = identify_ground_truth_days(results_df)

    report_rows = []
    model_names = [c.replace("anomaly_", "") for c in results_df.columns
                   if c.startswith("anomaly_")]

    for ticker, cases in config.SEBI_CASES.items():
        for start_date, end_date, description in cases:
            case_mask = (
                (results_df["ticker"] == ticker) &
                (results_df.index >= pd.Timestamp(start_date)) &
                (results_df.index <= pd.Timestamp(end_date))
            )
            case_data = results_df[case_mask]

            if len(case_data) == 0:
                continue

            row = {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "description": description[:60],
                "n_trading_days": len(case_data),
            }

            for name in model_names:
                col = f"anomaly_{name}"
                if col in case_data.columns:
                    n_flagged = case_data[col].sum()
                    row[f"flagged_{name}"] = int(n_flagged)
                    row[f"hit_rate_{name}"] = n_flagged / len(case_data)

            # Consensus
            if "consensus_anomaly" in case_data.columns:
                n_consensus = case_data["consensus_anomaly"].sum()
                row["flagged_consensus"] = int(n_consensus)
                row["hit_rate_consensus"] = n_consensus / len(case_data)

            report_rows.append(row)

    report = pd.DataFrame(report_rows)

    # Save report
    report_path = config.TABLES_DIR / "sebi_validation_report.csv"
    report.to_csv(report_path, index=False)
    logger.info(f"SEBI validation report saved to {report_path}")

    return report
