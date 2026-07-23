"""
F&O Expiry-Day Validation.

Tests whether unsupervised models systematically flag F&O expiry days
more often than non-expiry days for stocks with high derivative interest.

This validates against a known, real phenomenon ("pinning") and provides
a second independent evaluation axis beyond SEBI case validation.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def compute_expiry_anomaly_rates(results_df: pd.DataFrame) -> dict:
    """
    Compare anomaly rates on expiry days vs. non-expiry days
    for F&O-eligible stocks.

    Parameters
    ----------
    results_df : pd.DataFrame
        Anomaly results with 'is_fno', 'is_expiry_day', and
        'anomaly_*' columns

    Returns
    -------
    dict of model_name -> {expiry_rate, non_expiry_rate, ratio, p_value}
    """
    # Filter to F&O eligible stocks
    fno_mask = results_df.get("is_fno", pd.Series(False, index=results_df.index))
    if fno_mask.dtype != bool:
        fno_mask = fno_mask.astype(bool)
    fno_data = results_df[fno_mask]

    if len(fno_data) == 0:
        logger.warning("No F&O data available for expiry analysis")
        return {}

    # Identify expiry days
    if "is_expiry_day" not in fno_data.columns:
        # Compute from config
        expiry_set = config.FNO_EXPIRY_DATE_SET
        fno_data = fno_data.copy()
        fno_data["is_expiry_day"] = fno_data.index.map(
            lambda dt: 1 if dt.date() in expiry_set else 0
        )

    expiry_days = fno_data[fno_data["is_expiry_day"] == 1]
    non_expiry_days = fno_data[fno_data["is_expiry_day"] == 0]

    logger.info(
        f"F&O stocks: {fno_data['ticker'].nunique()} tickers, "
        f"{len(expiry_days)} expiry days, {len(non_expiry_days)} non-expiry days"
    )

    model_names = [c.replace("anomaly_", "") for c in results_df.columns
                   if c.startswith("anomaly_")]

    results = {}
    for name in model_names:
        col = f"anomaly_{name}"
        if col not in fno_data.columns:
            continue

        expiry_rate = expiry_days[col].mean() if len(expiry_days) > 0 else 0
        non_expiry_rate = non_expiry_days[col].mean() if len(non_expiry_days) > 0 else 0

        # Two-proportion z-test
        n1 = len(expiry_days)
        n2 = len(non_expiry_days)
        p1 = expiry_rate
        p2 = non_expiry_rate

        if n1 > 0 and n2 > 0 and (p1 + p2) > 0:
            p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
            if se > 0:
                z_stat = (p1 - p2) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                z_stat = 0
                p_value = 1.0
        else:
            z_stat = 0
            p_value = 1.0

        ratio = expiry_rate / non_expiry_rate if non_expiry_rate > 0 else float("inf")

        results[name] = {
            "expiry_anomaly_rate": expiry_rate,
            "non_expiry_anomaly_rate": non_expiry_rate,
            "rate_ratio": ratio,
            "z_statistic": z_stat,
            "p_value": p_value,
            "significant_0.05": p_value < 0.05,
            "significant_0.01": p_value < 0.01,
            "n_expiry_days": n1,
            "n_non_expiry_days": n2,
        }

        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        logger.info(
            f"  {name}: expiry={expiry_rate:.3%}, non-expiry={non_expiry_rate:.3%}, "
            f"ratio={ratio:.2f}, p={p_value:.4f} {significance}"
        )

    return results


def compute_expiry_week_analysis(results_df: pd.DataFrame) -> dict:
    """
    Extended analysis: compare anomaly rates in the expiry week
    (last 5 trading days before expiry) vs. other days.

    The pinning effect may start days before expiry as delta hedging
    intensifies, so the expiry-week window captures more of the signal.
    """
    fno_mask = results_df.get("is_fno", pd.Series(False, index=results_df.index))
    if fno_mask.dtype != bool:
        fno_mask = fno_mask.astype(bool)
    fno_data = results_df[fno_mask].copy()

    if len(fno_data) == 0:
        return {}

    if "is_expiry_week" not in fno_data.columns:
        if "days_to_expiry" in fno_data.columns:
            fno_data["is_expiry_week"] = (fno_data["days_to_expiry"] <= 5).astype(int)
        else:
            return {}

    expiry_week = fno_data[fno_data["is_expiry_week"] == 1]
    non_expiry_week = fno_data[fno_data["is_expiry_week"] == 0]

    model_names = [c.replace("anomaly_", "") for c in results_df.columns
                   if c.startswith("anomaly_")]

    results = {}
    for name in model_names:
        col = f"anomaly_{name}"
        if col not in fno_data.columns:
            continue

        ew_rate = expiry_week[col].mean() if len(expiry_week) > 0 else 0
        new_rate = non_expiry_week[col].mean() if len(non_expiry_week) > 0 else 0

        # Mann-Whitney U test on anomaly scores
        score_col = f"score_{name}"
        if score_col in fno_data.columns:
            try:
                u_stat, u_pvalue = stats.mannwhitneyu(
                    expiry_week[score_col].dropna(),
                    non_expiry_week[score_col].dropna(),
                    alternative="less",  # expiry scores more anomalous
                )
            except ValueError:
                u_stat, u_pvalue = 0, 1.0
        else:
            u_stat, u_pvalue = 0, 1.0

        results[name] = {
            "expiry_week_rate": ew_rate,
            "non_expiry_week_rate": new_rate,
            "rate_ratio": ew_rate / new_rate if new_rate > 0 else float("inf"),
            "mann_whitney_u": u_stat,
            "mann_whitney_p": u_pvalue,
        }

    return results


def generate_expiry_validation_report(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a publication-quality expiry validation report.

    Returns a DataFrame summarizing expiry-day and expiry-week
    anomaly rate comparisons for all models.
    """
    day_results = compute_expiry_anomaly_rates(results_df)
    week_results = compute_expiry_week_analysis(results_df)

    rows = []
    for model_name in day_results:
        row = {
            "model": model_name,
            **{f"day_{k}": v for k, v in day_results[model_name].items()},
        }
        if model_name in week_results:
            row.update({f"week_{k}": v for k, v in week_results[model_name].items()})
        rows.append(row)

    report = pd.DataFrame(rows)

    # Save
    report_path = config.TABLES_DIR / "expiry_validation_report.csv"
    report.to_csv(report_path, index=False)
    logger.info(f"Expiry validation report saved to {report_path}")

    return report
