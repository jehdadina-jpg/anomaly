"""
Publication-quality plots for NSE Anomaly Detection research.

Includes:
1. Price chart with anomaly overlay (candlestick/line + anomaly markers)
2. Feature correlation heatmap
3. Model anomaly rate comparison by category
4. Detailed SEBI case study plot (price, volume, delivery %, anomaly flags)
5. Expiry-day anomaly score distribution (boxplots)
6. Cross-model Venn / Overlap diagram
7. Circuit proximity & delivery % distribution plots
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_price_with_anomalies(df: pd.DataFrame, ticker: str,
                              save_path: Path = None) -> plt.Figure:
    """
    Plot stock price history with anomaly markers overlay.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1, 1]})

    # Price
    ax1.plot(df.index, df["Close"], label="Close Price", color="#1f77b4", lw=1.2)

    # Highlight Consensus Anomalies
    if "consensus_anomaly" in df.columns:
        anom_mask = df["consensus_anomaly"] == 1
        ax1.scatter(df.index[anom_mask], df.loc[anom_mask, "Close"],
                    color="#FF1744", label="Consensus Anomaly", s=35, zorder=5, alpha=0.85)

    ax1.set_title(f"{ticker} — Price & Detected Anomalies", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Price (INR)", fontsize=11)
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Volume
    colors = np.where(df["Close"] >= df["Open"], "#00E676", "#FF1744")
    ax2.bar(df.index, df["Volume"], color=colors, alpha=0.6, width=1.0)
    ax2.set_ylabel("Volume", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Delivery % if present
    if "delivery_pct" in df.columns and df["delivery_pct"].notna().any():
        ax3.plot(df.index, df["delivery_pct"] * 100, color="#FF9100", label="Delivery %", lw=1.2)
        ax3.set_ylabel("Delivery %", fontsize=11)
        ax3.set_ylim(0, 100)
        ax3.axhline(30, color="gray", linestyle=":", alpha=0.7, label="Low Delivery Threshold (30%)")
        ax3.legend(loc="upper left")
        ax3.grid(True, linestyle="--", alpha=0.5)
    else:
        ax3.text(0.5, 0.5, "No Delivery % Data", ha="center", va="center", transform=ax3.transAxes)

    ax3.set_xlabel("Date", fontsize=11)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")

    return fig


def plot_sebi_case_study(results_df: pd.DataFrame, ticker: str,
                         start_date: str, end_date: str,
                         save_path: Path = None) -> plt.Figure:
    """
    Detailed plot of a known SEBI manipulation period for paper case study.
    """
    mask = (results_df["ticker"] == ticker) & (results_df.index >= pd.Timestamp(start_date)) & (results_df.index <= pd.Timestamp(end_date))
    df = results_df[mask]

    if len(df) == 0:
        logger.warning(f"No data for {ticker} between {start_date} and {end_date}")
        return None

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1, 1]})

    # Price & SEBI flag
    axes[0].plot(df.index, df["Close"], color="#29b6f6", label="Close Price", lw=1.5)

    if "sebi_ground_truth" in df.columns:
        gt = df["sebi_ground_truth"] == 1
        axes[0].fill_between(df.index, df["Close"].min(), df["Close"].max(),
                             where=gt, color="#ffcdd2", alpha=0.4, label="SEBI Manipulation Period")

    if "consensus_anomaly" in df.columns:
        anom = df["consensus_anomaly"] == 1
        axes[0].scatter(df.index[anom], df.loc[anom, "Close"],
                        color="#d50000", s=45, label="Model Consensus Anomaly", zorder=5)

    axes[0].set_title(f"SEBI Case Study: {ticker} ({start_date} to {end_date})", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Price (INR)")
    axes[0].legend(loc="upper left")

    # Volume & z-score
    axes[1].bar(df.index, df["Volume"], color="#78909c", alpha=0.7)
    if "volume_zscore_20" in df.columns:
        ax1_twin = axes[1].twinx()
        ax1_twin.plot(df.index, df["volume_zscore_20"], color="#ab47bc", lw=1, label="Volume Z-Score")
        ax1_twin.axhline(3.0, color="#8e24aa", linestyle="--", alpha=0.7)
        ax1_twin.set_ylabel("Z-Score")

    axes[1].set_ylabel("Volume")

    # Delivery %
    if "delivery_pct" in df.columns and df["delivery_pct"].notna().any():
        axes[2].plot(df.index, df["delivery_pct"] * 100, color="#ff9800", lw=1.5)
        axes[2].axhline(30, color="#e65100", linestyle=":", alpha=0.8, label="30% Low Delivery Limit")
        axes[2].set_ylabel("Delivery %")
        axes[2].set_ylim(0, 100)
    else:
        axes[2].text(0.5, 0.5, "No Delivery % Data", ha="center", va="center", transform=axes[2].transAxes)

    # Model Agreement Count
    if "agreement_count" in df.columns:
        axes[3].bar(df.index, df["agreement_count"], color="#26a69a", alpha=0.85)
        axes[3].set_ylabel("Model Agreement (0-4)")
        axes[3].set_ylim(0, 4)
        axes[3].set_yticks([0, 1, 2, 3, 4])

    axes[3].set_xlabel("Date")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"SEBI case study plot saved to {save_path}")

    return fig


def plot_expiry_comparison(fno_df: pd.DataFrame, save_path: Path = None) -> plt.Figure:
    """
    Boxplot of anomaly scores: Expiry Days vs Non-Expiry Days.
    """
    if "is_expiry_day" not in fno_df.columns or "combined_score" not in fno_df.columns:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    df_plot = fno_df.copy()
    df_plot["Expiry Status"] = df_plot["is_expiry_day"].map({1: "Expiry Day", 0: "Non-Expiry Day"})

    sns.boxplot(x="Expiry Status", y="combined_score", data=df_plot,
                palette=["#ff5252", "#448aff"], ax=ax, width=0.4)

    ax.set_title("Distribution of Combined Anomaly Score\n(Expiry vs Non-Expiry Days)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Combined Anomaly Score (Lower = More Anomalous)")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
