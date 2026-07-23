"""
Ensemble model — runs all four detectors and computes cross-model agreement.

This is the core of the evaluation strategy: individual models may have
different false-positive profiles, but anomalies flagged by multiple
independent methods are much more likely to be genuine.

Cross-model agreement is the strongest evidence in an unsupervised setting.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from models.isolation_forest import IsolationForestDetector
from models.ocsvm import OCSVMDetector
from models.autoencoder import AutoencoderDetector
from models.lof import LOFDetector

logger = logging.getLogger(__name__)


def run_all_models(X: np.ndarray,
                    feature_names: list[str] = None,
                    test_mode: bool = False) -> dict:
    """
    Run all four anomaly detection models.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix (n_samples, n_features)
    feature_names : list[str]
        Feature names for interpretability
    test_mode : bool
        If True, use faster parameter settings

    Returns
    -------
    dict with keys:
        - 'isolation_forest': result dict
        - 'ocsvm': result dict
        - 'autoencoder': result dict
        - 'lof': result dict
        - 'agreement': agreement analysis
    """
    results = {}
    n_samples, n_features = X.shape
    logger.info(f"Running all models on {n_samples} samples × {n_features} features")

    # ---- 1. Isolation Forest ----
    logger.info("=" * 50)
    logger.info("Model 1/4: Isolation Forest")
    logger.info("=" * 50)

    if_params = config.MODEL_PARAMS["isolation_forest"]
    if_contamination = 0.05  # default contamination
    if_detector = IsolationForestDetector(
        contamination=if_contamination,
        n_estimators=if_params["n_estimators"] if not test_mode else 50,
        random_state=if_params["random_state"],
    )
    results["isolation_forest"] = if_detector.fit_predict(X, feature_names)
    results["isolation_forest"]["model"] = if_detector
    logger.info(
        f"  → {results['isolation_forest']['n_anomalies']} anomalies "
        f"({results['isolation_forest']['anomaly_rate']:.2%})"
    )

    # ---- 2. One-Class SVM ----
    logger.info("=" * 50)
    logger.info("Model 2/4: One-Class SVM")
    logger.info("=" * 50)

    ocsvm_params = config.MODEL_PARAMS["ocsvm"]
    ocsvm_detector = OCSVMDetector(
        nu=0.05,
        kernel=ocsvm_params["kernel"],
        gamma=ocsvm_params["gamma"],
        max_train_samples=ocsvm_params["max_train_samples"] if not test_mode else 2000,
    )
    results["ocsvm"] = ocsvm_detector.fit_predict(X, feature_names)
    results["ocsvm"]["model"] = ocsvm_detector
    logger.info(
        f"  → {results['ocsvm']['n_anomalies']} anomalies "
        f"({results['ocsvm']['anomaly_rate']:.2%})"
    )

    # ---- 3. Autoencoder ----
    logger.info("=" * 50)
    logger.info("Model 3/4: Autoencoder")
    logger.info("=" * 50)

    ae_params = config.MODEL_PARAMS["autoencoder"]
    ae_detector = AutoencoderDetector(
        hidden_dims=ae_params["hidden_dims"],
        learning_rate=ae_params["learning_rate"],
        batch_size=ae_params["batch_size"],
        epochs=ae_params["epochs"] if not test_mode else 20,
        patience=ae_params["patience"] if not test_mode else 5,
        threshold_sigmas=3.0,
        random_state=ae_params["random_state"],
    )
    results["autoencoder"] = ae_detector.fit_predict(
        X, feature_names,
        train_fraction=ae_params["train_fraction"],
    )
    results["autoencoder"]["model"] = ae_detector
    logger.info(
        f"  → {results['autoencoder']['n_anomalies']} anomalies "
        f"({results['autoencoder']['anomaly_rate']:.2%})"
    )

    # ---- 4. LOF ----
    logger.info("=" * 50)
    logger.info("Model 4/4: Local Outlier Factor")
    logger.info("=" * 50)

    lof_params = config.MODEL_PARAMS["lof"]
    lof_detector = LOFDetector(
        n_neighbors=20,
        contamination=lof_params["contamination"],
        metric=lof_params["metric"],
    )
    results["lof"] = lof_detector.fit_predict(X, feature_names)
    results["lof"]["model"] = lof_detector
    logger.info(
        f"  → {results['lof']['n_anomalies']} anomalies "
        f"({results['lof']['anomaly_rate']:.2%})"
    )

    # ---- Cross-Model Agreement ----
    logger.info("=" * 50)
    logger.info("Computing cross-model agreement")
    logger.info("=" * 50)

    results["agreement"] = compute_agreement(results)

    return results


def compute_agreement(model_results: dict) -> dict:
    """
    Compute cross-model agreement statistics.

    For each stock-day, count how many models flag it as anomalous.
    High agreement = high confidence that the anomaly is real.
    """
    model_names = ["isolation_forest", "ocsvm", "autoencoder", "lof"]

    # Build agreement matrix
    binary_arrays = []
    for name in model_names:
        if name in model_results and "binary" in model_results[name]:
            binary_arrays.append(model_results[name]["binary"].astype(int))

    if not binary_arrays:
        return {}

    agreement_matrix = np.column_stack(binary_arrays)
    agreement_count = agreement_matrix.sum(axis=1)  # 0-4 models agree

    # Agreement distribution
    agreement_dist = {}
    for i in range(len(model_names) + 1):
        count = (agreement_count == i).sum()
        agreement_dist[f"{i}_models"] = int(count)

    # Consensus anomalies (at least N models agree)
    min_agreement = config.ENSEMBLE_MIN_AGREEMENT
    consensus_binary = agreement_count >= min_agreement

    logger.info(f"Agreement distribution:")
    for k, v in agreement_dist.items():
        logger.info(f"  {k}: {v} stock-days")
    logger.info(
        f"Consensus anomalies (≥{min_agreement}/4): {consensus_binary.sum()} "
        f"({consensus_binary.mean():.2%})"
    )

    # Normalize anomaly scores across models for a combined score
    scores = []
    for name in model_names:
        if name in model_results and "scores" in model_results[name]:
            s = model_results[name]["scores"]
            # Normalize to [0, 1] range (0 = most anomalous)
            s_min, s_max = s.min(), s.max()
            if s_max > s_min:
                s_norm = (s - s_min) / (s_max - s_min)
            else:
                s_norm = np.zeros_like(s)
            scores.append(s_norm)

    if scores:
        # Average normalized score (lower = more anomalous)
        combined_score = np.mean(scores, axis=0)
    else:
        combined_score = np.zeros(agreement_count.shape)

    return {
        "agreement_count": agreement_count,
        "agreement_distribution": agreement_dist,
        "consensus_binary": consensus_binary,
        "consensus_n_anomalies": int(consensus_binary.sum()),
        "consensus_anomaly_rate": float(consensus_binary.mean()),
        "combined_score": combined_score,
        "model_names": model_names,
    }


def build_results_dataframe(panel: pd.DataFrame,
                             model_results: dict) -> pd.DataFrame:
    """
    Attach model results back to the feature panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Feature panel with ticker and date info
    model_results : dict
        Output from run_all_models()

    Returns
    -------
    pd.DataFrame
        Panel with anomaly flags and scores for each model
    """
    out = panel.copy()

    model_names = ["isolation_forest", "ocsvm", "autoencoder", "lof"]

    for name in model_names:
        if name in model_results:
            r = model_results[name]
            out[f"anomaly_{name}"] = r["binary"].astype(int)
            out[f"score_{name}"] = r["scores"]

    if "agreement" in model_results:
        a = model_results["agreement"]
        out["agreement_count"] = a["agreement_count"]
        out["consensus_anomaly"] = a["consensus_binary"].astype(int)
        out["combined_score"] = a["combined_score"]

    return out


def save_results(results_df: pd.DataFrame, model_results: dict):
    """Save all results to disk."""
    # Save annotated panel
    results_path = config.RESULTS_DIR / "anomaly_results.csv"
    results_df.to_csv(results_path)
    logger.info(f"Results saved to {results_path}")

    # Save models
    for name in ["isolation_forest", "ocsvm", "autoencoder", "lof"]:
        if name in model_results and "model" in model_results[name]:
            try:
                model_results[name]["model"].save()
            except Exception as e:
                logger.error(f"Failed to save {name}: {e}")

    # Save agreement summary
    if "agreement" in model_results:
        summary = {
            k: v for k, v in model_results["agreement"].items()
            if k not in ("agreement_count", "consensus_binary", "combined_score")
        }
        summary_path = config.RESULTS_DIR / "agreement_summary.txt"
        with open(summary_path, "w") as f:
            for k, v in summary.items():
                f.write(f"{k}: {v}\n")

    logger.info("All models and results saved")
