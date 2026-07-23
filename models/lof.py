"""
Local Outlier Factor (LOF) anomaly detector.

Density-based approach — good for detecting anomalies in clusters.
LOF compares the local density of a point with the local densities
of its neighbors. Points with substantially lower density are outliers.

Good for: anomalies that are normal in some feature dimensions but
form sparse clusters in the joint feature space.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class LOFDetector:
    """
    Local Outlier Factor wrapper with n_neighbors sweep.
    """

    def __init__(self, n_neighbors: int = 20,
                 contamination: float = 0.05,
                 metric: str = "euclidean"):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.model = None

    def fit_predict(self, X: np.ndarray,
                    feature_names: list[str] = None) -> dict:
        """
        Fit LOF and return anomaly labels/scores.

        Note: LOF is transductive — fit and predict happen together
        (novelty=False). For novelty detection, set novelty=True
        and use separate fit/predict.
        """
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            metric=self.metric,
            n_jobs=-1,
        )

        labels = self.model.fit_predict(X)
        # LOF negative_outlier_factor_ is the LOF score (more negative = more outlier)
        scores = self.model.negative_outlier_factor_

        return {
            "labels": labels,
            "scores": scores,  # Already negative for outliers
            "binary": labels == -1,
            "n_anomalies": (labels == -1).sum(),
            "anomaly_rate": (labels == -1).mean(),
        }

    def save(self, path: Path = None):
        path = path or config.MODELS_DIR / "lof.joblib"
        joblib.dump(self.model, path)
        logger.info(f"LOF saved to {path}")


def run_neighbors_sweep(X: np.ndarray,
                         feature_names: list[str] = None,
                         n_neighbors_values: list[int] = None) -> dict:
    """
    Run LOF with multiple n_neighbors values.

    n_neighbors controls the locality of density estimation:
    - Small n_neighbors: very local density → catches micro-outliers
    - Large n_neighbors: broader density → catches larger-scale anomalies
    """
    params = config.MODEL_PARAMS["lof"]
    n_neighbors_values = n_neighbors_values or params["n_neighbors_values"]

    results = {}
    for n in n_neighbors_values:
        logger.info(f"LOF (n_neighbors={n})")
        detector = LOFDetector(
            n_neighbors=n,
            contamination=params["contamination"],
            metric=params["metric"],
        )
        result = detector.fit_predict(X, feature_names)
        result["model"] = detector
        results[n] = result
        logger.info(
            f"  Anomalies: {result['n_anomalies']} "
            f"({result['anomaly_rate']:.2%})"
        )

    return results
