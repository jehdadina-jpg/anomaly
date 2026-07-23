"""
One-Class SVM anomaly detector.

Second baseline — different decision boundary shape from Isolation Forest.
OC-SVM fits a tight boundary around normal data in kernel space; points
outside the boundary are anomalies.

Trade-off: more computationally expensive (O(n²) to O(n³)), so we
subsample large datasets.

Good for: when the normal class has a well-defined shape in feature space.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class OCSVMDetector:
    """
    One-Class SVM wrapper with nu parameter sweep and subsampling.
    """

    def __init__(self, nu: float = 0.05, kernel: str = "rbf",
                 gamma: str = "scale", max_train_samples: int = 10000):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.max_train_samples = max_train_samples
        self.model = None

    def fit_predict(self, X: np.ndarray,
                    feature_names: list[str] = None) -> dict:
        """
        Fit OC-SVM and return anomaly labels/scores.

        If X is larger than max_train_samples, we subsample for training
        but predict on the full dataset.
        """
        n_samples = X.shape[0]

        self.model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma,
        )

        # Subsample if needed (OC-SVM scales poorly)
        if n_samples > self.max_train_samples:
            logger.info(
                f"  Subsampling: {n_samples} -> {self.max_train_samples} "
                f"for training"
            )
            rng = np.random.RandomState(42)
            train_idx = rng.choice(n_samples, self.max_train_samples, replace=False)
            X_train = X[train_idx]
        else:
            X_train = X

        # Fit on (possibly subsampled) training data
        self.model.fit(X_train)

        # Predict on full data
        labels = self.model.predict(X)
        scores = self.model.decision_function(X)

        return {
            "labels": labels,
            "scores": scores,
            "binary": labels == -1,
            "n_anomalies": (labels == -1).sum(),
            "anomaly_rate": (labels == -1).mean(),
        }

    def save(self, path: Path = None):
        path = path or config.MODELS_DIR / "ocsvm.joblib"
        joblib.dump(self.model, path)
        logger.info(f"OC-SVM saved to {path}")

    def load(self, path: Path = None):
        path = path or config.MODELS_DIR / "ocsvm.joblib"
        self.model = joblib.load(path)


def run_nu_sweep(X: np.ndarray,
                  feature_names: list[str] = None,
                  nu_values: list[float] = None) -> dict:
    """
    Run OC-SVM with multiple nu values.

    nu approximately controls the fraction of anomalies
    (upper bound on the fraction of training errors / support vectors).
    """
    params = config.MODEL_PARAMS["ocsvm"]
    nu_values = nu_values or params["nu_values"]

    results = {}
    for nu in nu_values:
        logger.info(f"OC-SVM (nu={nu})")
        detector = OCSVMDetector(
            nu=nu,
            kernel=params["kernel"],
            gamma=params["gamma"],
            max_train_samples=params["max_train_samples"],
        )
        result = detector.fit_predict(X, feature_names)
        result["model"] = detector
        results[nu] = result
        logger.info(
            f"  Anomalies: {result['n_anomalies']} "
            f"({result['anomaly_rate']:.2%})"
        )

    return results
