"""
Isolation Forest anomaly detector.

Baseline model — fast, interpretable via feature contribution.
Isolation Forest isolates anomalies by randomly selecting features and
split values, then measuring path length. Anomalies need fewer splits
to be isolated → shorter path length → higher anomaly score.

Good for: high-dimensional data, no assumptions about distribution shape.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """
    Isolation Forest wrapper with multi-contamination sweep
    and feature importance analysis.
    """

    def __init__(self, contamination: float = 0.05,
                 n_estimators: int = 200,
                 random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.feature_names = None

    def fit_predict(self, X: np.ndarray,
                    feature_names: list[str] = None) -> dict:
        """
        Fit the model and return anomaly labels and scores.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        feature_names : list[str]
            Names of features for importance analysis

        Returns
        -------
        dict with keys:
            - 'labels': binary labels (-1 = anomaly, 1 = normal)
            - 'scores': anomaly scores (lower = more anomalous)
            - 'binary': boolean array (True = anomaly)
        """
        self.feature_names = feature_names

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )

        labels = self.model.fit_predict(X)
        scores = self.model.decision_function(X)

        return {
            "labels": labels,
            "scores": scores,
            "binary": labels == -1,
            "n_anomalies": (labels == -1).sum(),
            "anomaly_rate": (labels == -1).mean(),
        }

    def get_feature_importance(self, X: np.ndarray) -> pd.DataFrame:
        """
        Compute feature importance via permutation importance.

        Uses the anomaly score as the "prediction" — features that
        change scores most when shuffled are most important.
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")

        # Permutation importance using the anomaly score
        result = permutation_importance(
            self.model, X,
            n_repeats=10,
            random_state=self.random_state,
            n_jobs=-1,
            scoring="accuracy",  # Not meaningful for unsupervised, but needed
        )

        names = self.feature_names or [f"f_{i}" for i in range(X.shape[1])]
        importance_df = pd.DataFrame({
            "feature": names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False)

        return importance_df

    def save(self, path: Path = None):
        """Save the fitted model."""
        path = path or config.MODELS_DIR / "isolation_forest.joblib"
        joblib.dump(self.model, path)
        logger.info(f"Isolation Forest saved to {path}")

    def load(self, path: Path = None):
        """Load a fitted model."""
        path = path or config.MODELS_DIR / "isolation_forest.joblib"
        self.model = joblib.load(path)
        logger.info(f"Isolation Forest loaded from {path}")


def run_contamination_sweep(X: np.ndarray,
                             feature_names: list[str] = None,
                             contamination_values: list[float] = None) -> dict:
    """
    Run Isolation Forest with multiple contamination values.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    feature_names : list[str]
        Feature names
    contamination_values : list[float]
        Contamination fractions to try

    Returns
    -------
    dict mapping contamination -> result dict
    """
    params = config.MODEL_PARAMS["isolation_forest"]
    contamination_values = contamination_values or params["contamination_values"]

    results = {}
    for c in contamination_values:
        logger.info(f"Isolation Forest (contamination={c})")
        detector = IsolationForestDetector(
            contamination=c,
            n_estimators=params["n_estimators"],
            random_state=params["random_state"],
        )
        result = detector.fit_predict(X, feature_names)
        result["model"] = detector
        results[c] = result
        logger.info(
            f"  Anomalies: {result['n_anomalies']} "
            f"({result['anomaly_rate']:.2%})"
        )

    return results
