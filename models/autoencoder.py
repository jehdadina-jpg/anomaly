"""
Autoencoder-based anomaly detector (PyTorch).

Trains on "normal" trading days; flags days with high reconstruction
error as anomalous. The autoencoder learns the manifold of normal
market behaviour — anything that doesn't compress/reconstruct well
is structurally different (potentially manipulated).

Architecture: input_dim → 64 → 32 → 16 → 32 → 64 → input_dim
Loss: MSE reconstruction error
Anomaly score: reconstruction error per sample
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Autoencoder(nn.Module):
    """
    Symmetric autoencoder for anomaly detection.

    Encoder: input → 64 → 32 → 16 (bottleneck)
    Decoder: 16 → 32 → 64 → input (reconstruction)
    """

    def __init__(self, input_dim: int, hidden_dims: list[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32, 16]

        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers (mirror of encoder)
        decoder_layers = []
        decoder_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        prev_dim = hidden_dims[-1]
        for i, h_dim in enumerate(decoder_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            if i < len(decoder_dims) - 1:
                decoder_layers.extend([
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ])
            # Last layer: no activation (reconstruction)
            prev_dim = h_dim

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


class AutoencoderDetector:
    """
    Autoencoder anomaly detector wrapper.

    Trains on normal data, scores all data by reconstruction error.
    """

    def __init__(self, input_dim: int = None, hidden_dims: list[int] = None,
                 learning_rate: float = 1e-3, batch_size: int = 256,
                 epochs: int = 100, patience: int = 10,
                 threshold_sigmas: float = 3.0,
                 random_state: int = 42):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [64, 32, 16]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.threshold_sigmas = threshold_sigmas
        self.random_state = random_state

        self.model = None
        self.threshold = None
        self.train_losses = []

    def fit(self, X_train: np.ndarray) -> list[float]:
        """
        Train the autoencoder on normal data.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix (presumed normal data)

        Returns
        -------
        list[float]
            Training loss history
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        input_dim = X_train.shape[1]
        self.input_dim = input_dim

        self.model = Autoencoder(input_dim, self.hidden_dims).to(DEVICE)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, verbose=False
        )
        criterion = nn.MSELoss()

        # Prepare DataLoader
        X_tensor = torch.FloatTensor(X_train).to(DEVICE)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop with early stopping
        best_loss = float("inf")
        patience_counter = 0
        self.train_losses = []

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.train_losses.append(avg_loss)
            scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model state
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                logger.info(
                    f"  Epoch {epoch}/{self.epochs}: loss={avg_loss:.6f} "
                    f"(best={best_loss:.6f})"
                )

            if patience_counter >= self.patience:
                logger.info(
                    f"  Early stopping at epoch {epoch} "
                    f"(best loss: {best_loss:.6f})"
                )
                break

        # Restore best model
        self.model.load_state_dict(best_state)

        # Compute threshold from training data reconstruction errors
        train_errors = self._compute_reconstruction_error(X_train)
        self.threshold = np.mean(train_errors) + self.threshold_sigmas * np.std(train_errors)
        logger.info(
            f"  Threshold (mean + {self.threshold_sigmas}σ): {self.threshold:.6f}"
        )
        logger.info(
            f"  Train error stats: mean={np.mean(train_errors):.6f}, "
            f"std={np.std(train_errors):.6f}, "
            f"max={np.max(train_errors):.6f}"
        )

        return self.train_losses

    def _compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample MSE reconstruction error."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(DEVICE)
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            return errors.cpu().numpy()

    def predict(self, X: np.ndarray) -> dict:
        """
        Score all samples by reconstruction error.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix to score

        Returns
        -------
        dict with keys: labels, scores, binary, n_anomalies, anomaly_rate
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")

        errors = self._compute_reconstruction_error(X)

        # Anomaly = reconstruction error above threshold
        binary = errors > self.threshold
        labels = np.where(binary, -1, 1)

        return {
            "labels": labels,
            "scores": -errors,  # Negative so lower = more anomalous (consistent with IF)
            "reconstruction_errors": errors,
            "binary": binary,
            "n_anomalies": binary.sum(),
            "anomaly_rate": binary.mean(),
            "threshold": self.threshold,
        }

    def fit_predict(self, X: np.ndarray,
                    feature_names: list[str] = None,
                    train_fraction: float = 0.9) -> dict:
        """
        Train on "normal" fraction of data, predict on all.

        Uses the bottom `train_fraction` by an initial roughness
        heuristic (row-wise variance) as training data.
        """
        # Heuristic to select "normal" training data:
        # Use the middle portion (exclude extreme rows by L2 norm)
        row_norms = np.linalg.norm(X, axis=1)
        threshold = np.percentile(row_norms, train_fraction * 100)
        train_mask = row_norms <= threshold
        X_train = X[train_mask]

        logger.info(
            f"  Training on {train_mask.sum()}/{len(X)} samples "
            f"(bottom {train_fraction:.0%} by L2 norm)"
        )

        self.fit(X_train)
        return self.predict(X)

    def get_feature_reconstruction_error(self, X: np.ndarray,
                                          feature_names: list[str] = None) -> pd.DataFrame:
        """
        Per-feature reconstruction error for interpretability.

        Shows which features are poorly reconstructed for each sample,
        indicating which aspects of the day were "abnormal".
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(DEVICE)
            reconstructed = self.model(X_tensor).cpu().numpy()

        feature_errors = (X - reconstructed) ** 2
        names = feature_names or [f"f_{i}" for i in range(X.shape[1])]

        return pd.DataFrame(feature_errors, columns=names)

    def save(self, path: Path = None):
        path = path or config.MODELS_DIR / "autoencoder.pt"
        torch.save({
            "model_state": self.model.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "threshold": self.threshold,
            "threshold_sigmas": self.threshold_sigmas,
            "train_losses": self.train_losses,
        }, path)
        logger.info(f"Autoencoder saved to {path}")

    def load(self, path: Path = None):
        path = path or config.MODELS_DIR / "autoencoder.pt"
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        self.input_dim = checkpoint["input_dim"]
        self.hidden_dims = checkpoint["hidden_dims"]
        self.threshold = checkpoint["threshold"]
        self.threshold_sigmas = checkpoint["threshold_sigmas"]
        self.train_losses = checkpoint["train_losses"]
        self.model = Autoencoder(self.input_dim, self.hidden_dims).to(DEVICE)
        self.model.load_state_dict(checkpoint["model_state"])
        logger.info(f"Autoencoder loaded from {path}")


def run_threshold_sweep(X: np.ndarray,
                         feature_names: list[str] = None,
                         sigma_values: list[float] = None) -> dict:
    """Run autoencoder with multiple anomaly thresholds."""
    params = config.MODEL_PARAMS["autoencoder"]
    sigma_values = sigma_values or params["threshold_sigmas"]

    # Train once, then sweep thresholds
    detector = AutoencoderDetector(
        hidden_dims=params["hidden_dims"],
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        patience=params["patience"],
        random_state=params["random_state"],
    )

    # Train on normal data
    row_norms = np.linalg.norm(X, axis=1)
    threshold = np.percentile(row_norms, params["train_fraction"] * 100)
    X_train = X[row_norms <= threshold]
    detector.fit(X_train)

    # Sweep thresholds
    errors = detector._compute_reconstruction_error(X)
    results = {}

    for sigma in sigma_values:
        thresh = np.mean(errors[row_norms <= threshold]) + sigma * np.std(errors[row_norms <= threshold])
        binary = errors > thresh
        labels = np.where(binary, -1, 1)

        results[f"{sigma}sigma"] = {
            "labels": labels,
            "scores": -errors,
            "binary": binary,
            "n_anomalies": binary.sum(),
            "anomaly_rate": binary.mean(),
            "threshold": thresh,
            "model": detector,
        }
        logger.info(
            f"  AE ({sigma}σ): {binary.sum()} anomalies "
            f"({binary.mean():.2%})"
        )

    return results
