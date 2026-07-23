"""Model definitions and factories for RFI mitigation experiments."""

from __future__ import annotations


def make_sklearn_model_set(random_state: int = 42, max_iter: int = 2000):
    """Return the current ladder of statistical baseline models."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import SGDClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    return {
        "SGD_LogReg": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        max_iter=max_iter,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "SGD_LinearSVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SGDClassifier(
                        loss="hinge",
                        max_iter=max_iter,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Poly2_LogReg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        max_iter=max_iter,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "RBFapprox_LogReg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("rbf", RBFSampler(gamma=1.0, n_components=256, random_state=random_state)),
                (
                    "clf",
                    SGDClassifier(
                        loss="log_loss",
                        max_iter=max_iter,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "HistGB": HistGradientBoostingClassifier(random_state=random_state),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=max_iter,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def get_positive_scores(model, x):
    """Return a positive-class score/probability for sklearn-like models."""
    import numpy as np
    from scipy.special import expit

    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return expit(model.decision_function(x))
    return np.asarray(model.predict(x), dtype=float)


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch is optional for docs/light installs.
    torch = None
    nn = None
    F = None


if nn is not None:

    class CNN1DRFI256Logits(nn.Module):
        """1D-CNN baseline for 256-sample channel profiles."""

        def __init__(self, dropout: float = 0.5):
            super().__init__()
            self.conv1 = nn.Conv1d(1, 64, kernel_size=7)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=7)
            self.conv3 = nn.Conv1d(128, 256, kernel_size=5)
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(256 * 57, 256)
            self.fc2 = nn.Linear(256, 1)

        def forward(self, x):
            x = F.max_pool1d(F.relu(self.conv1(x)), kernel_size=2)
            x = F.max_pool1d(F.relu(self.conv2(x)), kernel_size=2)
            x = F.max_pool1d(F.relu(self.conv3(x)), kernel_size=2)
            x = x.flatten(start_dim=1)
            x = self.dropout(F.relu(self.fc1(x)))
            return self.fc2(x).squeeze(-1)

else:

    class CNN1DRFI256Logits:  # pragma: no cover
        """Placeholder raised when PyTorch is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to use CNN1DRFI256Logits")
