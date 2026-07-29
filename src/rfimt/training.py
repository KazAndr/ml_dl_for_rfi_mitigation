"""Small PyTorch data and scoring helpers for profile baselines."""

from __future__ import annotations

import numpy as np


def normalize_profile_rows(x, variant: str):
    """Normalize a two-dimensional array of channel-profile rows."""
    values = np.asarray(x)
    if values.ndim != 2:
        raise ValueError("Profile rows must be a 2D array with shape (rows, time).")
    if variant == "none":
        return values.astype(np.float32, copy=False)
    values = values.astype(np.float32, copy=False)
    if variant == "legacy_max_per_channel":
        maximum = values.max(axis=1, keepdims=True)
        maximum = np.where(maximum < 1e-8, 1.0, maximum)
        return values / maximum
    mean = values.mean(axis=1, keepdims=True)
    if variant == "zscore_per_channel":
        std = values.std(axis=1, keepdims=True)
    else:
        raise ValueError(
            "Unsupported normalization variant {!r}; expected 'none', "
            "'legacy_max_per_channel', or 'zscore_per_channel'. "
            "Use normalize_profile_segments for zscore_per_segment.".format(variant)
        )

    # Constant inputs have no scale; preserve finite, centered input values.
    std = np.where(std == 0, 1.0, std)
    return (values - mean) / std


def normalize_profile_segments(x, segment_ids):
    """Z-score every complete spectrogram segment independently.

    ``x`` must contain all channel rows for the requested segments. This is
    intentionally separate from ``ProfileDataset``: a DataLoader batch is not
    a physical observation segment and must never define this normalization.
    """
    values = np.asarray(x, dtype=np.float32)
    groups = np.asarray(segment_ids)
    if values.ndim != 2:
        raise ValueError("Profile rows must be a 2D array with shape (rows, time).")
    if len(values) != len(groups):
        raise ValueError("Segment identifiers must align with profile rows.")

    _, inverse, row_counts = np.unique(groups, return_inverse=True, return_counts=True)
    n_values = row_counts * values.shape[1]
    row_sums = values.sum(axis=1, dtype=np.float64)
    row_square_sums = np.square(values, dtype=np.float64).sum(axis=1)
    sums = np.bincount(inverse, weights=row_sums)
    square_sums = np.bincount(inverse, weights=row_square_sums)
    means = sums / n_values
    variances = np.maximum(square_sums / n_values - np.square(means), 0.0)
    stds = np.sqrt(variances)
    stds = np.where(stds == 0, 1.0, stds)
    return ((values - means[inverse, None]) / stds[inverse, None]).astype(np.float32)


try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - torch is optional for light installs.
    torch = None
    Dataset = object


if torch is not None:

    class ProfileDataset(Dataset):
        """Dataset exposing profiles in the channel-first shape expected by Conv1d."""

        def __init__(self, x, y, normalization: str = "none"):
            profiles = normalize_profile_rows(x, normalization)
            targets = np.asarray(y)
            if targets.ndim != 1:
                raise ValueError("Targets must be a 1D array aligned with profile rows.")
            if len(profiles) != len(targets):
                raise ValueError("Profiles and targets must contain the same number of rows.")
            self.profiles = torch.from_numpy(profiles[:, np.newaxis, :])
            self.targets = torch.as_tensor(targets, dtype=torch.float32)

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return self.profiles[index], self.targets[index]


    def collect_torch_scores(model, loader, device):
        """Collect sigmoid probabilities while preserving loader sample order."""
        was_training = model.training
        model.eval()
        targets = []
        scores = []
        with torch.no_grad():
            for x, y in loader:
                logits = model(x.to(device))
                scores.append(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))
                targets.append(y.detach().cpu().numpy().reshape(-1))
        if was_training:
            model.train()
        return np.concatenate(targets), np.concatenate(scores)


else:

    class ProfileDataset:  # pragma: no cover
        """Placeholder raised when PyTorch is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to use ProfileDataset")


    def collect_torch_scores(model, loader, device):  # pragma: no cover
        """Placeholder raised when PyTorch is not installed."""
        raise ImportError("PyTorch is required to collect torch scores")
