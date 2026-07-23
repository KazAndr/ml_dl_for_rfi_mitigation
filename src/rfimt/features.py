"""Feature extraction and normalization for channel-level RFI data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import FEATURE_COLUMNS


def compute_stats(x: np.ndarray, axis=None):
    """Return mean, std, skewness and excess kurtosis along ``axis``.

    The implementation intentionally avoids SciPy so the feature contract is
    easy to reuse in notebooks and small scripts.
    """
    x = np.asarray(x)
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)

    centered = x - mean
    m3 = (centered**3).mean(axis=axis, keepdims=True)
    m4 = (centered**4).mean(axis=axis, keepdims=True)

    std_safe = np.where(std == 0, 1.0, std)
    skew = m3 / (std_safe**3)
    kurt = m4 / (std_safe**4) - 3.0

    return mean.squeeze(), std.squeeze(), skew.squeeze(), kurt.squeeze()


def normalize_minmax_per_channel(seg: np.ndarray, scale: float = 256.0) -> np.ndarray:
    """Normalize each channel independently into ``[0, scale]``."""
    seg = np.asarray(seg, dtype=float)
    chan_min = seg.min(axis=1, keepdims=True)
    chan_max = seg.max(axis=1, keepdims=True)
    denom = np.where(chan_max - chan_min == 0, 1.0, chan_max - chan_min)
    return (seg - chan_min) / denom * scale


def normalize_zscore_per_channel(seg: np.ndarray) -> np.ndarray:
    """Apply z-score normalization independently to each channel."""
    seg = np.asarray(seg, dtype=float)
    mean = seg.mean(axis=1, keepdims=True)
    std = seg.std(axis=1, keepdims=True)
    return (seg - mean) / np.where(std == 0, 1.0, std)


def normalize_zscore_per_segment(seg: np.ndarray) -> np.ndarray:
    """Apply one z-score normalization to the whole segment."""
    seg = np.asarray(seg, dtype=float)
    std = seg.std()
    return (seg - seg.mean()) / (1.0 if std == 0 else std)


def normalize_robust_mad_per_channel(seg: np.ndarray, scale: float = 1.4826) -> np.ndarray:
    """Apply robust median/MAD normalization independently to each channel."""
    seg = np.asarray(seg, dtype=float)
    median = np.median(seg, axis=1, keepdims=True)
    mad = np.median(np.abs(seg - median), axis=1, keepdims=True)
    return (seg - median) / np.where(mad == 0, 1.0, mad * scale)


def compute_ratio(per_channel_vals: np.ndarray, seg_val: float) -> np.ndarray:
    """Return channel values divided by the full-segment value."""
    per_channel_vals = np.asarray(per_channel_vals, dtype=float)
    if seg_val == 0:
        return np.full_like(per_channel_vals, np.nan, dtype=float)
    return per_channel_vals / seg_val


def extract_channel_features(seg: np.ndarray) -> pd.DataFrame:
    """Extract the current 16 statistical features for all channels.

    Parameters
    ----------
    seg:
        Dynamic-spectrum segment with shape ``(n_channels, n_time)``.
    """
    seg = np.asarray(seg)
    seg_norm = normalize_minmax_per_channel(seg)

    mean_o_seg, std_o_seg, skew_o_seg, kurt_o_seg = compute_stats(seg)
    mean_n_seg, std_n_seg, skew_n_seg, kurt_n_seg = compute_stats(seg_norm)

    mean_o, std_o, skew_o, kurt_o = compute_stats(seg, axis=1)
    mean_n, std_n, skew_n, kurt_n = compute_stats(seg_norm, axis=1)

    features = pd.DataFrame(
        {
            "mean_o": mean_o,
            "std_o": std_o,
            "skew_o": skew_o,
            "kurt_o": kurt_o,
            "mean_n": mean_n,
            "std_n": std_n,
            "skew_n": skew_n,
            "kurt_n": kurt_n,
            "mean_o_ratio": compute_ratio(mean_o, mean_o_seg),
            "std_o_ratio": compute_ratio(std_o, std_o_seg),
            "skew_o_ratio": compute_ratio(skew_o, skew_o_seg),
            "kurt_o_ratio": compute_ratio(kurt_o, kurt_o_seg),
            "mean_n_ratio": compute_ratio(mean_n, mean_n_seg),
            "std_n_ratio": compute_ratio(std_n, std_n_seg),
            "skew_n_ratio": compute_ratio(skew_n, skew_n_seg),
            "kurt_n_ratio": compute_ratio(kurt_n, kurt_n_seg),
        }
    )
    return features[FEATURE_COLUMNS]
