"""Human-readable visual review artifacts for channel-mask experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


def build_segment_review_queue(segment_table: pd.DataFrame, n_per_category: int) -> pd.DataFrame:
    """Select diverse, auditable segments instead of only the largest failures."""
    if n_per_category < 1:
        raise ValueError("n_per_category must be positive.")

    table = segment_table.copy()
    table["error_count"] = table[["FP", "FN"]].sum(axis=1)
    table["threshold_distance"] = (table["mean_score"] - table["threshold"]).abs()
    selections = (
        ("mixed_errors", table.loc[(table["FP"] > 0) & (table["FN"] > 0)].sort_values("error_count", ascending=False)),
        ("fp_heavy", table.loc[table["FP"] > 0].sort_values("FP", ascending=False)),
        ("fn_heavy", table.loc[table["FN"] > 0].sort_values("FN", ascending=False)),
        ("positive_recovery", table.loc[table["TP"] > 0].sort_values("TP", ascending=False)),
        ("clean_reference", table.loc[(table["TN"] > 0) & (table["FP"] == 0) & (table["FN"] == 0)].sort_values("TN", ascending=False)),
        ("threshold_borderline", table.sort_values("threshold_distance")),
    )

    queue = []
    seen = set()
    for reason, rows in selections:
        selected = 0
        for _, row in rows.iterrows():
            segment_index = row["segment_index"]
            if segment_index in seen:
                continue
            queue.append({"segment_index": segment_index, "review_reason": reason})
            seen.add(segment_index)
            selected += 1
            if selected == n_per_category:
                break
    return pd.DataFrame(queue)


def save_spectrogram_mask_pair(
    profiles: np.ndarray,
    predicted_mask: np.ndarray,
    destination: str | Path,
    title: str,
) -> None:
    """Save the legacy-style original and black predicted-mask comparison."""
    data = np.asarray(profiles)
    mask = np.asarray(predicted_mask, dtype=bool)
    if data.ndim != 2 or len(data) != len(mask):
        raise ValueError("Profiles and predicted mask must describe the same channel rows.")

    overlay = np.repeat(mask[:, None], data.shape[1], axis=1).astype(float)
    black_mask = ListedColormap([(0, 0, 0, 0.0), (0, 0, 0, 1.0)])
    vmin, vmax = float(np.min(data)), float(np.max(data))
    if vmin == vmax:
        vmax = vmin + 1.0

    fig, (original, masked) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(title, fontsize=12)
    original.imshow(data, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto", origin="upper")
    original.set(title="Original", xlabel="Time sample", ylabel="Frequency channel")
    masked.imshow(data, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto", origin="upper")
    masked.imshow(overlay, cmap=black_mask, vmin=0, vmax=1, aspect="auto", origin="upper")
    masked.set(title="With predicted channel mask", xlabel="Time sample")
    fig.tight_layout()
    fig.savefig(destination, dpi=250, bbox_inches="tight")
    plt.close(fig)
