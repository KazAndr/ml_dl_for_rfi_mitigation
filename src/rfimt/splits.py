"""Split and subset helpers for channel-level RFI datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_group_split_indices(
    meta: pd.DataFrame,
    group_col: str = "segment_index",
    test_size: float = 0.10,
    val_size: float = 0.10,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Create train/validation/test row indices without group leakage."""
    from sklearn.model_selection import GroupShuffleSplit

    if group_col not in meta.columns:
        raise ValueError(f"Missing group column: {group_col}")
    if not 0 < test_size < 1 or not 0 < val_size < 1:
        raise ValueError("test_size and val_size must be fractions between 0 and 1")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1")

    idx = np.arange(len(meta))
    groups = meta[group_col].to_numpy()

    first = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(first.split(idx, groups=groups))

    relative_val_size = val_size / (1.0 - test_size)
    second = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_val_size,
        random_state=random_state + 1,
    )
    train_rel, val_rel = next(
        second.split(train_val_idx, groups=groups[train_val_idx])
    )

    return {
        "train_idx": train_val_idx[train_rel],
        "val_idx": train_val_idx[val_rel],
        "test_idx": test_idx,
    }


def assert_no_group_overlap(
    meta: pd.DataFrame,
    splits: dict[str, np.ndarray],
    group_col: str = "segment_index",
) -> None:
    """Raise if any group appears in more than one split."""
    seen: dict[object, str] = {}
    for split_name, split_idx in splits.items():
        for group in meta.iloc[split_idx][group_col].unique():
            if group in seen:
                raise AssertionError(
                    f"Group {group!r} appears in both {seen[group]} and {split_name}"
                )
            seen[group] = split_name


def sample_balanced_within_indices(
    meta: pd.DataFrame,
    candidate_idx: np.ndarray,
    label_col: str = "label",
    labels_to_sample: tuple[str, ...] = ("NBRFI", "None"),
    n_per_class: int = 10_000,
    random_state: int = 42,
) -> np.ndarray:
    """Sample a balanced subset from a preselected row-index universe."""
    rng = np.random.default_rng(random_state)
    candidate_idx = np.asarray(candidate_idx)
    selected = []
    labels = meta[label_col].fillna("None").to_numpy()

    for label in labels_to_sample:
        label_idx = candidate_idx[labels[candidate_idx] == label]
        if len(label_idx) == 0:
            raise ValueError(f"No rows available for label={label!r}")
        k = min(int(n_per_class), len(label_idx))
        selected.append(rng.choice(label_idx, size=k, replace=False))

    result = np.concatenate(selected)
    rng.shuffle(result)
    return result
