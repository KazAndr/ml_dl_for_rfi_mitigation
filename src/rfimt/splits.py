"""Split and subset helpers for channel-level RFI datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd


SPLIT_NAMES = ("train", "val", "test")


def make_group_split_indices(
    meta: pd.DataFrame,
    group_col: str = "segment_index",
    test_size: float = 0.10,
    val_size: float = 0.10,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Create a class-stratified split of whole groups.

    Every group must have one label in the supplied metadata. This is true for
    the fitting universe of the RFI dataset: selected ``NBRFI`` rows belong to
    RFI groups and selected ``None`` rows belong to clean groups. The function
    first splits *groups*, not rows, and only maps those group assignments back
    to row indices afterwards.
    """
    from sklearn.model_selection import train_test_split

    if group_col not in meta.columns:
        raise ValueError(f"Missing group column: {group_col}")
    if not 0 < test_size < 1 or not 0 < val_size < 1:
        raise ValueError("test_size and val_size must be fractions between 0 and 1")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1")

    if "label" not in meta.columns:
        raise ValueError("Missing label column required for group stratification: label")

    group_labels = meta.groupby(group_col, sort=False)["label"].agg(
        lambda values: tuple(pd.unique(values))
    )
    mixed_groups = group_labels[group_labels.map(len).ne(1)]
    if not mixed_groups.empty:
        examples = ", ".join(map(str, mixed_groups.index[:5]))
        raise ValueError(
            "Each split group must have one fitting label; mixed groups include: "
            f"{examples}"
        )

    groups = group_labels.index.to_numpy()
    group_targets = group_labels.map(lambda values: values[0]).to_numpy()
    train_val_groups, test_groups = train_test_split(
        groups,
        test_size=test_size,
        random_state=random_state,
        stratify=group_targets,
    )

    relative_val_size = val_size / (1.0 - test_size)
    train_val_targets = group_labels.loc[train_val_groups].map(
        lambda values: values[0]
    ).to_numpy()
    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=relative_val_size,
        random_state=random_state + 1,
        stratify=train_val_targets,
    )

    row_groups = meta[group_col].to_numpy()

    return {
        "train_idx": np.flatnonzero(np.isin(row_groups, train_groups)),
        "val_idx": np.flatnonzero(np.isin(row_groups, val_groups)),
        "test_idx": np.flatnonzero(np.isin(row_groups, test_groups)),
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


def summarize_group_splits(
    meta: pd.DataFrame,
    splits: dict[str, np.ndarray],
    group_col: str = "segment_index",
    label_col: str = "label",
) -> pd.DataFrame:
    """Return row, group and label counts for a declared split.

    The summary is deliberately calculated from the metadata used by the
    split, so a notebook can save it beside the generated indices as a small
    audit artifact. It does not infer scientific labels or rebalance data.
    """
    missing = {group_col, label_col}.difference(meta.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    unexpected = set(splits).difference(SPLIT_NAMES)
    if unexpected:
        raise ValueError(f"Unexpected split names: {sorted(unexpected)}")

    rows: list[dict[str, object]] = []
    for split_name in SPLIT_NAMES:
        split_idx = np.asarray(splits.get(split_name, []), dtype=int)
        split_meta = meta.iloc[split_idx]
        counts = split_meta[label_col].value_counts(dropna=False)
        for label, count in counts.items():
            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "row_count": int(count),
                    "group_count": int(split_meta[group_col].nunique()),
                }
            )

    return pd.DataFrame(rows, columns=["split", "label", "row_count", "group_count"])


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
