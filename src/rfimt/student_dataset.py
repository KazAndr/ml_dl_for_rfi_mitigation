"""Helpers for a small, auditable RFI data package used in supervision."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


STUDENT_CORE_LABELS = ("BGN", "NBRFI")
SOURCE_TO_STUDENT_LABEL = {"None": "BGN", "NBRFI": "NBRFI"}
# The full raw CSV encodes background as an empty field, while the curated core
# metadata spells the same source class as "None".
SOURCE_BACKGROUND_LABELS = frozenset({"", "None"})


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash an exported artifact so its manifest can detect accidental drift."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def map_source_labels_to_student(source_labels: pd.Series) -> pd.Series:
    """Rename source annotations without changing the research data contract."""
    unexpected = sorted(set(source_labels.astype(str)) - set(SOURCE_TO_STUDENT_LABEL))
    if unexpected:
        raise ValueError(f"Unexpected core source labels: {unexpected}")
    return source_labels.astype(str).map(SOURCE_TO_STUDENT_LABEL)


def assign_split_column(meta: pd.DataFrame, splits: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Attach declared split names and reject incomplete or overlapping rows."""
    result = meta.copy()
    split = np.full(len(result), "", dtype=object)
    for name in ("train", "val", "test"):
        indices = np.asarray(splits[name], dtype=int)
        if np.any(split[indices] != ""):
            raise ValueError("A core row appears in more than one declared split.")
        split[indices] = name
    if np.any(split == ""):
        raise ValueError("Some core rows are not assigned to train, val or test.")
    result["split"] = split
    return result


def assert_disjoint_split_groups(meta: pd.DataFrame, group_col: str = "segment_index") -> None:
    """Ensure correlated channels cannot leak between core split partitions."""
    if group_col not in meta:
        raise ValueError(f"Missing group column: {group_col}")
    groups_by_split = {
        split: set(frame[group_col].tolist())
        for split, frame in meta.groupby("split", sort=False)
    }
    names = tuple(groups_by_split)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = groups_by_split[left] & groups_by_split[right]
            if overlap:
                raise ValueError(f"Split groups overlap between {left} and {right}: {next(iter(overlap))}")


def select_challenge_segments(
    full_meta: pd.DataFrame,
    excluded_segments: Sequence[object],
    n_clean: int,
    n_hard: int,
    random_state: int,
    group_col: str = "segment_index",
) -> pd.DataFrame:
    """Choose clean and RFI-containing segments outside the student's core data."""
    excluded = set(excluded_segments)
    rows = []
    for segment, frame in full_meta.groupby(group_col, sort=False):
        if segment in excluded:
            continue
        labels = set(frame["label"].astype(str))
        if labels and labels <= SOURCE_BACKGROUND_LABELS:
            rows.append({group_col: segment, "challenge_kind": "clean"})
        elif "NoneWNBRFI" in labels:
            rows.append({group_col: segment, "challenge_kind": "rfi_containing"})

    candidates = pd.DataFrame(rows)
    rng = np.random.default_rng(random_state)
    selected = []
    for kind, requested in (("clean", n_clean), ("rfi_containing", n_hard)):
        available = candidates.loc[candidates["challenge_kind"].eq(kind), group_col].to_numpy()
        if len(available) < requested:
            raise ValueError(f"Need {requested} {kind} challenge segments, found {len(available)}.")
        selected.extend(
            {group_col: segment, "challenge_kind": kind}
            for segment in rng.choice(available, size=requested, replace=False)
        )
    return pd.DataFrame(selected).reset_index(drop=True)


def stack_complete_segments(
    full_array: np.ndarray,
    full_meta: pd.DataFrame,
    segment_ids: Sequence[object],
    group_col: str = "segment_index",
    channel_col: str = "channel_index",
    expected_channels: int = 256,
    expected_time: int = 256,
) -> np.ndarray:
    """Materialize complete ordered spectrograms for visual student exercises."""
    if full_array.ndim != 2 or full_array.shape[1] != expected_time:
        raise ValueError(f"Expected full array shape (rows, {expected_time}), got {full_array.shape}.")
    spectra = []
    for segment in segment_ids:
        frame = full_meta.loc[full_meta[group_col].eq(segment)].sort_values(channel_col)
        if len(frame) != expected_channels:
            raise ValueError(f"Segment {segment!r} has {len(frame)} rows, expected {expected_channels}.")
        source_indices = frame["source_row_index"].to_numpy(dtype=int)
        spectra.append(np.asarray(full_array[source_indices], dtype=np.float32))
    return np.stack(spectra, axis=0)


def student_readme() -> str:
    """Return a beginner-facing package guide without revealing challenge labels."""
    return """# RFI Channel Anomaly Detection: Student Data Package

## Purpose

This package supports a first supervised machine-learning exercise on radio
observation data. The task is to distinguish frequency channels marked as
`NBRFI` from channels marked as `BGN` (background noise).

The goal is not to create a deployable RFI-cleaning system. It is to practice
loading scientific arrays, inspecting data, defining a binary target, keeping
training/validation/test roles separate, training a simple model and analysing
its mistakes.

## Data Objects

- A **spectrogram segment** is a small dynamic-spectrum image with shape
  `(256 frequency channels, 256 time samples)`.
- A **channel time series** is one row of such a segment with shape `(256,)`.
- `core_channel_time_series.npy` contains one channel time series per training
  example. It is not a pulsar average profile.
- `core_channel_metadata.csv` identifies every core example and contains its
  binary label and declared split.
- `challenge_spectrogram_segments.npy` contains complete spectrograms for a
  later visual generalisation check. Their channel labels are deliberately not
  included in this student package.

## Labels

- `BGN` means background noise: no selected narrow-band RFI was identified for
  the channel.
- `NBRFI` means the channel lies in a manually selected narrow-band RFI range.

The original research annotation called the background class `None`. It is
renamed to `BGN` here so that it cannot be confused with a missing value.
These labels are practical annotations for this exercise, not a complete
physical description of every signal in the observation.

## Splits

All rows from one `segment_index` belong to exactly one of `train`, `val` or
`test`.

- Use **train** to fit model parameters.
- Use **validation** to compare choices such as model settings or a threshold.
- Use **test** only after those choices are fixed, for one final reported
  evaluation.

## Challenge Spectrograms

The challenge contains both clean segments and segments containing more subtle
RFI structure, including channels marked `NoneWNBRFI` in the full research
annotations. Do not use these spectra to tune the model. First make predictions
and inspect the predicted channel masks; the supervisor will provide the
corresponding reference labels for discussion afterwards.

This challenge is a qualitative visual exercise, not an additional independent
test split. Some complete spectrograms may contain channel time series already
present in the core package, but their challenge labels remain hidden.
"""
