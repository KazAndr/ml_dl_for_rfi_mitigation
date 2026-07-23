"""Label helpers for manually defined narrowband-RFI regions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def get_rfi_block_id(segment_index: int, rfi_blocks_for_file: Mapping[int, tuple[int, int]]):
    """Return the RFI block id containing ``segment_index``, or ``None``."""
    for block_id, (start, end) in rfi_blocks_for_file.items():
        if int(start) <= int(segment_index) <= int(end):
            return block_id
    return None


def normalize_channel_ranges(ranges: Sequence) -> list[tuple[int, int]]:
    """Normalize ``[a, b]`` or ``[[a, b], ...]`` into range tuples."""
    if len(ranges) == 0:
        return []
    first = ranges[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        values = ranges
    else:
        values = [ranges]
    return [(int(start), int(end)) for start, end in values]


def get_channel_ranges_for_block(block_id: int, rfi_channels_for_file: Mapping[int, Sequence]):
    """Return channel ranges assigned to one manual RFI block."""
    return normalize_channel_ranges(rfi_channels_for_file[int(block_id)])


def build_rfi_mask(n_channels: int, channel_ranges: Sequence[tuple[int, int]]) -> np.ndarray:
    """Return a boolean vector with ``True`` for RFI channels."""
    mask = np.zeros(int(n_channels), dtype=bool)
    for start, end in channel_ranges:
        s = max(0, int(start))
        e = min(int(n_channels) - 1, int(end))
        if s <= e:
            mask[s : e + 1] = True
    return mask


def label_segment_channels(
    segment_index: int,
    original_segment_label,
    n_channels: int,
    rfi_blocks_for_file: Mapping[int, tuple[int, int]],
    rfi_channels_for_file: Mapping[int, Sequence],
    none_label: str = "None",
    rfi_label: str = "NBRFI",
    hard_negative_label: str = "NoneWNBRFI",
) -> np.ndarray | None:
    """Build channel labels for one segment.

    Returns ``None`` when a segment should be skipped.
    """
    if pd.isna(original_segment_label) or original_segment_label == none_label:
        return np.full(n_channels, none_label, dtype=object)

    if original_segment_label != "bright_NBRFI":
        return None

    block_id = get_rfi_block_id(segment_index, rfi_blocks_for_file)
    if block_id is None:
        return None

    ranges = get_channel_ranges_for_block(block_id, rfi_channels_for_file)
    mask = build_rfi_mask(n_channels, ranges)
    labels = np.full(n_channels, hard_negative_label, dtype=object)
    labels[mask] = rfi_label
    return labels
