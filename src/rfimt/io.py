"""Data-loading helpers used by RFI mitigation notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def segment_start(segment_index: int, nsamp: int = 256, rand: int = 0) -> int:
    """Return the sample offset for a fixed-length segment."""
    return int(segment_index) * int(nsamp) + int(rand)


def read_filterbank_segment(
    filterbank_file: Any,
    segment_index: int,
    nsamp: int = 256,
    rand: int = 0,
) -> np.ndarray:
    """Read one filterbank segment as ``(n_channels, n_time)``.

    ``filterbank_file`` is expected to expose the ``your``-style ``get_data``
    method. The helper deliberately does not import ``your`` so notebooks can
    decide how to open files in their own environment.
    """
    nstart = segment_start(segment_index, nsamp=nsamp, rand=rand)
    return np.asarray(filterbank_file.get_data(nstart=nstart, nsamp=nsamp).T)


def read_metadata(path: str | Path, none_label: str = "None") -> pd.DataFrame:
    """Read a metadata CSV and replace missing labels with ``none_label``."""
    meta = pd.read_csv(path)
    if "label" in meta.columns:
        meta["label"] = meta["label"].fillna(none_label)
    return meta
