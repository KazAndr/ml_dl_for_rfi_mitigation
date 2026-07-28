"""Tests for group-split invariants."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rfimt.splits import (
    assert_no_group_overlap,
    make_group_split_indices,
    summarize_group_splits,
)


class GroupSplitTests(unittest.TestCase):
    def setUp(self):
        group_labels = np.where(np.arange(30) % 2 == 0, "NBRFI", "None")
        self.meta = pd.DataFrame(
            {
                "segment_index": np.repeat(np.arange(30), 4),
                "label": np.repeat(group_labels, 4),
            }
        )
        self.splits = make_group_split_indices(self.meta, random_state=42)

    def test_group_split_has_no_overlap(self):
        assert_no_group_overlap(self.meta, self.splits)

    def test_summary_accounts_for_every_row(self):
        summary = summarize_group_splits(self.meta, self.splits)
        self.assertEqual(int(summary["row_count"].sum()), len(self.meta))
        self.assertEqual(set(summary["split"]), {"train", "val", "test"})

    def test_mixed_group_is_rejected(self):
        mixed = self.meta.copy()
        mixed.loc[0, "label"] = "None"
        with self.assertRaises(ValueError):
            make_group_split_indices(mixed, random_state=42)


if __name__ == "__main__":
    unittest.main()
