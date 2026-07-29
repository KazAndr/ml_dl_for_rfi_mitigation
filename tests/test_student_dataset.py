"""Tests for the student-facing RFI package helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from rfimt.student_dataset import (
    assert_disjoint_split_groups,
    assign_split_column,
    map_source_labels_to_student,
    select_challenge_segments,
    stack_complete_segments,
)


class StudentDatasetTests(unittest.TestCase):
    def setUp(self):
        self.core = pd.DataFrame(
            {
                "segment_index": [1, 1, 2, 2, 3, 3],
                "channel_index": [0, 1, 0, 1, 0, 1],
                "label": ["None", "None", "NBRFI", "NBRFI", "None", "None"],
            }
        )

    def test_core_split_assignment_rejects_overlap_and_preserves_groups(self):
        assigned = assign_split_column(
            self.core,
            {"train": np.array([0, 1]), "val": np.array([2, 3]), "test": np.array([4, 5])},
        )

        assert_disjoint_split_groups(assigned)
        self.assertEqual(set(assigned["split"]), {"train", "val", "test"})
        with self.assertRaisesRegex(ValueError, "more than one"):
            assign_split_column(
                self.core,
                {"train": np.array([0, 1]), "val": np.array([1, 2]), "test": np.array([3, 4, 5])},
            )

    def test_challenge_selection_excludes_core_segments(self):
        full = pd.DataFrame(
            {
                "segment_index": [1, 1, 2, 2, 3, 3, 4, 4],
                "label": ["", "", "None", "NoneWNBRFI", "NoneWNBRFI", "NBRFI", "", ""],
            }
        )

        selected = select_challenge_segments(full, excluded_segments=[1], n_clean=1, n_hard=1, random_state=7)

        self.assertNotIn(1, set(selected["segment_index"]))
        self.assertEqual(set(selected["challenge_kind"]), {"clean", "rfi_containing"})

    def test_student_labels_rename_none_to_background_noise(self):
        labels = map_source_labels_to_student(pd.Series(["None", "NBRFI"]))

        self.assertEqual(labels.tolist(), ["BGN", "NBRFI"])

    def test_complete_segment_stack_keeps_channel_order(self):
        full = pd.DataFrame(
            {
                "segment_index": [10, 10, 11, 11],
                "channel_index": [1, 0, 1, 0],
                "source_row_index": [0, 1, 2, 3],
            }
        )
        rows = np.array([[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42]])

        spectra = stack_complete_segments(rows, full, [10, 11], expected_channels=2, expected_time=3)

        np.testing.assert_array_equal(spectra[0], rows[[1, 0]])
        np.testing.assert_array_equal(spectra[1], rows[[3, 2]])


if __name__ == "__main__":
    unittest.main()
