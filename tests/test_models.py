"""CPU-only tests for PyTorch profile baseline helpers."""

from __future__ import annotations

import unittest

import numpy as np

from rfimt.training import normalize_profile_rows

try:
    import torch
except ImportError:
    torch = None


class ProfileNormalizationTests(unittest.TestCase):
    def test_none_preserves_profile_values_as_float32(self):
        profiles = np.array([[1, 2, 3]], dtype=np.int64)

        normalized = normalize_profile_rows(profiles, "none")

        self.assertEqual(normalized.dtype, np.float32)
        np.testing.assert_array_equal(normalized, profiles)

    def test_zscore_normalizes_each_profile_independently(self):
        profiles = np.array([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]])

        normalized = normalize_profile_rows(profiles, "zscore_per_channel")

        np.testing.assert_allclose(normalized[0].mean(), 0.0, atol=1e-6)
        np.testing.assert_allclose(normalized[0].std(), 1.0, atol=1e-6)
        np.testing.assert_array_equal(normalized[1], np.zeros(3, dtype=np.float32))

    def test_normalization_rejects_non_row_array(self):
        with self.assertRaisesRegex(ValueError, "2D array"):
            normalize_profile_rows(np.array([1.0, 2.0]), "none")


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CNNArchitectureTests(unittest.TestCase):
    def test_legacy_cnn_produces_one_logit_per_256_sample_profile(self):
        from rfimt.models import CNN1DRFI256Logits

        model = CNN1DRFI256Logits(dropout=0.0)
        output = model(torch.zeros(3, 1, 256))

        self.assertEqual(model.conv3.kernel_size, (10,))
        self.assertEqual(model.fc1.in_features, 256 * 25)
        self.assertEqual(tuple(output.shape), (3,))


if __name__ == "__main__":
    unittest.main()
