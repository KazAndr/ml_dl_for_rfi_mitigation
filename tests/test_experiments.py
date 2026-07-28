"""Tests for experiment specification helpers."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rfimt.experiments import (
    comparison_fingerprint,
    load_experiment_spec,
    make_run_manifest,
    write_run_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs" / "experiments" / "experiment_spec_template.json"


class ExperimentSpecTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_experiment_spec(TEMPLATE)

    def test_output_path_does_not_change_comparison_fingerprint(self):
        first = comparison_fingerprint(self.spec)
        changed = json.loads(json.dumps(self.spec))
        changed["outputs"]["run_directory"] = "another-local-path"
        self.assertEqual(first, comparison_fingerprint(changed))

    def test_invalid_split_mode_is_rejected(self):
        invalid = json.loads(json.dumps(self.spec))
        invalid["split"]["mode"] = "row_and_group"
        with self.assertRaises(ValueError):
            comparison_fingerprint(invalid)

    def test_manifest_is_written_as_json(self):
        manifest = make_run_manifest(
            self.spec,
            metrics={"row_test": {"f1": 0.5}},
            code_revision="abc123",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_run_manifest(Path(directory) / "manifest.json", manifest)
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["experiment_id"], self.spec["experiment_id"])
        self.assertEqual(loaded["metrics"]["row_test"]["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()
