"""Experiment specifications and compact result manifests for ``rfimt``.

The helpers deliberately use JSON and the Python standard library. Notebooks
remain responsible for executing research runs; this module records enough
context to compare their outcomes without guessing which preprocessing or split
was used.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


EXPERIMENT_SPEC_VERSION = 1
VALID_SPLIT_MODES = {"legacy_row", "group_segment"}
VALID_NORMALIZATION_VARIANTS = {
    "none",
    "minmax_per_channel",
    "zscore_per_channel",
    "zscore_per_segment",
    "robust_mad_per_channel",
}

_REQUIRED_TOP_LEVEL_FIELDS = (
    "experiment_id",
    "purpose",
    "status",
    "dataset",
    "labels",
    "split",
    "preprocessing",
    "representation",
    "model",
    "selection",
    "evaluation",
    "outputs",
)


def load_experiment_spec(path: str | Path) -> Dict[str, Any]:
    """Load and validate one JSON experiment specification."""
    with Path(path).open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    return validate_experiment_spec(spec)


def validate_experiment_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the minimum comparison contract and return a detached copy.

    Validation is intentionally structural. Domain-specific constraints, such
    as whether a chosen array actually has a given metadata column, belong in
    dataset-building notebooks and their later reusable helpers.
    """
    if not isinstance(spec, Mapping):
        raise ValueError("Experiment specification must be a mapping.")

    missing = [field for field in _REQUIRED_TOP_LEVEL_FIELDS if field not in spec]
    if missing:
        raise ValueError(f"Experiment specification is missing fields: {missing}")

    value = deepcopy(dict(spec))
    if not isinstance(value["experiment_id"], str) or not value["experiment_id"].strip():
        raise ValueError("experiment_id must be a non-empty string.")

    for section in (
        "dataset",
        "labels",
        "split",
        "preprocessing",
        "representation",
        "model",
        "selection",
        "evaluation",
        "outputs",
    ):
        if not isinstance(value[section], Mapping):
            raise ValueError(f"{section} must be a mapping.")

    split_mode = value["split"].get("mode")
    if split_mode not in VALID_SPLIT_MODES:
        raise ValueError(
            f"split.mode must be one of {sorted(VALID_SPLIT_MODES)}, got {split_mode!r}."
        )

    normalization = value["preprocessing"].get("normalization_variant")
    if normalization not in VALID_NORMALIZATION_VARIANTS:
        raise ValueError(
            "preprocessing.normalization_variant must be one of "
            f"{sorted(VALID_NORMALIZATION_VARIANTS)}, got {normalization!r}."
        )

    universes = value["evaluation"].get("universes")
    if not isinstance(universes, Sequence) or isinstance(universes, (str, bytes)):
        raise ValueError("evaluation.universes must be a sequence of named universes.")
    if not universes:
        raise ValueError("evaluation.universes must not be empty.")

    return value


def comparison_fingerprint(spec: Mapping[str, Any]) -> str:
    """Return a stable fingerprint of scientifically relevant run settings.

    The output directory is deliberately excluded because changing a local path
    must not make two otherwise identical experiments incomparable.
    """
    validated = validate_experiment_spec(spec)
    comparison_spec = deepcopy(validated)
    comparison_spec.pop("outputs", None)
    comparison_spec.pop("code_revision", None)
    payload = json.dumps(comparison_spec, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()[:16]


def make_run_manifest(
    spec: Mapping[str, Any],
    metrics: Optional[Mapping[str, Any]] = None,
    code_revision: Optional[str] = None,
    artifacts: Optional[Mapping[str, str]] = None,
    notes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build a compact JSON-serializable run record.

    ``metrics`` are supplied by the notebook after it evaluates the declared
    universes. The helper does not calculate metrics or select a threshold.
    """
    validated = validate_experiment_spec(spec)
    return {
        "manifest_version": EXPERIMENT_SPEC_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": validated["experiment_id"],
        "comparison_fingerprint": comparison_fingerprint(validated),
        "code_revision": code_revision or validated.get("code_revision"),
        "spec": validated,
        "metrics": dict(metrics or {}),
        "artifacts": dict(artifacts or {}),
        "notes": list(notes or []),
    }


def write_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Write one human-readable JSON manifest and return its path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination
