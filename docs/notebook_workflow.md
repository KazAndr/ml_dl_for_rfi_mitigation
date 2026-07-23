# Notebook-First Research Workflow

This project should remain notebook-first while the scientific questions are
still being refined.

The intended workflow is:

1. Explore data in notebooks.
2. When a helper function is reused in several notebooks, move it to
   `src/rfimt/`.
3. When a notebook run becomes repeatable, describe its parameters in
   `configs/`.
4. When a run produces large artifacts, keep the artifacts outside git and
   describe them in `docs/artifact_manifest.md`.
5. When the meaning of rows, labels, splits, or metrics changes, update
   `docs/data_contract.md`.

## What Belongs In Notebooks

- visual inspection;
- exploratory plots;
- qualitative examples;
- first-pass model comparison;
- debugging of new normalization ideas;
- side-by-side checks of real data and model masks.

## What Belongs In `src/rfimt/`

- stable data loading helpers;
- feature extraction functions;
- normalization functions;
- split builders;
- metric functions;
- plotting helpers used by multiple notebooks.

## What Belongs In `configs/`

- dataset-generation parameters;
- selected input file identifiers;
- RFI block definitions, once stable;
- split parameters;
- normalization variants;
- training parameters;
- inference thresholds.

## First Refactoring Target

The first helper worth extracting is the group-split and subset-generation
logic. It controls the validity of all later metrics and should become stable
before new models are trained.

## Dependencies

The lightweight feature and labeling helpers require only `numpy` and `pandas`.

The model, split, and metric helpers require additional scientific packages:

- `scikit-learn` for group splits, statistical models, thresholds, and metrics;
- `torch` for the 1D-CNN class;
- `scipy` for converting sklearn decision scores with `expit`.

These imports are intentionally lazy where possible so notebooks can use data
inspection helpers even in a minimal environment.
