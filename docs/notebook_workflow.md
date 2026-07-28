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

## Progress Reporting

Every long-running notebook procedure must visibly report progress with the
standard server-compatible import:

```python
from tqdm import tqdm
```

Wrap loops over segments, files, parameter combinations, repeated evaluations
and training epochs where the framework does not already provide equivalent
progress. Do not import `tqdm.notebook`; widget rendering is not assumed on
the server.

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

The first immediate notebook task is to reproduce the legacy baseline without
changing its row-level split. This gives the project a control point before
methodological fixes.

After that, the first helper worth extracting into the active workflow is the
group-split and subset-generation logic. It controls the validity of all later
metrics and should become stable before new corrected models are trained.

## Dependencies

The repository package and new reusable notebooks support Python `3.8+`. The
current server standard is the `pytorch4punch.sif` container, built on Python
`3.8`; use that same container for corrected runs until the research
environment is deliberately migrated. Install the project in editable mode
before importing `rfimt` from a notebook:

```bash
python -m pip install -e .
```

Python `3.8` is end-of-life, so a future container upgrade is sensible, but it
is not a prerequisite for the current refactor. Corrected runs must record the
container/kernel and package revision in their run manifest; do not mix an
untracked local kernel with container-derived results.

The lightweight feature and labeling helpers require only `numpy` and `pandas`.

The model, split, and metric helpers require additional scientific packages:

- `scikit-learn` for group splits, statistical models, thresholds, and metrics;
- `torch` for the 1D-CNN class;
- `scipy` for converting sklearn decision scores with `expit`.

These imports are intentionally lazy where possible so notebooks can use data
inspection helpers even in a minimal environment.
