# ML/DL for RFI Mitigation

This directory contains exploratory notebooks and intermediate artifacts for
channel-level radio-frequency-interference mitigation experiments.

The work started from the idea that narrowband RFI can be treated as a
one-dimensional channel-classification problem: each frequency channel is
represented by its short time series, and the model decides whether that
channel is contaminated by RFI.

The current state is intentionally notebook-first. The notebooks are the main
place for data inspection, model comparison, qualitative checks, and figure
generation. Future refactoring should make reusable code available to the
notebooks without replacing the notebook-based research workflow.

## Scientific Scope

The current experiments focus on narrowband RFI (`NBRFI`) in filterbank-like
data. The first object used for dataset creation is:

- `B0531+21_59000_48386.fil`

The practical motivation is to test whether a channel-level model can support
the broader PhD RFI-mitigation work and later help MICA-like single-pulse
pipelines by removing or flagging narrowband contamination that is not well
represented in DM-time images.

## Main Classes

The current labeling scheme has three relevant classes:

- `NBRFI`: frequency channels inside manually selected narrowband-RFI ranges.
- `None`: channels from segments where no RFI was identified.
- `NoneWNBRFI`: channels inside RFI-containing segments but outside the
  manually selected RFI frequency ranges.

At the current baseline stage, models are trained mainly on `NBRFI` versus
`None`. `NoneWNBRFI` is better treated as a hard-negative or stress-test class,
not as an ordinary training negative.

## Important Methodological Note

The current notebooks are exploratory and should not yet be treated as a fully
reproducible pipeline.

One important issue to fix before drawing final quantitative conclusions:

- `notebooks/01_dataset_creation/subdataset_creation.ipynb` currently creates
  `split_indices.npz` with a stratified row-level split.
- For channel-level data, channels from the same segment are correlated.
- The next cleaned version should use a group split by `segment_index`, so that
  all channels from one segment belong to only one of `train`, `validation`, or
  `test`.

This is especially important for statistical features that use ratios between
per-channel quantities and whole-segment quantities.

## Repository Layout

```text
.
|-- README.md
|-- configs/
|   `-- README.md
|-- docs/
|   |-- artifact_manifest.md
|   |-- data_contract.md
|   |-- notebook_workflow.md
|   `-- roadmap.md
|-- legacy_exports/
|   `-- old notebook-to-Python exports
|-- notebooks/
|   |-- 01_dataset_creation/
|   |-- 02_feature_exploration/
|   |-- 03_model_training/
|   `-- 04_full_file_tests/
`-- src/
    `-- rfimt/
        |-- constants.py
        |-- features.py
        |-- io.py
        |-- labels.py
        |-- metrics.py
        |-- models.py
        `-- splits.py
```

The repository is organized around notebooks because the project is still in a
research-analysis stage. `src/rfimt/` contains stable helpers that are useful
across notebooks, but notebooks remain the primary analysis interface.

Install the local helper package in editable mode when you want notebooks to
import `rfimt` without setting `PYTHONPATH` manually:

```bash
pip install -e .
```

## Notebook Map

### Dataset Creation

- `notebooks/01_dataset_creation/creating_dataset_from_filterbank_by_indexes.ipynb`
  - Reads selected segments from `B0531+21_59000_48386.fil`.
  - Uses manually defined RFI segment ranges and frequency-channel ranges.
  - Creates channel-level examples.
  - Computes raw and normalized per-channel statistics.
  - Writes the large channel dataset and metadata.

- `notebooks/01_dataset_creation/subdataset_creation.ipynb`
  - Creates a smaller balanced subset for faster experiments.
  - Current baseline subset: `10000` `NBRFI` examples and `10000` `None`
    examples.
  - Needs a group-split update before final metrics are trusted.

### Exploratory Statistics

- `notebooks/02_feature_exploration/stat_analysis.ipynb`
  - Visual inspection of statistical features.
  - Useful for understanding whether simple feature projections separate
    classes.

### Model Training

- `notebooks/03_model_training/1_cnn_model.ipynb`
  - PyTorch 1D-CNN baseline on raw 256-sample channel time series.
  - Tests the Rishi Kumar-inspired channel-profile approach with input length
    adapted from `512` to `256`.
  - Includes segment-level visual checks and timing measurements.

- `notebooks/03_model_training/classical_learning_stat.ipynb`
  - Scikit-learn models trained on engineered statistical features.
  - Compares linear and nonlinear baselines:
    - SGD logistic regression;
    - SGD linear SVM;
    - polynomial degree-2 logistic regression;
    - RBF approximation plus logistic regression;
    - histogram gradient boosting;
    - MLP.
  - Performs validation-threshold selection and feature-importance checks.
  - Saves selected top-k feature models.

### Full-File / Real-Data Tests

- `notebooks/04_full_file_tests/1d_cnn_global_test_rfi_cleaning_real_test.ipynb`
  - Applies a trained 1D-CNN model to larger data products.
  - Contains FITS/PSRFITS-oriented masking helpers.

- `notebooks/04_full_file_tests/mlp_global_test_rfi_cleaning_real_test.ipynb`
  - Applies selected scikit-learn/statistical models to larger data products.
  - Contains masking, segment writing, and diagnostic plotting helpers.

## Data And Artifact Policy

The repository should track source code, notebooks, small metadata examples,
configuration files, and documentation.

The repository should not track large data or generated diagnostics:

- raw filterbank / FITS / PSRFITS / archive files;
- large `.npy` arrays;
- full metadata CSV files with millions of rows;
- generated diagnostic PNG directories;
- temporary Jupyter checkpoints;
- local caches and logs.

Heavy files should be moved between machines with `rsync` or another external
data-transfer mechanism. When a heavy artifact is scientifically important,
record its filename, origin, shape, size, and generation notebook in a small
manifest rather than committing the artifact itself.

## Current Large Artifacts

Known large local artifacts include:

- `B0531+21_59000_48386_channels.npy`
  - Full channel dataset.
  - Approximate local size: `6.6G`.

- `B0531+21_59000_48386_channels_meta.csv`
  - Full channel metadata table.
  - Approximate local size: `2.3G`.

- `B0531+21_59000_48386_subset_channels.npy`
  - Balanced subset used by current notebooks.
  - Approximate local size: `20M`.

- `B0531+21_59000_48386_subset_channels_meta.csv`
  - Metadata for the current balanced subset.
  - Approximate local size: `5.8M`.

- diagnostic directories such as
  `1_cnn_individual_key_metrics_segment_nbrf_*` and
  `stat_individual_key_metrics_segment_nbrf_*`.

## Recommended Near-Term Cleanup

The next cleanup should preserve the notebook-first workflow while making the
analysis easier to restart:

1. Replace the current row-level split with a segment-level group split.
2. Re-run the baseline 1D-CNN and statistical models after the split fix.
3. Add config files for dataset creation, baseline training, z-score tests, and
   full-file inference.
4. Extract stable helper functions into `src/rfimt/`, while keeping notebooks
   as the main research interface.
5. Only after the baseline is reproducible, revisit z-score normalization and
   FITS offset handling.
6. Prepare a separate student-facing task and dataset only after the research
   data contract is stable.

The current working plan is tracked in `docs/roadmap.md`.

## Relation To PhD Planning

This work supports the RFI-mitigation follow-ups in the PhD planning system:

- `rfi_filterbank_test`: test the method directly on filterbank files.
- `rfi_zscore_normalization_test`: check whether z-score style normalization
  improves transfer to other observations.
- `rfi_fits_offset_recheck`: check whether FITS subintegration offsets caused
  failure on B0355+54 and FRB20240114A-like data.

## Git Policy

Before committing, check that large data products are ignored:

```bash
git status --short
```

Expected tracked file types:

- `.md`
- `.ipynb`
- `.py`
- small `.yaml` / `.json` configs
- small manifest files

Expected untracked/ignored file types:

- `.npy`, `.npz`
- `.fil`, `.fits`, `.ar`
- large `.csv`
- generated `.png`
- `.ipynb_checkpoints/`
- model checkpoints unless explicitly selected for release
