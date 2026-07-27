# Legacy Baseline Reproduction Plan

This document defines the next safe step for the RFI-mitigation project:
reproduce the old exploratory baseline before changing the split logic,
normalization, data contract, or student-facing task.

The goal is not to defend the old methodology as final. The goal is to create
a stable reference point. Once the old 1D-CNN and statistical-model results are
recovered or their failures are documented, later changes can be compared
against something concrete.

## Main Rule

Do not change the legacy experiment while reproducing it.

During this phase, keep the following choices fixed:

- row-level stratified split from `split_indices.npz`;
- `NBRFI` as the positive class;
- `None` as the ordinary negative class;
- `NoneWNBRFI` as hard-negative or stress-test material, not as a normal
  training negative;
- current `10000` `NBRFI` + `10000` `None` balanced subset;
- current per-channel min-max style normalization;
- current 1D-CNN architecture and statistical-feature definitions;
- current validation-threshold selection where the old notebooks use it.

The row-level split is a known methodological weakness for channel-level data,
because channels from the same segment are correlated. It is kept here only to
reproduce the legacy control point.

## Required Local Artifacts

The following files are expected to exist locally but should not be committed
to git:

- `B0531+21_59000_48386.fil`;
- `B0531+21_59000_48386_channels.npy`;
- `B0531+21_59000_48386_channels_meta.csv`;
- `B0531+21_59000_48386_subset_channels.npy`;
- `B0531+21_59000_48386_subset_channels_meta.csv`;
- `B0531+21_59000_48386_subset_indices.npy`;
- `split_indices.npz`;
- `subset_y.npy`;
- `best_model_by_acc.pt`;
- `segments_index_B0531+21_59000_48386.csv`;
- `saved_models_selected_topk/`, if statistical top-k models are evaluated;
- `saved_models_stat/` or `saved_models_stat_no_scaler/`, if older statistical
  model artifacts are needed.

If an artifact is missing, record the missing filename and the notebook that is
expected to regenerate it. Do not silently replace it with a new artifact that
changes the experiment definition.

## Notebook Order

Run notebooks from the repository root so relative paths resolve as expected:

```bash
cd /.aux_mnt/pc009b/akazantsev/my_development/rfimt
jupyter lab
```

Recommended reproduction order:

1. `notebooks/01_dataset_creation/creating_dataset_from_filterbank_by_indexes.ipynb`
2. `notebooks/01_dataset_creation/subdataset_creation.ipynb`
3. `notebooks/02_feature_exploration/stat_analysis.ipynb`
4. `notebooks/03_model_training/1_cnn_model.ipynb`
5. `notebooks/03_model_training/classical_learning_stat.ipynb`
6. `notebooks/04_full_file_tests/1d_cnn_global_test_rfi_cleaning_real_test.ipynb`
7. `notebooks/04_full_file_tests/mlp_global_test_rfi_cleaning_real_test.ipynb`

The first pass should answer a simple question: can the old analysis still be
run from the organized repository layout?

## Values To Record

Record the following values in a short reproduction note or notebook output:

- full dataset shape;
- subset shape;
- label counts for `NBRFI`, `None`, and `NoneWNBRFI` where available;
- train, validation, and test sizes from `split_indices.npz`;
- 1D-CNN train, validation, and test metrics;
- selected 1D-CNN threshold, if it is not fixed at `0.5`;
- statistical-model comparison table;
- selected top-k feature models;
- segment-level metrics for clean and RFI-containing segments;
- representative qualitative plots or paths to diagnostic directories.

For each metric, state the evaluation universe: row/channel-level subset,
segment-level check, clean segment, RFI-containing segment, or hard-negative
material.

## Audit Checklist

Before changing methodology, check:

- required local artifacts exist;
- notebooks still find their inputs after the directory reorganization;
- the Python environment has the old dependencies:
  `numpy`, `pandas`, `matplotlib`, `torch`, `scikit-learn`, `scipy`, `astropy`,
  `tqdm`, and `your`;
- every notebook is run from the repository root or uses an explicit root path;
- generated large files and diagnostic images remain ignored by git;
- any failure is recorded with the failing notebook, cell purpose, error type,
  and required next action.

## Decision Gate

Only after this legacy baseline is reproduced, or after its failure modes are
explicitly documented, proceed to the corrected methodology:

- group split by `segment_index`;
- regenerated subset with canonical metadata;
- rerun 1D-CNN and statistical baselines;
- compare legacy row-split results with corrected group-split results;
- then test z-score and robust normalization variants.

This order keeps the research history interpretable: first recover what was
done, then improve the methodology.
