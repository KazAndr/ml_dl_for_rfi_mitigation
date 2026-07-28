# Experiment Pipeline Design

## Purpose

The next `rfimt` stage is a notebook-first experiment pipeline for fair
comparison of RFI-cleaning approaches, normalization variants, feature
representations and models. It is not a replacement for exploratory notebooks
or an attempt to turn the research workflow into a CLI-only application.

The unit of comparison is one versioned **experiment specification** plus its
result manifest. A result is not comparable unless its specification records
the input dataset, labels, split, preprocessing, representation, model,
threshold rule and evaluation universe.

## Frozen Legacy Control

The reproduced row-level baseline remains available as `legacy_row_split_v1`.
It is a historical exploratory control only:

- its 1D-CNN result is comparable with the earlier run;
- its MLP result is materially worse than the earlier run and remains an
  unresolved baseline difference;
- neither result may be compared directly with later group-split metrics.

Do not edit the legacy notebooks to make them look like corrected experiments.
New work must record a distinct experiment identifier.

## Experiment Contract

Every run must declare the following fields.

| Area | Required fields | Why it matters |
|---|---|---|
| Identity | `experiment_id`, `purpose`, `status`, `code_revision` | Distinguishes a frozen control, exploratory run and accepted baseline. |
| Dataset | dataset identifier, array/metadata locations, source observation, segment length, selected labels | Prevents silent changes of data universe. |
| Labels | positive class, ordinary negative class, hard-negative handling | Keeps `NoneWNBRFI` separate from ordinary `None`. |
| Split | `legacy_row` or `group_segment`, group column, seed, train/validation/test fractions | Makes leakage status explicit. |
| Preprocessing | cleaning variant and parameters, normalization variant and parameters | Gives cleaning and normalization approaches equal status as experimental axes. |
| Representation | raw channel series or named feature set | Prevents a CNN and feature model from appearing to use the same input by implication. |
| Model | family, named factory entry, hyperparameters, random seed | Supports a fair model comparison. |
| Selection | validation-only threshold rule and grid | Prevents test-set operating-point selection. |
| Evaluation | row/channel, segment, hard-negative and full-file universes | Keeps metrics tied to the population they describe. |
| Outputs | run directory, result manifest, selected model artifact and diagnostic paths | Lets notebooks generate heavy artifacts without committing them. |

The JSON shape is documented in
`configs/experiments/experiment_spec_template.json`. A specification is a
declaration, not an executable command until the corresponding helper path is
implemented.

## Pipeline Layers

### 1. Dataset and Label Layer

Notebook: dataset inspection and controlled regeneration.

Reusable code:

- filterbank/FITS segment loading;
- manual-label expansion;
- metadata validation;
- balanced subset selection from a preselected row universe;
- artifact manifest creation.

The corrected metadata contract must retain `sample_index`, `segment_index`,
`channel_index`, `frequency`, `label` and `original_segment_label` together
with the feature columns. No corrected subset may discard its group column.

### 2. Split Layer

Notebook: inspect group counts, label balance and representative segments.

Reusable code:

- frozen legacy row-split reader;
- corrected class-stratified group split by `segment_index`;
- group-overlap assertion;
- split summary table written into the run manifest.

`legacy_row` and `group_segment` are alternative named modes. They must never
share a result table without an explicit split column.

### 3. Representation and Preprocessing Layer

Notebook: visualize raw channels and the effect of each variant.

Reusable code:

- normalization registry: `minmax_per_channel`, `zscore_per_channel`,
  `zscore_per_segment`, `robust_mad_per_channel` and `none`;
- feature extraction with an explicit source-normalization choice;
- named cleaning adapters with parameters recorded even when the variant is
  `none`.

RFI cleaning is an upstream experimental axis. A future cleaning method must
produce the same documented channel/segment identity or explicitly declare a
different evaluation universe.

### 4. Model, Selection and Evaluation Layer

Notebooks: train within one model family, inspect errors and select diagnostic
cases. CNN and statistical models remain separate runnable paths because they
have different representations, training loops and diagnostic artifacts.

Reusable code:

- sklearn model factories and the 1D-CNN factory;
- validation-only threshold selection;
- row-level metrics, segment aggregation and hard-negative evaluation;
- run-result serialization.

The MLP discrepancy is recorded in `legacy_row_split_v1`; it is not a special
case in the pipeline and should not be tuned until the corrected baseline is
available.

### 5. Full-File and Student Export Layer

Notebook: full-file visual review, masks and scientifically meaningful false
positive/false negative examples.

Reusable code:

- input adapters for filterbank and checked FITS conventions;
- batch inference;
- mask writing;
- diagnostic artifact records.

A student export is a separate output of a named, accepted group-split
experiment. It must contain a small immutable data package, metadata,
manifest, starter notebook and task statement. It must not reuse an arbitrary
research subset or a legacy row-split result.

## Notebook Map

Keep the existing notebooks as historical controls. Add new notebooks only
when their role is stable:

1. `00_reproducibility/legacy_row_split_report.ipynb`: records frozen legacy
   inputs, results and known MLP difference without retraining choices.
2. `01_dataset_creation/group_split_subset.ipynb`: creates and audits a
   corrected subset with canonical metadata.
3. `03_model_training/cnn_experiment_runner.ipynb`: trains one 1D-CNN
   specification and writes a compact result manifest.
4. `03_model_training/statistical_experiment_runner.ipynb`: trains one
   statistical-model specification and writes the same manifest schema.
5. `03_model_training/experiment_comparison.ipynb`: reads manifests only and
   compares like-for-like runs.
6. `04_full_file_tests/experiment_inference.ipynb`: evaluates a selected run
   on a declared full-file universe.
7. `05_student_export/create_student_rfi_package.ipynb`: remains absent until
   a corrected experiment is accepted.

The old `1_cnn_model.ipynb` and `classical_learning_stat.ipynb` stay readable
and runnable as legacy notebooks. Do not refactor them in place during the
first transition.

## Runtime Decision

The active server research environment is
`/.aux_mnt/pc009b/akazantsev/SINGULARITY_IMAGES/pytorch4punch.sif`. It provides
Python `3.8`, PyTorch `1.8.1`, scikit-learn `1.3.1`, NumPy `1.24.3`, pandas
`1.5.1`, SciPy `1.9.3` and the Jupyter/tqdm stack used by the existing
notebooks.

The reusable package therefore supports Python `3.8+` during this refactor.
Run the corrected notebook path in that same container and record the image
path or image digest in the run manifest. Python `3.8` is end-of-life, so a
future container migration should be a separate controlled task, not an
implicit prerequisite that makes legacy and corrected runs incomparable.

## Bounded Implementation Sequence

1. Implement the specification and run-manifest helpers, with no model or
   data behaviour change.
2. Build the corrected metadata/subset and class-stratified group-split
   notebook, then confirm no segment overlap and that both fitting labels are
   represented in every split.
3. Route one statistical baseline and one 1D-CNN baseline independently
   through their model-specific runners, producing the shared result schema.
4. Test `zscore_per_channel`, then optionally `zscore_per_segment`, one
   normalization variant at a time and only on shared evaluation universes.
   `none` and per-channel min-max remain legacy controls, not new comparison
   work.
5. Add cleaning variants and full-file transfer tests on newly formed
   filterbank data after the corrected baseline is stable.
6. Define a student-safe export from an accepted experiment only.

## Acceptance Gate

The first corrected baseline is accepted only when its subset can be rebuilt,
group overlap is zero, its run manifest identifies every experimental axis,
and its 1D-CNN/statistical metrics are labelled as group-split metrics. Only
then may normalization, cleaning or student-data comparisons be treated as
scientific evidence.

## Long Procedures

Every long loop in a notebook must display progress using only:

```python
from tqdm import tqdm
```

Use it for dataset generation, per-segment inference, threshold sweeps with
large grids, repeated resampling and model-search loops. Do not use
`tqdm.notebook`: the server environment supports the standard import
reliably, while notebook-widget rendering is not a pipeline requirement.
