# Artifact Manifest

This manifest records the main local artifacts present in the exploratory
`rfimt` directory.

Large artifacts should not be committed to git. They should be moved between
machines with `rsync` or regenerated from notebooks/scripts.

## Core Data

| Artifact | Approx. size | Role | Track in git |
|---|---:|---|---|
| `B0531+21_59000_48386_channels.npy` | `6.6G` | Full channel-level dataset generated from the filterbank file. | no |
| `B0531+21_59000_48386_channels_meta.csv` | `2.3G` | Full metadata and feature table for the channel dataset. | no |
| `B0531+21_59000_48386_subset_channels.npy` | `20M` | Balanced subset used in current experiments. | no |
| `B0531+21_59000_48386_subset_channels_meta.csv` | `5.8M` | Current subset metadata. Missing segment-level audit columns. | no |
| `B0531+21_59000_48386_subset_indices.npy` | `160K` | Row indices used to create the current subset. | no |
| `split_indices.npz` | `160K` | Current train/val/test row split. Needs group-split replacement. | no |
| `subset_y.npy` | `160K` | Binary labels for the current subset. | no |

## Model Artifacts

| Artifact | Role | Track in git |
|---|---|---|
| `best_model_by_acc.pt` | PyTorch 1D-CNN checkpoint from exploratory training. | no |
| `saved_models_selected_topk/` | Selected scikit-learn models after feature selection. | no |
| `saved_models_stat/` | Earlier saved statistical models. | no |
| `saved_models_stat_no_scaler/` | Saved statistical models from no-scaler experiments. | no |

Selected models may later be published as explicit release artifacts, but they
should not be committed casually.

## Diagnostic Figures

Large diagnostic directories:

- `1_cnn_individual_key_metrics_segment_nbrf_accuracy_leq_0.95/`
- `1_cnn_individual_key_metrics_segment_nbrf_precision_leq_0.8/`
- `1_cnn_individual_key_metrics_segment_nbrf_recall_leq_0.95/`
- `stat_individual_key_metrics_segment_nbrf_accuracy_leq_0.9/`
- `stat_individual_key_metrics_segment_nbrf_precision_leq_0.5/`
- `stat_individual_key_metrics_segment_nbrf_recall_leq_0.9/`

These directories are useful for visual audit, but they are generated outputs
and should remain outside git history.

## Source Notebooks

Notebooks that define the current experiment state:

- `notebooks/01_dataset_creation/creating_dataset_from_filterbank_by_indexes.ipynb`
- `notebooks/01_dataset_creation/subdataset_creation.ipynb`
- `notebooks/02_feature_exploration/stat_analysis.ipynb`
- `notebooks/03_model_training/1_cnn_model.ipynb`
- `notebooks/03_model_training/classical_learning_stat.ipynb`
- `notebooks/04_full_file_tests/1d_cnn_global_test_rfi_cleaning_real_test.ipynb`
- `notebooks/04_full_file_tests/mlp_global_test_rfi_cleaning_real_test.ipynb`

The `.py` files currently mirror some notebooks and should be treated as
exports in `legacy_exports/` until a clean reusable source package is created.
