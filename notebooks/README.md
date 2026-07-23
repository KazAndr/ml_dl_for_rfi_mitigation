# Notebook Workflow

The project is notebook-first. Notebooks are grouped by analysis stage rather
than by implementation layer.

Run Jupyter from the repository root:

```bash
cd /.aux_mnt/pc009b/akazantsev/my_development/rfimt
jupyter lab
```

Many current notebooks use relative paths to local data artifacts in the
repository root, for example:

- `B0531+21_59000_48386_channels.npy`
- `B0531+21_59000_48386_channels_meta.csv`
- `B0531+21_59000_48386_subset_channels.npy`
- `B0531+21_59000_48386_subset_channels_meta.csv`
- `split_indices.npz`

If a notebook is executed with its own directory as the working directory, these
relative paths may fail. A later cleanup should replace implicit relative paths
with an explicit project-root helper.

## Stages

1. `01_dataset_creation/`
   - Build the channel-level dataset.
   - Build a smaller subset for fast experiments.
   - Next required fix: group split by `segment_index`.

2. `02_feature_exploration/`
   - Inspect engineered statistics.
   - Use this stage to decide whether new normalization or feature variants are
     worth testing.

3. `03_model_training/`
   - Train the 1D-CNN baseline on channel time series.
   - Train statistical and tabular baselines on engineered features.
   - Re-run only after the group-split correction.

4. `04_full_file_tests/`
   - Apply trained models to larger files.
   - Compare masks and before/after behavior.
   - This stage belongs after the baseline model contract is stable.

## Notebook Policy

- Keep exploratory analysis in notebooks.
- Move only stable reusable helpers into `src/rfimt/`.
- Do not commit large generated outputs.
- When a notebook produces a large artifact, document it in
  `docs/artifact_manifest.md`.
- When a notebook changes the dataset or split contract, update
  `docs/data_contract.md`.
