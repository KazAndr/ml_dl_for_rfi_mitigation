# Configs

This directory contains small, versioned experiment specifications. Configs
declare a run; they never contain heavy data or generated diagnostics.

`experiments/experiment_spec_template.json` defines the fields required for a
comparable experiment. It is a template, not a runnable baseline configuration.
The first executable group-split configuration will be added only after the
corrected subset and its metadata contract exist.

Expected named configurations:

- `legacy_row_split_v1.json` as a frozen historical declaration;
- `b0531_group_split_baseline_v1.json` after corrected subset creation;
- one configuration per normalization or cleaning variant;
- separate full-file inference declarations.

Config files should contain parameters, not large data.
