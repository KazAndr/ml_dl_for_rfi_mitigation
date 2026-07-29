# Configs

This directory contains small, versioned experiment specifications. Configs
declare a run; they never contain heavy data or generated diagnostics.

`experiments/experiment_spec_template.json` defines the fields required for a
comparable experiment. It is a template, not a runnable baseline configuration.
The first runnable group-split configurations point to the server-side
corrected artifact directory created by `group_split_subset.ipynb`.

Expected named configurations:

- `legacy_row_split_v1.json` as a frozen historical declaration;
- `b0531_group_split_cnn_zscore_v1.json` for the per-channel z-score CNN;
- `b0531_group_split_cnn_legacy_max_v1.json` for the group-split control that
  retains the legacy row-maximum scaling;
- `b0531_group_split_cnn_zscore_segment_v1.json` for the experiment that
  z-scores each complete physical spectrogram segment before selecting rows;
- `b0531_group_split_statistical_both_v1.json` for the first corrected
  engineered-feature baselines;
- `b0531_cnn_full_heldout_segments_v1.json` for the complete-row evaluation
  of segments held out by the corrected CNN group split; it reuses, rather
  than reselects, the source run's validation threshold;
- corresponding `b0531_cnn_full_heldout_segments_*_v1.json` files for the
  legacy-maximum and per-segment-z-score CNN controls; each uses the matching
  source checkpoint and writes the same visual-review artifact set;
- `b0531_statistical_full_heldout_segments_v1.json` for the analogous CPU
  evaluation of every frozen statistical candidate on the same complete
  held-out segments;
- one configuration per normalization or cleaning variant;
- separate full-file inference declarations.

Config files should contain parameters, not large data.
