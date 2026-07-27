# Reproducibility Notebooks

This directory is reserved for audit notebooks that check whether earlier
experiments can still be reproduced after repository cleanup.

The first planned notebook is:

- `legacy_baseline_audit.ipynb`

Its purpose will be to inspect the existing artifacts and rerun the old
baseline in a controlled way. It should not change the legacy split, labels,
normalization, or model definitions. The old row-level split is kept here only
as a historical control point before the corrected group-split baseline is
introduced.

Use this directory for compact audit notebooks, not for broad exploratory
analysis. Exploratory work should remain in the stage-specific directories.
