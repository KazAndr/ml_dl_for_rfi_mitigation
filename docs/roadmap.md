# Roadmap

This roadmap separates the old exploratory baseline, the PhD continuation, and
the future student-facing work.

## Track A: Legacy Baseline Reproduction

Goal: reproduce the current `B0531+21_59000_48386` row-split experiment before
changing methodology. This creates a historical control point for later
group-split, z-score, and student-facing work.

Steps:

1. Preserve the current exploratory notebooks as historical working notebooks.
2. Check that the required local artifacts for the old run are present.
3. Re-run the legacy notebooks without changing the row-level split, labels,
   normalization, or model definitions.
4. Record:
   - subset shape and label counts;
   - train, validation, and test sizes;
   - 1D-CNN metrics;
   - statistical-model comparison;
   - selected top-k feature models;
   - segment-level checks and qualitative diagnostics.
5. Document any failure with the failing notebook, cell purpose, and required
   next action.

Done criteria:

- the old baseline either runs again or has a precise failure report;
- row-split results are clearly labeled as legacy exploratory results;
- required heavy artifacts are listed without being committed;
- the next corrected-baseline step has a stable starting point.

See `docs/legacy_baseline_reproduction_plan.md`.

## Track A2: Corrected Reproducible Baseline

Goal: make the baseline methodologically cleaner after the legacy run is
recoverable or its failures are understood.

Steps:

1. Fix the data contract using `src/rfimt/features.py`, `src/rfimt/labels.py`,
   and `src/rfimt/splits.py`:
   - preserve `sample_index`, `segment_index`, `channel_index`, `frequency`,
     `label`, and `original_segment_label` in subset metadata;
   - define one canonical metadata schema;
   - define one canonical label map.
2. Replace the legacy row-level split with a group split by `segment_index`.
3. Regenerate the balanced subset using the corrected metadata contract.
4. Re-run:
   - 1D-CNN baseline;
   - statistical-feature baselines;
   - selected top-k feature models.
5. Recompute segment-level metrics under the corrected split.
6. Compare corrected group-split results with the legacy row-split control
   point.

Done criteria:

- one command/notebook path can rebuild the subset;
- train/validation/test groups do not share segments;
- baseline metrics are recomputed and clearly labeled as group-split metrics;
- `NoneWNBRFI` is evaluated as hard-negative material.

## Track B: Z-Score And Transfer Tests

Goal: test whether normalization explains poor transfer to other observations
or file formats.

Normalization variants to compare:

- current per-channel min-max style normalization;
- per-channel z-score;
- per-segment z-score;
- robust median/MAD variant, if ordinary z-score is unstable;
- no normalization for statistical models where raw scale may be meaningful.

Evaluation targets:

- original `B0531+21_59000_48386` split after group-split correction;
- unseen `B0531+21_60482_57794` or related filterbank data;
- B0355+54 / FRB20240114A-like FITS data only after FITS offset handling is
  checked.

Done criteria:

- z-score variants are defined mathematically and in code;
- each variant has a config;
- each result records model, normalization, threshold, split, and evaluation
  universe;
- transfer failures are separated into normalization problems, file-format
  problems, and model/domain-shift problems.

## Track C: FITS Offset Recheck

Goal: decide whether FITS subintegration offsets caused models to mark all
channels as contaminated for B0355+54 and FRB20240114A-like data.

Steps:

1. Isolate the FITS reading and writing logic from the current full-file
   notebooks.
2. Inspect raw array values before and after offset/scaling application.
3. Compare one small set of segments:
   - raw loaded data;
   - offset-corrected data;
   - normalized data;
   - model mask.
4. Re-run one small inference sample after correction.

Done criteria:

- the FITS loading convention is documented;
- the failure mode is classified as offset-related, normalization-related, or
  model/domain-shift-related.

## Track D: Future Student Task

Goal: later create a clean, bounded student task around RFI-channel anomaly
detection.

This should not start until Track A has a reproduced legacy control point and
Track A2 has a stable data contract.

Likely student framing:

- normal class: channels that look noise-like after normalization;
- anomaly: channel behavior that deviates from the nominal/noise-like class;
- RFI is the main scientific example of anomaly, but the model should be
  introduced as anomaly scoring rather than only supervised classification.

Candidate student materials:

- small metadata CSV;
- small array file or CSV table;
- short notebook template;
- task statement;
- expected plots:
  - example normal channels;
  - example RFI channels;
  - anomaly-score distribution;
  - threshold sweep;
  - false-positive examples.

Do not mix this student dataset with the PhD baseline until the research
contract is stable.

## Track E: Repository Hygiene

Goal: keep the repository usable without hiding the exploratory nature of the
work.

Near-term structure:

- notebooks remain the main analysis interface;
- `src/rfimt/` receives only stable helpers;
- configs describe repeatable runs;
- docs describe data contracts, artifacts, and decisions.

Do not commit:

- raw observations;
- large arrays;
- generated diagnostic image directories;
- model checkpoints unless intentionally released.
