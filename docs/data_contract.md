# Data Contract

This note describes the current channel-level RFI dataset contract.

The contract is based on the existing exploratory notebooks and should be
treated as the baseline to clean up before new scientific conclusions are made.

## Segment

A segment is a short dynamic-spectrum slice with shape:

```text
n_channels x n_time = 256 x 256
```

In the current `B0531+21_59000_48386.fil` experiments:

- one segment is read with `nsamp = 256`;
- each row is one frequency channel;
- each row contains a 256-sample time series.

## Channel Example

One machine-learning example is one frequency channel from one segment.

For raw 1D-CNN experiments, the model input is:

```text
channel_time_series: shape (256,)
```

For statistical-model experiments, the model input is a row of engineered
features derived from the channel and the full segment.

## Labels

Current labels:

- `NBRFI`: channel lies inside the manually selected RFI frequency ranges for a
  segment that belongs to a manually selected RFI segment block.
- `None`: channel comes from a segment where no RFI was identified.
- `NoneWNBRFI`: channel belongs to an RFI-containing segment but lies outside
  the manually selected RFI frequency ranges.

The current supervised baseline trains on:

```text
NBRFI vs None
```

`NoneWNBRFI` should be preserved as hard-negative / stress-test material.

## Full Metadata Columns

The full metadata CSV produced by dataset creation should preserve:

- `sample_index`
- `global_index`
- `segment_index`
- `channel_index`
- `frequency`
- `mean_o`
- `std_o`
- `skew_o`
- `kurt_o`
- `mean_n`
- `std_n`
- `skew_n`
- `kurt_n`
- `mean_o_ratio`
- `std_o_ratio`
- `skew_o_ratio`
- `kurt_o_ratio`
- `mean_n_ratio`
- `std_n_ratio`
- `skew_n_ratio`
- `kurt_n_ratio`
- `label`
- `original_segment_label`

The current subset metadata only contains feature columns and `label`. This is
not enough for robust segment-level audit or group splitting.

## Statistical Features

Raw-channel statistics:

- `mean_o`
- `std_o`
- `skew_o`
- `kurt_o`

Per-channel normalized statistics:

- `mean_n`
- `std_n`
- `skew_n`
- `kurt_n`

Ratios between per-channel and whole-segment values:

- `mean_o_ratio`
- `std_o_ratio`
- `skew_o_ratio`
- `kurt_o_ratio`
- `mean_n_ratio`
- `std_n_ratio`
- `skew_n_ratio`
- `kurt_n_ratio`

## Split Rule

The next cleaned baseline must use segment-level splitting:

```text
all channels from one segment must belong to exactly one split
```

Valid split groups:

- `segment_index`
- or `global_index`, if it is confirmed to represent one segment uniquely.

Do not use a random row-level split for final metrics, because channels from
the same segment are correlated and ratio features explicitly depend on
whole-segment statistics.

## Current Known Issue

The current `split_indices.npz` was produced from a stratified row-level split
inside `notebooks/01_dataset_creation/subdataset_creation.ipynb`.

This is acceptable as an exploratory first pass, but final baseline metrics
should be recomputed after a group-split correction.
