# rfimt Source Package

This package is intentionally minimal for now.

Only stable helpers that are reused across notebooks should be moved here.

Current modules:

- `constants.py`: B0531+21 baseline constants, feature lists, manual RFI block definitions.
- `io.py`: filterbank segment reading and metadata loading helpers.
- `features.py`: statistics, min-max normalization, z-score variants, MAD normalization, channel-feature extraction.
- `labels.py`: manual RFI block lookup and channel-label construction.
- `splits.py`: group split and balanced sampling helpers.
- `metrics.py`: threshold selection, confusion counts, binary metrics.
- `models.py`: sklearn baseline factory and optional PyTorch 1D-CNN class.

Exploratory code should stay in notebooks until its contract is clear.
