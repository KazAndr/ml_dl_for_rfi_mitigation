# Reproducible performance comparison

This directory leaves the legacy notebooks untouched. It separates the
    expensive and logically different steps required for the RFI-mitigation
    performance figure.

    1. Run `01_train_1d_cnn_cpu_gpu.ipynb` with `RUN_DEVICE = "cpu"`.
    2. Run it again with `RUN_DEVICE = "cuda"`.
    3. Run `02_train_mlp_orig_top3.ipynb`.
    4. Run `03_benchmark_model_inference.ipynb` on a CUDA-capable node.
    5. Run `04_plot_report_performance.ipynb` to export the final PNG and PDF.
    6. Run `05_plot_learning_curves.ipynb` to inspect and export learning curves
       already available from the CNN and legacy sklearn MLP runs.
    7. Run `06_train_mlp_pytorch_orig_top3.ipynb` to train the separate
       PyTorch MLP and save epoch-level loss and accuracy history.
    8. Run `07_compare_mlp_pytorch_vs_legacy.ipynb` to compare the PyTorch
       checkpoint with `saved_models_selected_topk/rfi_model_MLP_orig_top3.joblib`
       on complete physical segments excluded from the sampled subset.

`03_benchmark_model_inference.ipynb` reads the immutable training artifacts
from `b0531_legacy_performance_v1`. Its results are written separately to
`benchmark_runs/b0531_full_segments_model_only`; the plotting notebook reads
the same directory. If a scientifically distinct repetition is needed, create
a new explicitly named sibling directory rather than overwrite this one.

The notebooks are self-contained and therefore do not require a local checkout
of the `rfimt` Python package. They use the current Jupyter working directory
as their working root. The output root is
`outputs/performance_comparison/<RUN_TAG>/`. It is ignored
by Git and contains checkpoints, the fitted MLP bundle, learning history, raw
timing samples, timing summaries, the exact protocol, and the final figure.
No notebook overwrites an existing result directory.

The MLP is the legacy selected `MLP_orig_top3` pipeline with the three recorded
features `mean_o`, `std_o`, and `skew_o`. The CNN uses
an embedded `CNN1DRFI256Logits` definition, preserving the legacy architecture
and right-padding behaviour.

The PyTorch MLP is a separately recorded implementation of the same input
layout and hidden-layer sizes. It is not assumed to reproduce the legacy
sklearn weights exactly: it records its own validation-selected threshold and
must be compared with the saved sklearn model before either result is used as a
replacement for the legacy result.

For the report figure, inference is deliberately `model_only`: it includes the
trained model and its final decision, but excludes file I/O, feature
calculation, maximum normalization, and CPU-to-GPU transfer. This matches the
requested comparison of the ML methods themselves. Raw data and the protocol
are retained so that a broader pipeline-level timing can be added later
without retraining the models.
