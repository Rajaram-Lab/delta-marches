# Data and Path Configuration

This directory contains `data_dir.yaml` and a `datasets/` placeholder for downloaded and prepared data. See the [data preparation guide](../code/README_Preparation.md) for dataset sources and preprocessing instructions.

Update [data_dir.yaml](./data_dir.yaml) to match your local paths for datasets, model weights, annotations, and generated analysis results. Use absolute paths or paths relative to the repository root, and run the corresponding workflows from that root.

Obtain or generate the files required by your workflow before running it. Update any script-specific paths separately, including DIFFAE checkpoint paths.
