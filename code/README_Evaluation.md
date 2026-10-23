# AAE Training and Latent-Space Analysis for Histopathology

This directory contains scripts for training an adversarial autoencoder (AAE) on histopathology patches, extracting latent embeddings, reconstructing images, and comparing latent spaces. You can download the [pretrained model checkpoints from Zenodo](https://doi.org/10.5281/zenodo.23542520).

## Prerequisites

1. Create and activate the environment specified in [README_Environment.md](../README_Environment.md).
2. Complete patch extraction and dataset preparation as described in [README_Preparation.md](./README_Preparation.md).
3. Create the required output directories and configure dataset and output paths.

Run all commands from the repository root.

## Input Arguments

| Argument | Description | Default |
|---|---|---|
| `--full_data` | Path to the prepared full patch dataset used for training and latent-space analysis. | `./datasets/grade_data_120000.pkl` |
| `--subset_data` | Path to the patch subset used for reconstruction and downstream evaluation. | `./datasets/grade_data_4000.pkl` |
| `--latent_dim` | Integer specifying the latent embedding dimensionality; used during training. | — |

## Train the Adversarial Autoencoder

```bash
python AAE_model.py \
    --full_data /path/to/training_data.pkl \
    --latent_dim LATENT_DIM \
    --subset_data /path/to/subset_data.pkl
```

The script trains the AAE and exports model weights, latent embeddings, and reconstructed patches.

### Output Artifacts

| Artifact | Contents | Location |
|---|---|---|
| Encoder and decoder weights | Model parameters for patch encoding and reconstruction. | `./checkpoints/` |
| Latent-vector pickle file | Approximately 120,000 embeddings corresponding to the full patch dataset. | `./analysis/` |
| Reconstruction pickle file | Approximately 4,000 reconstructed images corresponding to the analysis subset. | `./analysis/` |

Output counts depend on dataset size and sample-selection settings. Maintain consistent sample identifiers or ordering across source patches, embeddings, and reconstructions.

## Run Latent-Space Analysis

```bash
python Latent_space_analysis.py \
    --full_data /path/to/training_data.pkl
```

Configure the model checkpoint, latent-vector, and output paths required by `Latent_space_analysis.py` before execution. Verify that the analysis subset corresponds to the intended samples in the full dataset.