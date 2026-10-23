# Code Overview

This folder contains scripts for histopathology data preparation, adversarial autoencoder (AAE) training and evaluation, and analysis of synthetic image traversals for renal tumor grading, WSI1, and colorectal progression (HunCRC).

## Setup

Create and activate the environment described in [README_Environment.md](../README_Environment.md). Prepare the required datasets and [pretrained model checkpoints](https://doi.org/10.5281/zenodo.23542520), configure input and output paths, and create output directories before running the scripts.

## Workflow

1. **Prepare data:** Extract tumor patches, assign grade labels, generate training and analysis subsets, and prepare the HunCRC dataset. See [README_Preparation.md](README_Preparation.md).
2. **Train and evaluate the AAE:** Train the model, export latent embeddings and reconstructed images, and compare latent representations. See [README_Evaluation.md](README_Evaluation.md).
3. **Analyze image traversals:** Apply nuclei and vasculature segmentation, generate Delta-Marches, and measure spatial overlap. The analysis guide also covers WSI1 and HunCRC workflows. See [README_analysis.md](README_analysis.md).

Generate DIFFAE traversals using the [training and traversal guide](../DIFFAE_grade/README.md) before traversal analysis. For visualization and dataset-specific analyses, see [README.md](plotting_notebook/README.md).

**Note:** Segmentation and traversal-analysis scripts save results back to the input pickle files. Preserve sample ordering across images, labels, embeddings, and reconstructions.
