## DIFFAE Training and Latent-Space Manipulation for Histopathology

This directory contains the workflow for training a diffusion autoencoder (DIFFAE) on hematoxylin and eosin (H&E) histopathology patches, extracting semantic representations, training a linear classifier, and generating synthetic images through latent-space manipulation.

### Code Attribution and Acknowledgments

Most of the code in this directory is adapted from the [official implementation of Diffusion Autoencoders (DiffAE)](https://github.com/konpatp/diffae) by Preechakul et al. (CVPR 2022). The original implementation has been modified to support images with a resolution of 224 × 224 pixels and to enable training and analysis on our histopathology datasets. These modifications include adjustments to model configurations, data loading, preprocessing, and training pipelines to meet the requirements of the Delta-Marches framework. We acknowledge the original authors for making their implementation publicly available.

### Prerequisites

Before running the workflow:

1. Configure and activate the environment described in [README_Environment.md](../README_Environment.md).
2. Complete the dataset extraction and patch preparation described in [README_Preparation.md](../code/README_Preparation.md)..
3. Create the required input and output directories and update dataset, checkpoint, and output paths in the scripts.

The commands below assume the current working directory contains the DIFFAE scripts. Replace command-line placeholders with the corresponding paths or class labels.

**Compute resources:** The documented training configuration uses a node equipped with four NVIDIA A100 GPUs. Semantic-vector extraction, classifier training, and image manipulation can be performed on an NVIDIA V100 GPU, subject to batch size and available GPU memory.

You can download the [pretrained model checkpoints from Zenodo](https://doi.org/10.5281/zenodo.23542520).

### Workflow

The pipeline consists of four stages:

1. Unsupervised diffusion-autoencoder training.
2. Supervised linear-classifier training on annotated latent representations.
3. Image reconstruction and latent-space traversal across colorectal cancer stage.

### 1. Train the Unsupervised DIFFAE Model

The training workflow uses pickle file which should be saved in ./datasets/HunCRC_data.pkl

Start diffusion-autoencoder training:

```bash
python run_grade224.py
```

The documented checkpoint directory is:

```text
checkpoints/
└── grade224_autoenc_130M/
    ├── last.ckpt
    └── latent.pkl
```

- `last.ckpt` stores the model checkpoint used for subsequent encoding and reconstruction.
- `latent.pkl` stores latent representations used by the downstream classifier workflow.

Confirm that the dataset paths, checkpoint paths, and training configuration refer to the intended patch subset before launching training.

### 2. Train the Latent Linear Classifier

Configure the classifier workflow to use the latent representations all trained patches:

```text
checkpoints/grade224_autoenc_130M/latent.pkl
```

Ensure that the latent vectors and grade labels refer to the same patches and use consistent sample ordering.

Train the linear classifier:

```bash
python run_grade224_cls.py
```

The classifier learns grade-associated directions in the semantic latent space for use in subsequent image-manipulation experiments.

### 3. Generate Synthetic Traversals

Before running these scripts, configure the required model and classifier checkpoint paths and ensure that the output directory exists.

#### Traversal from normal to high grade dysplasia

```bash
python manipulate.py
```
The documented experiment generates normal to high grade dysplasia traversals for approximately 5,000 patches and saves the results as a `.pkl` artifact in `imgs_manipulated/`.