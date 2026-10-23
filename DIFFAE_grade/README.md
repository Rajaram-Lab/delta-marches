## DIFFAE Training and Latent-Space Manipulation for Histopathology

This directory contains the workflow for training a diffusion autoencoder (DIFFAE) on hematoxylin and eosin (H&E) histopathology patches, extracting semantic representations, training a linear classifier, and generating synthetic images through latent-space manipulation.

### Code Attribution and Acknowledgments

Most of the code in this directory is adapted from the [official implementation of Diffusion Autoencoders (DiffAE)](https://github.com/konpatp/diffae) by Preechakul et al. (CVPR 2022). The original implementation has been modified to support images with a resolution of 224 × 224 pixels and to enable training and analysis on our histopathology datasets. These modifications include adjustments to model configurations, data loading, preprocessing, and training pipelines to meet the requirements of the Delta-Marches framework. We acknowledge the original authors for making their implementation publicly available.

### Prerequisites

Before running the workflow:

1. Configure and activate the environment described in [README_Environment.md](../README_Environment.md).
2. Complete the dataset extraction and patch preparation described in [README_Preparation.md](../code/README_Preparation.md).
3. Create the required input and output directories and update dataset, checkpoint, and output paths in the scripts.

The commands below assume the current working directory contains the DIFFAE scripts. Replace command-line placeholders with the corresponding paths or class labels.

**Compute resources:** The documented training configuration uses a node equipped with four NVIDIA A100 GPUs. Semantic-vector extraction, classifier training, and image manipulation can be performed on an NVIDIA V100 GPU, subject to batch size and available GPU memory.

You can download the [pretrained model checkpoints from Zenodo](https://doi.org/10.5281/zenodo.23542520).

### Workflow

The pipeline consists of four stages:

1. Unsupervised diffusion-autoencoder training.
2. Semantic latent-vector extraction.
3. Supervised linear-classifier training on annotated latent representations.
4. Image reconstruction and latent-space traversal across grades or patients.

### 1. Train the Unsupervised DIFFAE Model

The training workflow uses two H&E patch subsets: an annotated subset containing approximately 8,000 patches and a larger subset containing approximately 18,000 patches.

Prepare the selected subset in an image directory named `grade`, then configure the input and output paths in `data_resize_grade.py`.

Create the LMDB dataset:

```bash
python data_resize_grade.py
```

The preprocessing script produces `grade.lmdb`. Rename this artifact to `grade256.lmdb` to match the dataset name expected by the training configuration.

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


### 2. Extract Semantic Latent Vectors

To extract semantic representations for comparison with the autoencoder (AE) baseline, run:

```bash
python manipulate_cond_vec.py /path/to/real_patches
```

The output is:

```text
imgs_manipulated/cond_vec_list.npy
```

For the complete prepared dataset, this NumPy array is expected to contain semantic vectors for approximately 120,000 patches. The actual number of vectors depends on the input dataset and any filtering performed by the script.

Preserve the correspondence between vector indices and source patches when using these representations for downstream analysis.

### 3. Train the Latent Linear Classifier

Configure the classifier workflow to use the latent representations of the approximately 8,000 annotated patches:

```text
checkpoints/grade224_autoenc_130M/latent.pkl
```

Ensure that the latent vectors and grade labels refer to the same patches and use consistent sample ordering.

Train the linear classifier:

```bash
python run_grade224_cls.py
```

The classifier learns grade-associated directions in the semantic latent space for use in subsequent image-manipulation experiments.

### 4. Reconstruct Images and Generate Synthetic Traversals

Before running these scripts, configure the required model and classifier checkpoint paths and ensure that the output directory exists.

#### Image Reconstruction

Reconstruct real patches using the trained diffusion autoencoder:

```bash
python manipulate_reconstruction.py /path/to/real_patches
```

This step evaluates reconstruction of the input patches using their encoded representations.

#### Across-Grade Traversal

Generate synthetic transitions from a source grade to a target grade:

```bash
python manipulate.py /path/to/real_patches SOURCE_CLASS TARGET_CLASS
```

Replace `SOURCE_CLASS` and `TARGET_CLASS` with the class identifiers expected by the script.

The documented experiment generates grade 1-to-grade 4 traversals for approximately 10,000 grade 1 patches and saves the results as a `.pkl` artifact in `imgs_manipulated/`.

For other grade transitions, update the class arguments and any experiment-specific settings in the script, including sample selection, traversal parameters, and output filenames.

#### Across-Patient Traversal Within a Grade

Generate across-patient transitions while retaining the selected grade:

```bash
python manipulate_samegrade_transition.py \
    /path/to/real_patches \
    /path/to/synthetic_traversal.pkl \
    SOURCE_CLASS
```

Arguments:

- `/path/to/real_patches`: input directory containing real histopathology patches.
- `/path/to/synthetic_traversal.pkl`: traversal artifact to which the script adds the generated results.
- `SOURCE_CLASS`: grade identifier used for the within-grade traversal.

Configure the patient selection, grade selection, and traversal settings in the script for the intended experiment. The documented workflow adds the across-patient synthetic transitions to the specified traversal artifact in `imgs_manipulated/`.

#### WSI1 multiregion data across-punches interpolation

Generate across-patient transitions while retaining the selected grade:

```bash
python manipulate_WSI1_data.py \
    /path/to/WSI1_patches \
    SOURCE_CLASS
    TARGET_CLASS
```

Replace `SOURCE_CLASS` and `TARGET_CLASS` with the region names expected by the script.
.