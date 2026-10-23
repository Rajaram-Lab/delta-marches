# Model Weights and Checkpoints

This directory holds trained model weights and supporting assets for **Delta-Marches: Generative AI-based image synthesis to decode disease-driving morphologic transformations**.

Download the [released model weights from Zenodo](https://doi.org/10.5281/zenodo.23542520). Extract the downloaded files and place the model directories as described below. AAE, image classifier, PAX8 nuclei segmentation, and vasculature weights belong in this `checkpoints/` directory. DIFFAE checkpoints belong in the corresponding `DIFFAE_grade/checkpoints/` and `DIFFAE_HunCRC/checkpoints/` directories at the repository root.

Vasculature segmentation weights are available separately from the [AngioPaper repository](https://zenodo.org/records/14008282).

## Released Models

| Model directory | Contents |
|---|---|
| `AAE_model/` | Adversarial autoencoder encoder and decoder weights for latent dimensions 128, 256, 512, 1024, and 2048. Each dimension has two files: `AAE_model_n{dimension}_enc.pth` and `AAE_model_n{dimension}_dec.pth` (10 weight files total). |
| `img_classifier/` | Renal tumor grade classifier (`grade_classifier_image_224_weight_alldata.pth`) and three-class HunCRC VGG19 classifier (`HunCRC_image_vgg19_classifier_3classes.pth`). |
| `PAX8_Nuclei_Segmentation/` | Released PAX8 nuclei segmentation model (`PAX8_Resnet_1WCCE_9Dice_20X_hovernet_masks-30.h5`) and normalization target (`Updated_PAX8_Normalization_Target.npy`). |
| `DIFFAE_grade/checkpoints/` | Renal tumor grade DIFFAE checkpoints in `grade224_autoenc_130M/`, `grade224_autoenc_130M_8k/`, and `grade224_autoenc_cls/`. |
| `DIFFAE_HunCRC/checkpoints/` | HunCRC DIFFAE checkpoints in `grade224_autoenc_130M/` and `grade224_autoenc_cls/`. |

Each DIFFAE model directory contains `last.ckpt` and `hparams.yaml`. Grade and HunCRC models are dataset-specific, even where their filenames are identical.

## Installation

Copy the extracted model directories into the code repository using the following mapping. Destination paths are relative to the repository root. Preserve directory names, filenames, and capitalization.

| Extracted source directory | Destination in the code repository |
|---|---|
| `AAE_model/` | `checkpoints/AAE_model/` |
| `img_classifier/` | `checkpoints/img_classifier/` |
| `PAX8_Nuclei_Segmentation/` | `checkpoints/PAX8_Nuclei_Segmentation/` |
| `DIFFAE_grade/checkpoints/` | `DIFFAE_grade/checkpoints/` |
| `DIFFAE_HunCRC/checkpoints/` | `DIFFAE_HunCRC/checkpoints/` |

### Vasculature Segmentation Weights

Obtain the vasculature segmentation weights by following the instructions in the [AngioPaper repository](https://zenodo.org/records/14008282). These weights are distributed separately from the Delta-Marches model release.

Place the required model files in:

```text
checkpoints/AngioPaper/Models/MM_PyT_Dice_MultiBatch/
```

Alternatively, update the corresponding `WEIGHTS` entries in [data/data_dir.yaml](../data/data_dir.yaml) to match their location.

### Expected Directory Layout

After installing the released models and the separately obtained vasculature weights:

```text
delta-marches/
├── checkpoints/
│   ├── README.md
│   ├── AAE_model/
│   ├── img_classifier/
│   ├── AngioPaper/
│   │   └── Models/MM_PyT_Dice_MultiBatch/
│   └── PAX8_Nuclei_Segmentation/
│       ├── PAX8_Resnet_1WCCE_9Dice_20X_hovernet_masks-30.h5
│       └── Updated_PAX8_Normalization_Target.npy
├── DIFFAE_grade/
│   └── checkpoints/
│       ├── grade224_autoenc_130M/
│       ├── grade224_autoenc_130M_8k/
│       └── grade224_autoenc_cls/
└── DIFFAE_HunCRC/
    └── checkpoints/
        ├── grade224_autoenc_130M/
        └── grade224_autoenc_cls/
```

## Path Configuration and Use

The `WEIGHTS` section of [data/data_dir.yaml](../data/data_dir.yaml) configures the AAE, image classifier, vasculature, and PAX8 model paths, along with the PAX8 normalization target. Its default paths match the layout above.

Run workflows that use repository-relative paths from the repository root, unless their documentation specifies otherwise. Use weights compatible with the architecture and configuration expected by each script.

### DIFFAE Checkpoints

DIFFAE scripts configure checkpoint paths separately from `data/data_dir.yaml`. Manipulation scripts load:

- `checkpoints/grade224_autoenc_130M/last.ckpt`
- `checkpoints/grade224_autoenc_cls/last.ckpt`, where required

These paths are relative to the corresponding `DIFFAE_grade/` or `DIFFAE_HunCRC/` working directory.

The additional grade checkpoint in `grade224_autoenc_130M_8k/` requires selecting its checkpoint path and using a compatible configuration.

See the [grade DIFFAE guide](../DIFFAE_grade/README.md) and [HunCRC DIFFAE guide](../DIFFAE_HunCRC/README.md) for workflow instructions.

### Using Another Storage Location

Weights may remain outside the repository. Update the corresponding entries under `WEIGHTS` in [data/data_dir.yaml](../data/data_dir.yaml), using absolute paths for external storage. Update DIFFAE and other script-specific checkpoint paths separately.

## Generated Analysis Assets

Generated analysis assets, such as `transition_vector.npy`, are not included in the model release. Follow the relevant workflows to generate the inputs and outputs required for your analysis.

See the [environment setup](../README_Environment.md), [data preparation guide](../code/README_Preparation.md), [AAE evaluation guide](../code/README_Evaluation.md), and [analysis guide](../code/README_analysis.md) for the full pipeline.