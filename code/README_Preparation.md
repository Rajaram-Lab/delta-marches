# Data Preprocessing

This directory contains scripts for converting histopathology datasets into patch-level inputs for model training, inference, and downstream analysis. The workflow includes tumor-region selection, grade-label assignment, patch extraction, subset sampling, and dataset serialization.

## Prerequisites

Create and activate the environment described in [README_Environment.md](../README_Environment.md) before running the preprocessing scripts.

The commands below assume execution from the repository root. Download the required datasets, create the output directories, and configure the input and output paths before processing.

Data supporting the manuscript analyses are provided through the manuscript, Supplementary Materials, and the dataset sources listed below.

## Dataset Sources and Input Requirements

### Grade-Annotated Renal Cell Carcinoma Cohort

The grading cohort contains **128 whole-slide images (WSIs)**:

- 100 WSIs from UTSW.
- 28 WSIs from the [Mayo Clinic repository](https://doi.org/10.25452/figshare.plus.19310870).

An expert pathologist, V. Panwar, annotated spatially localized tumor regions with coherent morphology and assigned each region a grade from **1 to 4**. Case counts and patch distributions by grade are reported in **Supplementary Table S1**.

The preprocessing workflow requires an HDF5 data directory containing the slide-level image and annotation information. Supply this directory through `--data_path`.

### Additional Mayo Training Cohort

An additional cohort of approximately **1,000 WSIs** was obtained from the [Mayo Clinic repository](https://doi.org/10.25452/figshare.plus.19310870).

Prepare this cohort by identifying tumor regions with the tumor-region prediction model and extracting **224 × 224 pixel patches at 20× magnification**. Serialize the extracted patches into a Python pickle (`.pkl`) file using the structure expected by the preprocessing scripts.

This cohort supplies additional tumor patches for the expanded DIFFAE training dataset.

### WSI1 Multi-Region Cohort

The WSI1 multi-region sequencing cohort is described in the [associated publication](https://pubmed.ncbi.nlm.nih.gov/40968388/).

This cohort supports analyses linking histologic morphology to spatially matched molecular measurements.

### HunCRC Colorectal Cohort

The [HunCRC dataset](https://doi.org/10.6084/m9.figshare.c.5927795) provides annotated colorectal histology patches and labels used in the colorectal progression analysis.

Preparation instructions and the required serialized data structure are provided below.

## Patch Generation for Tumor-Grading Analysis

Tumor-grading analyses use patches extracted from regions identified as tumor by a previously trained tumor-region segmentation or prediction model. Grade labels are derived from the expert pathologist annotations.

The target patch size is **224 × 224 pixels at 20× magnification**.

### Run the Preprocessing Script

For the grade-annotated cohort:

```bash
python grade_data_load.py \
    --data UTSW \
    --data_path /path/to/grade_hdf5_data
```

For the additional Mayo cohort:

```bash
python grade_data_load.py \
    --data Mayo \
    --data_path /path/to/mayo_data
```

| Argument | Description |
|---|---|
| `--data` | Dataset selector: `UTSW` or `Mayo`. |
| `--data_path` | Path to the source data required by the selected processing branch. |

For `UTSW`, provide the HDF5 directory containing image and annotation information. For `Mayo`, provide the prepared pickle input location expected by the script.

### Generated Datasets

The preprocessing workflow produces the following datasets. Patch counts are approximate and depend on the available inputs and sampling configuration.

| Dataset | Approximate size | Output format | Purpose |
|---|---:|---|---|
| Full grade-annotated tumor dataset | 120,000 patches | `.pkl` file | Patch-level analysis using tumor-grade annotations. |
| Tumor-grade analysis subset | 4,000 patches | `.pkl` file | Downstream tumor-grade analyses. |
| DIFFAE/AAE comparison subset | 8,000 patches | Image directory | DIFFAE training and comparison of DIFFAE and AAE latent representations. |
| Expanded DIFFAE training dataset | 18,000 patches | Image directory | DIFFAE training with additional tumor morphology from the Mayo cohort. |

The expanded training dataset combines approximately **8,000 grade-annotated patches** from the renal cell carcinoma grading cohort with **10,000 additional tumor patches** from the Mayo cohort.

Preserve the association between each patch, its source slide, and its available annotations when generating subsets.

## Patch Preparation for Colorectal Progression Analysis

### Download the HunCRC Images and Labels

Download the slide directories corresponding to the **248 × 248 µm tissue field of view** from the [HunCRC repository](https://doi.org/10.6084/m9.figshare.c.5927795).

Each image has a spatial resolution of **512 × 512 pixels**. Each slide directory contains:

- A directory of `.jpg` image patches.
- A `.csv` file containing the associated patch labels.

The class distribution used in the colorectal progression analysis is reported in **Table 3** of the manuscript.

### Assemble the Serialized Dataset

Combine the images and labels into a single Python dictionary and serialize it as a pickle file.

The required keys and array shapes are:

| Key | Shape | Contents |
|---|---|---|
| `img` | `(N_patches, 512, 512, 3)` | RGB image patches. |
| `class` | `(N_patches, 1)` | Corresponding colorectal progression labels. |

The dictionary should follow this structure:

```python
dataset = {
    "img": images,
    "class": labels,
}
```

Here, `images` and `labels` contain the arrays described above. Image and label entries must use identical sample ordering: `images[i]` must correspond to `labels[i]`.

### Save the DIFFAE Input

Save the serialized dictionary at:

```text
./datasets/HunCRC_data.pkl
```

The HunCRC DIFFAE data loader expects this path and loads the file during training. Ensure that the `datasets` directory exists and that the saved artifact contains both required keys with matching sample counts.