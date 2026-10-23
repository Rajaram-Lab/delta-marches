# Segmentation, Delta-Marches, and Traversal Analysis

This workflow applies pretrained nuclei and vasculature segmentation models to synthetic image traversals, generates Delta-Marches, and analyzes the spatial correspondence between image changes and segmented structures.

## Prerequisites

Before running the analysis:

1. Download the required [pretrained segmentation models](https://doi.org/10.5281/zenodo.23542520) and configure their checkpoint paths.
2. Generate the traversal datasets as described in [README.md](../DIFFAE_grade/README.md).
3. Activate the appropriate software environment and configure the input and output paths.

The segmentation and heatmap scripts below save analysis outputs to the input pickle (`.pkl`) files.

## Nuclei Segmentation

Apply the pretrained nuclei segmentation model to the traversal images:

```bash
python Nuclei_segmentation_code.py /path/to/traversal_data.pkl
```

The script adds segmentation masks with the following shape:

```text
(N_sample, N_step, 224, 224)
```

Here, `N_sample` is the number of traversal sequences, and `N_step` is the number of images in each sequence.

The masks use the following labels:

| Value | Class |
|---|---|
| 0 | Background |
| 1 | Non-tumor cell |
| 2 | Tumor cell |

## Vasculature Segmentation

Apply the pretrained vasculature segmentation model to the same traversal dataset:

```bash
python SampleGenerate_VascularMask.py /path/to/traversal_data.pkl
```

The script adds segmentation masks with the following shape:

```text
(N_sample, N_step, 224, 224)
```

The masks use the following labels:

| Value | Class |
|---|---|
| 0 | Background |
| 1 | Vasculature |

## Grad-CAM for Real Images

[`GradCam_mask.py`](./GradCam_mask.py) generates Grad-CAM++ heatmaps for the tumor grade classifier using the 4,000 real image patches.

Set `real_data.real_data_4000patches` (input dataset) and `WEIGHTS.grade_classifier` (classifier checkpoint) in `data/data_dir.yaml`. Run from the repository root:

```bash
python code/GradCam_mask.py
```

The script saves the heatmaps under `data['Gradcam']['Gradcampp_mask']` in the same input pickle file. Generate these heatmaps before running spatial analysis on real images.

## Delta-Marches and Spatial Analysis for Tumor Grading

After completing segmentation, run the appropriate command for your dataset.

For synthetic image traversals:

```bash
python Hypothesis_generating.py syn /path/to/traversal_data.pkl
```

For real images:

```bash
python Hypothesis_generating.py real /path/to/image_data.pkl
```

For synthetic traversals, the script generates delta maps with the following shape and saves them to the traversal pickle file:

```text
(N_sample, N_step, 224, 224)
```

For real images, the script analyzes spatial correspondence using Grad-CAM++ heatmaps.

The script computes Jaccard overlap results and saves an analysis report to the configured image output folder.

For instructions on visualizing the results, see [README.md](./plotting_notebook/README.md).

## Pathologist Evaluation

[`pathologist_datacollect.py`](./pathologist_datacollect.py) analyzes an evaluation in which four pathologists each reviewed 40 images (20 real and 20 synthetic). For each image, they identified whether it was real or synthetic and assigned a tumor grade category: low (grades 1–2) or high (grades 3–4).

Configure the ground-truth and four pathologist response Excel files under `pathologist_result` in `data/data_dir.yaml`. Each file must contain `Data` (`Real` or `Synthetic`) and `Tumor grade` (`Low grade` or `High grade`) columns, with rows in the same image order.

Run from the repository root, ensuring the `image/` output folder exists:

```bash
python code/pathologist_datacollect.py
```

The script pools responses from all four pathologists and saves confusion matrix plots to `image/`, showing real-versus-synthetic identification overall and by grade, and tumor grading separately for real and synthetic images.

## WSI1 Dataset Analysis

Follow [README.md](./plotting_notebook/README.md) to analyze image traversal results for the WSI1 multiregion dataset.

## HunCRC Dataset Analysis

Before analyzing the HunCRC dataset, classify the traversed images and perform gland and goblet cell segmentation.

To classify the traversed images into three categories—normal, low-grade dysplasia, and high-grade dysplasia—run:

```bash
python huncrc_image_vgg19_classifier.py /path/to/traversal_data.pkl
```

The gland and goblet cell segmentation models come from published projects. Refer to the following publications for details and segmentation instructions:

- **Gland segmentation:** [Published study](https://doi.org/10.1038/s41598-026-36256-0)
- **Goblet cell segmentation:** [Published study](https://doi.org/10.1007/s00535-022-01924-1)

Save the classification results and extracted segmentation features to the traversal pickle file using the keys expected by the analysis scripts.

After completing classification and segmentation, follow [README.md](./plotting_notebook/README.md) to analyze the HunCRC dataset.
