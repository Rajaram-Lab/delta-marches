# Plotting Notebooks

This folder contains Jupyter notebooks for visualizing histopathology analysis results and synthetic image traversals.

| Notebook | Purpose |
|---|---|
| [Result_plotting.ipynb](Result_plotting.ipynb) | Plot renal tumor grade results, Delta-Marches, and morphology measurements. |
| [WSI1_inference.ipynb](WSI1_inference.ipynb) | Analyze traversal results for the WSI1 multiregion dataset. |
| [HunCRC_result_plot.ipynb](HunCRC_result_plot.ipynb) | Analyze colorectal progression traversals and extracted features. |

## Before Running

Activate the environment described in [README_Environment.md](../../README_Environment.md) and complete the preparation, model training, traversal generation, segmentation, and feature extraction described in the [code overview](../README.md).

Save inference results in the expected directories and update [data_dir.yaml](../../data/data_dir.yaml) and any notebook-specific input and output paths. The notebooks use paths relative to the repository root, so launch Jupyter there and ensure the notebook kernel uses that working directory:

```bash
jupyter notebook code/plotting_notebook/Result_plotting.ipynb
```

## Tumor Grade Analysis

Use [Result_plotting.ipynb](Result_plotting.ipynb) to visualize renal tumor grade results, Delta-Marches, and morphology measurements.

## WSI1 Dataset Analysis

Use [WSI1_inference.ipynb](WSI1_inference.ipynb) to analyze image traversal results for the WSI1 multiregion dataset.

## HunCRC Dataset Analysis

Before plotting HunCRC results, classify the traversed images into normal, low-grade dysplasia, and high-grade dysplasia categories, and perform gland and goblet cell segmentation.

The classification workflow uses the following command, with the classifier script path adjusted to its location in your environment:

```bash
python /path/to/huncrc_image_vgg19_classifier.py path/to/traverse_data.pkl
```

The classifier script is not included in this repository. The gland and goblet cell segmentation models come from published projects; refer to the following publications for model details and segmentation instructions:

- **Gland segmentation:** [Published method](https://doi.org/10.1038/s41598-026-36256-0).
- **Goblet cell segmentation:** [Published method](https://doi.org/10.1007/s00535-022-01924-1).

Save the extracted features in `path/to/traverse_data.pkl` using the corresponding keys expected by the analysis code. Configure the notebook input paths to point to the processed datasets and analysis outputs.

After completing classification, segmentation, and feature extraction, use [HunCRC_result_plot.ipynb](HunCRC_result_plot.ipynb) to analyze colorectal progression traversals and extracted features.
