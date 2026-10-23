# Environment Setup and Dependencies

This project uses separate environments for the main analysis and DIFFAE workflow, and for nuclei and vasculature segmentation. The package versions below document the respective software configurations.

Run the setup commands from the repository root, where `req.txt` and `environment_segmentation.txt` are located.

## Main Project Environment

Create a Conda environment from the supplied package specification:

```bash
conda create --name histopathology --file req.txt
conda activate histopathology
```

Replace `histopathology` with your preferred environment name.

Alternatively, configure an environment using the following package versions. This list describes the core dependencies; the supplied specification may include additional packages required by individual scripts.

| Package | Version |
|---|---|
| Python | 3.9 |
| PyTorch | 2.2.2 |
| torchvision | 0.17.2 |
| torchaudio | 2.2.2 |
| PyTorch Lightning | 1.4.5 |
| TensorFlow | 2.16.1 |
| TensorBoard | 2.16.2 |
| NumPy | 1.26.4 |
| pandas | 1.5.3 |
| SciPy | 1.13.0 |
| scikit-learn | 1.4.1.post1 |
| matplotlib | 3.8.4 |
| seaborn | 0.13.2 |
| Jupyter | 1.0.0 |
| JupyterLab | 4.2.3 |
| requests | 2.31.0 |
| tqdm | 4.66.2 |
| pip | 23.3.1 |

The DIFFAE implementation was adapted from the [original DIFFAE repository](https://github.com/phizaz/diffae). Consult its environment specifications and setup instructions for additional model-specific dependencies.

## Nuclei and Vasculature Segmentation Environment

The nuclei and vasculature segmentation models were previously trained by the Rajaram lab using a separate software environment. Use the segmentation package specification to reproduce that configuration.

Create and activate the segmentation environment:

```bash
conda create --name histopathology-segmentation --file environment_segmentation.txt
conda activate histopathology-segmentation
```

Replace `histopathology-segmentation` with your preferred environment name.

The core segmentation dependencies are:

| Package | Version |
|---|---|
| PyTorch | 1.13.0 |
| torchvision | 0.14.0 |
| PyTorch Lightning | 1.8.2 |
| TensorFlow | 2.3.0 |
| NumPy | 1.21.6 |
| pandas | 1.3.5 |
| SciPy | 1.7.3 |
| scikit-learn | 1.0.2 |
| scikit-image | 0.19.3 |
| matplotlib | 3.5.2 |
| Scanpy | 1.8.2 |
| Squidpy | 1.2.2 |
| napari | 0.4.14 |

Use the Python version and any additional dependencies specified in `environment_segmentation.txt`, since the core package list does not define the complete segmentation environment. Activate the appropriate environment before executing its associated scripts.