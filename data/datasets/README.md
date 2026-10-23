# Downloaded and Prepared Datasets

This directory serves as a placeholder for downloaded datasets and prepared inputs used for training, inference, and analysis. Download the required source data and generate the processed inputs before running the corresponding workflows.

See the [data preparation guide](../../code/README_Preparation.md) for dataset sources, download links, preprocessing instructions, and expected data structures.

Typical contents include:

- Original image patches and associated annotations for the UTSW, Mayo, and WSI1 cohorts.
- Prepared patch datasets such as `grade_data_120000.pkl` and `grade_data_4000.pkl`.
- Other downloaded or processed inputs required by the selected analysis.

## Configuring Dataset Locations

Update `original_dir`, `real_data`, and `WSI1_data.Patches_dir` in [data_dir.yaml](../data_dir.yaml) to match your dataset locations. 

Datasets may also remain in another directory; set the YAML entries to their actual paths instead of copying the data here. Use absolute paths for external storage, or paths relative to the repository root when running scripts and notebooks from that directory. Update any script-specific input paths separately if they are not loaded from the YAML configuration.
