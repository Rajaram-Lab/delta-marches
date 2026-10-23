# Delta-Marches: Generative AI-based image synthesis to decode disease-driving morphologic transformations 

Code for the paper "Delta-Marches: Generative AI-based image synthesis to decode disease-driving morphologic transformations." by Thuong Nguyen et al.

Delta-Marches is a framework for discovering morphological features associated with disease progression. By combining latent-space traversals with generative AI, it creates class-shifted image counterparts and identifies the cellular and tissue features most affected by these transitions. Delta-Marches is demonstrated on renal carcinoma grading and further evaluated on colorectal dysplastic progression.

## Instructions
The code provided here describes data preparation, model training, latent traversal, Delta-Marches analysis, and the generation of the figures in the paper. Data sources are described in [Preparation](./code/README_Preparation.md); the UTSW and WSI1 images are restricted (see the paper’s Data and Materials Availability statement). We have publicly deposited:

1. The code (this repository)
2. The software environments needed to run the code (see [Environment](./README_Environment.md))
3. Our trained models (DIFFAE, AAE, tumor grade classifier, HunCRC classifier, and nuclei segmentation), archived on  [Zenodo](https://doi.org/10.5281/zenodo.23542520)
4. Notebooks used to generate the figures in the paper

To run the pipeline, follow the steps below in order.

| Step | Description | Documentation |
|----------|----------|----------|
| 1. Environment Setup | Install the main PyTorch environment and the separate segmentation environment | [Environment](./README_Environment.md) |
| 2. Data Preparation | Extract labeled patches and build the training and analysis datasets | [Preparation](./code/README_Preparation.md) |
| 3. DIFFAE training and latent traversal | Train DIFFAE models and generate synthetic class transitions (ccRCC grade, WSI1, HunCRC) | [Grade](./DIFFAE_grade) [HunCRC](./DIFFAE_HunCRC) |
| 4. AAE training and comparison | Train the AAE and compare its latent space with DIFFAE’s | [Evaluation](./code/README_Evaluation.md) |
| 5. Delta-Marches analysis | Segment nuclei, vasculature, glands and goblet cells, and localize changes with Delta-Maps | [Analysis](./code/README_analysis.md) |
| 6. Figure generation | Notebooks that produce the figures for the grade, WSI1 and HunCRC analyses | [Figures](./code/plotting_notebook) |
