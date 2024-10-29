# KE3DLoc: Knowledge Enhanced Protein Subcellular Localization Prediction

This repository contains a deep learning-based pipeline, **KE3DLoc**, for predicting protein subcellular localization patterns from 3D fluorescence images. For further details, please refer to the publication: "Knowledge enhanced protein subcellular localization prediction from 3D fluorescence microscope images."

## Table of Contents
1. [1-Data Preprocessing](#1-data-preprocessing)
2. [2-Cell Feature Extraction Module of KE3DLoc](#2-cell-feature-extraction-module-of-KE3DLoc)
3. [3-Statistical Cell](#3-statistical-cell)
4. [4-Cytoself](#4-cytoself)
5. [5-KE3DLoc](#5-ke3dloc)

---

### 1-Data Preprocessing
1. **OpenCell Dataset**: Access the dataset from the [OpenCell](https://opencell.czbiohub.org/download). Use the AWS S3 CLI for downloading, following the instructions provided on the website.
2. **Allen Cell Dataset**: Access via [QuiltData](https://open.quiltdata.com/b/allencell/packages/aics/pipeline_integrated_single_cell). Download with the `download_Allen_cell_data.py` script.
3. **Allen hiPSC Dataset**: Access via [QuiltData](https://open.quiltdata.com/b/allencell/packages/aics/hipsc_single_cell_image_dataset). Download with the `download_Allen_hiPSC_data.py` script.
4. After downloading, segment the cell images using the corresponding `preprocessing.py` to prepare data for training and inference.

### 2-Cell Feature Extraction Module of KE3DLoc
The cell feature extraction module in KE3DLoc provides various experiments:

- **3D Branch**: Run `3D.py` for 3D experiments.
- **2D Branch**: Run `2D.py` for 2D experiments.
- **Mix  Module**: Run `mix.py` to conduct mixed branching experiments.
- **ID Loss Experiments**: Execute `ID.py` for ID loss-related experiments.
  
To experiment with subcellular location classification loss, modify the `criterion`'s asymmetric loss parameter and the class confidence weight in `ProteinDataset` in the respective `.py` files. Additional loss details are in `src/losses`.

### 3-Statistical Cell
[Statistical Cell](AllenCellModeling/pytorch_integrated_cell) is trained with the Allen Cell dataset. After training, its encoder (`StatisticalCell-encoder.pth`) is extracted and can be loaded through `3-Statistical Cell Model/*.py` for image feature extraction and classification.

### 4-Cytoself
[Pre-trained Cytoself](https://github.com/royerlab/cytoself/tree/cytoself-tensorflow) is used for feature extraction. Follow these steps:
1. Convert 3D images to 2D format for cytoself with `image_to2D.py`.
2. Use `examples/simple_example.py` in the [Cytoself](https://github.com/royerlab/cytoself/tree/cytoself-tensorflow) to extract global and local features, saving them in `embedding_feature`.

   These features can be loaded for classification using `4-Cytoself/*.py`.

### 5-KE3DLoc
- **GO Feature Extraction**: GO features, pre-extracted with PubmedBERT, are stored in `best_model/go_embedding.pth`.
- **Knowledge Graph Triplets**: Protein-GO and GO-GO triplets from the Protein Knowledge Graph associated with proteins in the Opencell dataset can be found in:
  - `best_model/Protein_GO_ID_old_weight.pkl`
  - `best_model/GO_GO_ID_old_weight.pkl`
  
   Knowledge representation learning methods are implemented in `src/KGEmodel.py`.

