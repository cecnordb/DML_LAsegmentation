## Semantic Segmentation of Left Atrium 

This project implements a patch-based segmentation technique for the left atrium using 3D medical images. The primary objective is to accurately segment the left atrium from MRI scans, addressing the challenges posed by a small training dataset with significant variability.

### Data Set
* Data Source: Medical Decathlon (http://medicaldecathlon.com/)
* Target Structure: Left Atrium
* Modality: Mono-modal MRI
* Dataset Size: 30 3D volumes (20 for Training, 10 for Testing)

The dataset presents challenges due to its limited size and the high variability in the images, which may affect the model's generalization capabilities.

_Citation of data: https://arxiv.org/abs/1902.09063_

### Model Architecture
We employ a 3D U-Net model for the segmentation task. This model serves as a baseline for performance evaluation against more advanced architectures, such as UNETR, Swin UNETR, nnFormer.

### Installation
---
