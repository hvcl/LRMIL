# LRMIL: Efficient Low-Resolution Multiple Instance Learning via High-Resolution Knowledge Distillation for Whole Slide Image Classification

MICCAI 2026 Early Accept (Top 9%) · Paper link (https://arxiv.org/pdf/2606.06864)

## Overview
![main_figure](https://github.com/hvcl/LRMIL/blob/main/main_figure.png)
Overview of our LRMIL framework. (a) Patch-level cross-resolution distillation. Fine-grained semantic knowledge is distilled to a coarse-level patch encoder. (b) Slidelevel distillation for MIL. An LR-based student MIL model is trained using both baglevel supervision and teacher guidance.

---
## Abstract
Multiple instance learning (MIL) has become a standard paradigm for whole slide image (WSI) analysis in digital pathology, as it enables slide-level prediction without dense annotations. Existing MIL methods typically rely on exhaustive extraction and encoding of high-resolution patches. However, this practice suffers from two critical limitations in real-world clinical settings: it struggles to capture global visual cues at lower magnifications, and incurs substantial computational overhead due to the massive number of high-resolution patches per slide.

To address these limitations, we propose an efficient low-resolution multiple instance learning (LRMIL) framework that transfers high-resolution knowledge to low-resolution representations. LRMIL adopts a two-stage distillation strategy. First, patch-level cross-resolution distillation aligns low-resolution patch embeddings with high-resolution representations. Second, slide-level knowledge distillation trains a low-resolution student MIL model under both slide-level supervision and teacher guidance. At inference time, LRMIL operates exclusively on low-resolution patches, substantially reducing data preprocessing and computational cost.

---
## Datasets
You can download datasets by below links

TCGA cohorts(BRCA, NSCLC, RCC) : https://portal.gdc.cancer.gov/analysis_page?app=Projects

BRACs : https://www.bracs.icar.cnr.it/

---
## How to use
### Train Stage 1
```text
python main.py --stage stage1
```
### Train Stage 2 Step 1
```text
python main.py --stage stage2_teacher
```
### Train Stage 2 Step 2
```text
python main.py --stage stage2_student
```
---
## Citation
If you find this work useful, please cite our paper:
```bibtex
@article{shin2026lrmil,
  title={LRMIL: Efficient Low-Resolution Multiple Instance Learning via High-Resolution Knowledge Distillation for Whole Slide Image Classification},
  author={Shin, Yonghan and Jeong, Won-Ki},
  journal={arXiv preprint arXiv:2606.06864},
  year={2026}
}
```

---
## Acknowledgements
This work was conducted at the **High-performance Visual Computing Lab (HVCL), Korea University**.

We gratefully acknowledge the public datasets used for evaluation, vision backbone used for experiment, and the authors of ABMIL, CLAM, DSMIL, TransMIL, DTFD-MIL, ZOOMMIL and HDMIL for their foundational contributions.
