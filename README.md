# LRMIL: Efficient Low-Resolution Multiple Instance Learning via High-Resolution Knowledge Distillation for Whole Slide Image Classification

MICCAI 2026 Early Accept (Top 9%) · Paper link (https://arxiv.org/pdf/2606.06864)

## Overview
![main_figure](https://github.com/hvcl/LRMIL/blob/main/main_figure.png)
Fig. 1. Overview of our LRMIL framework. (a) Patch-level cross-resolution distillation.
Fine-grained semantic knowledge is distilled to a coarse-level patch encoder. (b) Slidelevel distillation for MIL. An LR-based student MIL model is trained using both baglevel supervision and teacher guidance.

---
## Abstract
Multiple instance learning (MIL) has become a standard paradigm for whole slide image (WSI) analysis in digital pathology, as it enables slide-level prediction without dense annotations. Existing MIL methods typically rely on exhaustive extraction and encoding of high-resolution patches. However, this practice suffers from two critical limitations in real-world clinical settings: it struggles to capture global visual cues at lower magnifications, and incurs substantial computational overhead due to the massive number of high-resolution patches per slide. To address these limitations, we propose an efficient low-resolution multiple instance learning (LRMIL) framework that transfers high-resolution knowledge to low-resolution representations. LRMIL adopts a two-stage distillation strategy. First, patch-level cross-resolution distillation aligns low-resolution patch embeddings with high-resolution representations. Second, slide-level knowledge distillation trains a low-resolution student MIL model under both slide-level supervision and teacher guidance. At inference time, LRMIL operates exclusively on low-resolution patches, substantially reducing data preprocessing and computational cost. Extensive experiments on multiple WSI benchmarks demonstrate that LRMIL consistently outperforms state-of-the-art MIL methods while achieving more efficient inference. These results highlight LRMIL as a practical and scalable solution for WSI analysis in clinical pathology.
