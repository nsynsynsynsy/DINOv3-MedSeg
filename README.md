# Adaptive Feature Gating and Decoding for Lightweight Medical Image Segmentation with DINOv3

Official PyTorch implementation of the paper **"Adaptive Feature Gating and Decoding for Lightweight Medical Image Segmentation with DINOv3"** (Accepted by **IEEE BIBM 2026**).

## 📖 Overview

While fully-supervised 3D convolutional networks achieve state-of-the-art accuracy in medical image segmentation, their substantial computational cost limits deployment in resource-constrained clinical settings. We propose a parameter-efficient **2.5D segmentation framework** built atop a completely frozen **DINOv3** backbone, comprising lightweight learnable components (**only 1.75M trainable parameters** and **5.91G FLOPs**):

- **2.5D Contextual Input**: Stacks adjacent CT slices ($S_{z-1}, S_z, S_{z+1}$) along the channel axis to inject local inter-slice anatomical continuity with zero parameter overhead.
- **Adaptive Feature Gating Module (AFGM)**: Applies independent bottleneck MLPs at four intermediate backbone layers to recalibrate channel-wise importance weights, bridging the natural-to-medical domain gap.
- **Progressive Refinement Decoder (PRD)**: Aggregates multi-layer gated features at a single scale and progressively recovers spatial resolution through alternating $2\times$ bilinear upsampling and lightweight residual refinement blocks.

## 📂 Repository Structure

- `blocks.py` / `dpt.py`: Implementation of the core model components (AFGM, PRD, and backbone feature extraction).
- `dataset.py` / `preprocess_msd.py`: Data loading, 2.5D slice construction, and preprocessing pipelines for AMOS22 and MSD Lung Tumor datasets.
- `train_segdino.py`: Training script for our proposed method on the **AMOS22** dataset.
- `train_segdino_msd.py`: Training script for our proposed method on the **MSD Lung Tumor** dataset.
- `eval_all_metrics.py` / `final_eval_and_plot.py`: Evaluation scripts for computing 3D volumetric Dice (mDice) and 95th percentile Hausdorff Distance (mHD95).
- `calc_flops.py`: Script for computing trainable parameters and FLOPs.
- `paper_vis_comparison.py` / `draw_paper_plot.py`: Scripts for generating qualitative visualization figures and charts in the paper.

## 🛠️ Requirements

- Python >= 3.8
- PyTorch >= 2.0 (with CUDA support)

```bash
pip install torch torchvision timm numpy scipy nibabel SimpleITK matplotlib tqdm
