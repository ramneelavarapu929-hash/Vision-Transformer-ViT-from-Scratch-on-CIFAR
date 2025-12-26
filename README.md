**Vision Transformer (ViT) from Scratch on CIFAR-10**

This repository contains a modular implementation of a Vision Transformer (ViT) designed specifically for the CIFAR-10 dataset. 

Unlike standard ViT models that use $224 \times 224$ images, this version is optimized for $32 \times 32$ resolution using $4 \times 4$ patches to maintain high sequence density and computational efficiency on laptop GPUs (like the NVIDIA RTX A3000).


**🚀 Key Features**


  Modular Architecture: Clean separation between Patch Embedding, Transformer Encoder, and MLP heads.
  
  Optimized for Small Data: Custom patch sizing ($P=4$) to ensure $64$ tokens per image.
  
  Advanced Training Pipeline: Includes OneCycleLR scheduling, AdamW optimization, and dataset-specific normalization.
  
  Production-Ready: Script for manual "side-loading" of CIFAR-10 to bypass proxy issues.


**Project Structure**
├── data/                  # Local CIFAR-10 binaries

├── src/

│   ├── modules/

│   │   ├── patching.py    # Patching & Linear Projection

│   │   ├── transformer.py # Multi-head Attention & LayerNorm

│   │   └── mlp.py         # Feed-forward blocks

│   ├── dataset.py         # Custom loaders & CIFAR-10 stats

│   └── model.py           # Model stitching (ViT-Tiny/Small)

├── train.py               # Main training script (with Windows multiprocessing support)

├── infer.py             # Inference script for single images

├── checkpoints/

     ├── vit_cifar10_weights.pth # Saved model state_dict
     


**Training Configuration**


  For training on an RTX A3000, the following hyperparameters are recommended:
  
  Patch Size: $4 \times 4$Embedding
  
  Dim: $256$
  
  Batch Size: $128$
  
  Optimizer: AdamW ($wd=0.05$)
  
  LR Scheduler: OneCycleLR (Max $LR=5e-4$)
  
  Normalization: Mean (0.4914, 0.4822, 0.4465), Std (0.2023, 0.1994, 0.2010)
