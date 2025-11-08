# AstroVAE

A lightweight variational autoencoder (VAE) project to learn the underlying distribution of astronomical images from the GalaxiesML dataset and generate realistic synthetic galaxy images for research purposes. For mathematical details, please see [Original VAE Paper](https://arxiv.org/abs/1906.02691)

![A simple demonstration of Variational AutoEncoder](asset/image_2.jpeg)

## Overview
This repository trains a VAE to model galaxy images so researchers can:
- Generate synthetic images for data augmentation, algorithm testing, and simulation.
- Explore the learned latent space for morphology / feature analysis.
- Produce controllable samples by manipulating latent vectors.

## Key ideas
- Unsupervised density modeling of astronomical images using a VAE.
- Modular training and sampling pipeline (data loader, model, trainer).
- Export of generated images and latent representations for downstream tasks.

## Dataset
- Dataset: GalaxiesML ([link](https://datalab.astro.ucla.edu/galaxiesml.html))
- Expected layout:
    - data/
        - train/
        - val/
        - test/
    - Images should be preprocessed (same size, normalized).

## Requirements
- Python 3.13
- torch, torchvision, numpy, matplotlib, pandas, tqdm, rich, scipy, h5py 

Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt # Install torch-related packages separately to avoid bugs
```

## Quickstart

1. Prepare dataset in data folder containing train/val/test.
2. Train a VAE (example):
```bash
python -m script.train \
    --config_path /path/to/experiment/config \
    --checkpoint_path /path/to/saved/checkpoint \  # Optional
    --train_data_dir /path/to/train/dir \
    --val_data_dir /path/to/val/dir \
    --test_data_dir /path/to/test/dir              # Optional
```
3. Generate samples from a trained checkpoint:
```bash
python -m script.sample \
    --checkpoint outputs/experiment1/checkpoint.pt \
    --num-samples 100 \
    --output-dir outputs/experiment1/samples
```
4. Visualize latent traversals or reconstructions with:
```bash
python -m script.visualize 
    --checkpoint outputs/experiment1/checkpoint.pt \
    --mode reconstructions
```

## Training Notes
- Normalize images to [0, 1] or standardized values consistent with the likelihood model you use.
- Start with a small latent dimension (e.g., 32–64) and scale up if needed.
- Monitor reconstruction loss and KL divergence separately to diagnose under/over-regularization.
- Experiment with KL divergence weight in YAML config 
- Use data augmentation (rotations, flips) if morphology invariance is desired.

## Outputs
- Trained checkpoints (model weights and optimizer state)
- Sample images (PNG/NPY)
- Latent codes for the dataset (for downstream analysis)

## Experiments & evaluation
- Quantitative: reconstruction loss, ELBO, FID (or other perceptual metrics) vs. baseline.
- Qualitative: side-by-side comparisons of real vs. generated images and latent traversals.

## Contributing
- Small, focused PRs for new models, training strategies, or visualization tools.
- Include reproducible scripts and configuration for experiments.

## License
MIT LICENSE

