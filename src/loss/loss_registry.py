from torch import nn
from .vae_loss import VAELoss

LOSS_REGISTRY = {
    'cross_entropy_loss': nn.CrossEntropyLoss,
    'bce_with_logits_loss': nn.BCEWithLogitsLoss,
    'mse_loss': nn.MSELoss,
    'vae_loss': VAELoss,
    # Add more loss functions here as needed
}