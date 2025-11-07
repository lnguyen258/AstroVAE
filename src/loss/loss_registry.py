from torch import nn
from .vae_loss import vae_loss


LOSS_REGISTRY = {
    'cross_entropy_loss': nn.CrossEntropyLoss,
    'bce_with_logits_loss': nn.BCEWithLogitsLoss,
    'mse_loss': nn.MSELoss,
    'vae_loss': vae_loss,
    # Add more loss functions here as needed
}