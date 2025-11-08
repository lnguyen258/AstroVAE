import torch
from src.model import BaseVAE
from src.config import VAE_Config
from torch import nn, Tensor
from torch.nn import functional as F
from typing import *


class VanillaVAE(BaseVAE):
    def __init__(
            self,
            config: Optional[VAE_Config] = None,
            in_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            image_size: Optional[int] = None,
            latent_dim: Optional[int] = None,
            hidden_dims: Optional[List[int]] = None,
            **kwargs
    ):
        super(VanillaVAE, self).__init__()

        if config is not None:
            self.in_channels = in_channels if in_channels is not None else config.in_channels
            self.out_channels = out_channels if out_channels is not None else config.out_channels
            self.image_size = image_size if image_size is not None else config.image_size
            self.latent_dim = latent_dim if latent_dim is not None else config.latent_dim
            self.hidden_dims = hidden_dims if hidden_dims is not None else config.hidden_dims
        else:
            self.in_channels = in_channels if in_channels is not None else 3
            self.out_channels = out_channels if out_channels is not None else 3
            self.image_size = image_size if image_size is not None else 64
            self.latent_dim = latent_dim if latent_dim is not None else 128
            self.hidden_dims = hidden_dims if hidden_dims is not None else [32, 64, 128, 256, 512]

        modules = []

        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            self.in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.size_after_conv = self.image_size // (2 ** len(self.hidden_dims))
        area_after_conv = self.size_after_conv ** 2
        self.fc_mu = nn.Linear(self.hidden_dims[-1] * area_after_conv, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1] * area_after_conv, self.latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[-1] * area_after_conv)
        self.hidden_dims.reverse()
        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
    
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(self.hidden_dims[-1],
                                               self.hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(self.hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(self.hidden_dims[-1], self.out_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], self.size_after_conv, self.size_after_conv)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def sample(self, num_samples: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]