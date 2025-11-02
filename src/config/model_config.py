from typing import Dict, List
from dataclasses import dataclass

@dataclass
class VAE_Config:
    in_channels: int
    out_channels: int
    image_size: int
    latent_dim: int
    hidden_dims: List[int]

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)