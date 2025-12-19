# autoencoder.py
import torch
import torch.nn as nn
from manifold import Manifold

"""
Creates manifold latent space embeddings
Autoencoder? I barely know 'er!
"""

def mlp(in_dim: int, out_dim: int, hidden: int, depth: int) -> nn.Sequential:
    layers = []
    d = in_dim
    for _ in range(depth - 1):
        layers.append(nn.Linear(d, hidden))
        layers.append(nn.Tanh())  # mild nonlinearity; stable for polynomial geometry experiments
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)

class MLPAutoencoder(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, hidden: int, depth: int, manifold: Manifold):
        super().__init__()
        self.manifold = manifold
        self.enc = mlp(x_dim, z_dim, hidden, depth)
        self.dec = mlp(z_dim, x_dim, hidden, depth)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_raw = self.enc(x)
        z = self.manifold.project(z_raw)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
