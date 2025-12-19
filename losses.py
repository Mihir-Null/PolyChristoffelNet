# losses.py
import torch
from config import Signature

"""
Recession indicators
"""

def traj_mse(z_pred: torch.Tensor, z_true: torch.Tensor) -> torch.Tensor:
    return ((z_pred - z_true) ** 2).mean()

def recon_mse(x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    return ((x_pred - x_true) ** 2).mean()

def symmetry_loss(g: torch.Tensor) -> torch.Tensor:
    return ((g - g.transpose(1, 2)) ** 2).mean()

def logabsdet_loss(g: torch.Tensor, floor: float = -2.0) -> torch.Tensor:
    g = 0.5 * (g + g.transpose(1, 2))
    sign, logabsdet = torch.linalg.slogdet(g)
    logabsdet = torch.where(sign == 0, torch.full_like(logabsdet, -1e9), logabsdet)
    return torch.nn.functional.softplus(floor - logabsdet).mean()

def signature_loss(g: torch.Tensor, sig: Signature, beta: float = 10.0) -> torch.Tensor:
    g = 0.5 * (g + g.transpose(1, 2))
    eigs = torch.linalg.eigvalsh(g)  # (B,d), ascending
    d = eigs.shape[-1]
    target = torch.tensor([-1.0] * sig.q + [1.0] * sig.p, device=g.device).view(1, d)
    return torch.nn.functional.softplus(-beta * target * eigs).mean()
