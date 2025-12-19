# manifold.py
import torch
import torch.nn as nn

class Manifold(nn.Module):
    """
    [OPTIONAL]
    Manifold constraint layer for the latent embeddings - optional, keep off unless things aren't going well
    """
    def __init__(self, kind: str, eps: float = 1e-6):
        super().__init__()
        assert kind in ("none", "sphere", "hyperboloid")
        self.kind = kind
        self.eps = eps

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project points onto the chosen manifold.
        z: (..., d)
        """
        if self.kind == "none":
            return z

        if self.kind == "sphere":
            norm = torch.linalg.norm(z, dim=-1, keepdim=True).clamp_min(self.eps)
            return z / norm

        if self.kind == "hyperboloid":
            # Lorentz norm: <z,z>_L = -z0^2 + sum_{i>0} zi^2
            z0 = z[..., :1]
            zs = z[..., 1:]
            space = torch.sum(zs * zs, dim=-1, keepdim=True)
            # enforce -z0^2 + space = -1 => z0 = sqrt(1 + space)
            z0_new = torch.sqrt(1.0 + space).clamp_min(self.eps)
            return torch.cat([z0_new, zs], dim=-1)

        raise RuntimeError("Unknown manifold kind.")

    def tangent_project(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Project velocity to the tangent space (optional but useful if you enforce constraints).
        """
        if self.kind == "none":
            return v

        if self.kind == "sphere":
            # Tangent: v <- v - <v,z> z
            inner = torch.sum(v * z, dim=-1, keepdim=True)
            return v - inner * z

        if self.kind == "hyperboloid":
            # Lorentz inner product: <a,b>_L = -a0 b0 + sum_{i>0} ai bi
            a0 = v[..., :1]
            as_ = v[..., 1:]
            b0 = z[..., :1]
            bs = z[..., 1:]
            inner_L = (-a0 * b0 + torch.sum(as_ * bs, dim=-1, keepdim=True))
            # Tangent condition: <v,z>_L = 0 -> subtract component along z
            # v <- v - inner_L * z / <z,z>_L ; but <z,z>_L = -1 on hyperboloid
            return v + inner_L * z

        raise RuntimeError("Unknown manifold kind.")
