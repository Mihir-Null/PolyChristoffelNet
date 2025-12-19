# poly_features.py
import torch
import torch.nn as nn
"""
Monomial basis phi(z) up to degree 2 rn, extend to higher later as needed
"""
class PolyFeatures(nn.Module):
    """
    degree 0: 1
    degree 1: z_i
    degree 2: z_i z_j for i<=j
    """
    def __init__(self, d: int, degree: int = 2):
        super().__init__()
        assert degree in (0, 1, 2)
        self.d = d
        self.degree = degree
        self.M = self._feature_dim(d, degree)

    @staticmethod
    def _feature_dim(d: int, degree: int) -> int:
        M = 1
        if degree >= 1:
            M += d
        if degree >= 2:
            M += d * (d + 1) // 2
        return M

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B,d)
        B, d = z.shape
        feats = [torch.ones(B, 1, device=z.device, dtype=z.dtype)]
        if self.degree >= 1:
            feats.append(z)
        if self.degree >= 2:
            quads = []
            for i in range(d):
                for j in range(i, d):
                    quads.append((z[:, i] * z[:, j]).unsqueeze(-1))
            feats.append(torch.cat(quads, dim=-1))
        return torch.cat(feats, dim=-1)
