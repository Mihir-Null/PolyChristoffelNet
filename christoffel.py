# christoffel.py
import torch
import torch.nn as nn
from poly_features import PolyFeatures


class ChristoffelPoly(nn.Module):
    """
    Christoffel polynomial model gamma_ijk(z) - torsion free symmetry enforced, could also do levi civita for addn loss signals
    """

    def __init__(self, d: int, degree: int = 2):
        super().__init__()
        self.d = d
        self.phi = PolyFeatures(d, degree)
        M = self.phi.M
        # Coeff tensor C[i,j,k,m]
        self.C = nn.Parameter(0.01 * torch.randn(d, d, d, M))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B,d)
        returns Gamma: (B,d,d,d) with Gamma[b,i,j,k] = Gamma^i_{jk}(z_b)
        """
        phi = self.phi(z)  # (B,M)
        Gamma = torch.einsum("ijkm,bm->bijk", self.C, phi)
        # torsion-free: symmetric in (j,k)
        Gamma = 0.5 * (Gamma + Gamma.transpose(2, 3))
        return Gamma
