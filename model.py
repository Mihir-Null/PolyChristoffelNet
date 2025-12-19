# model.py
import torch
import torch.nn as nn
from config import Signature
from manifold import Manifold
from autoencoder import MLPAutoencoder
from christoffel import ChristoffelPoly
from integrators import rollout_geodesic_resnet

"""
Full model wrapper because I got sick of trying to get the individual pieces in DataParallel
"""
class BaseMetric(nn.Module):
    """
    Learnable base metric g(z_base) with fixed signature by construction:
      g0 = L diag(signs * exp(scales)) L^T
    """
    def __init__(self, d: int, sig: Signature):
        super().__init__()
        assert sig.p + sig.q == d
        self.d = d
        self.sig = sig
        self.L = nn.Parameter(torch.randn(d, d) * 0.01)
        self.log_scales = nn.Parameter(torch.zeros(d))
        signs = torch.tensor([-1.0] * sig.q + [1.0] * sig.p)
        self.register_buffer("signs", signs)

    def forward(self) -> torch.Tensor:
        L = torch.tril(self.L)
        scales = torch.exp(self.log_scales)
        D = torch.diag(self.signs * scales)
        g0 = L @ D @ L.transpose(0, 1)
        g0 = 0.5 * (g0 + g0.transpose(0, 1))
        return g0

class FullGeoModel(nn.Module):
    def __init__(
        self,
        x_dim: int,
        d: int,
        poly_degree: int,
        use_ae: bool,
        ae_hidden: int,
        ae_depth: int,
        manifold_kind: str,
        signature: Signature,
    ):
        super().__init__()
        self.d = d
        self.manifold = Manifold(manifold_kind)
        self.christoffel = ChristoffelPoly(d=d, degree=poly_degree)
        self.base_metric = BaseMetric(d=d, sig=signature)

        self.use_ae = use_ae
        if use_ae:
            self.ae = MLPAutoencoder(x_dim=x_dim, z_dim=d, hidden=ae_hidden, depth=ae_depth, manifold=self.manifold)
        else:
            self.ae = None

    def encode_seq(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (B,T,x_dim)
        returns z_seq: (B,T,d)
        """
        if not self.use_ae:
            return self.manifold.project(x_seq)  # assumes x_dim==d in no-AE mode

        B, T, xdim = x_seq.shape
        x_flat = x_seq.reshape(B * T, xdim)
        z_flat = self.ae.encode(x_flat)
        return z_flat.reshape(B, T, self.d)

    def decode_seq(self, z_seq: torch.Tensor) -> torch.Tensor:
        """
        z_seq: (B,T,d) -> x_pred: (B,T,x_dim)
        """
        if not self.use_ae:
            return z_seq  # identity

        B, T, d = z_seq.shape
        z_flat = z_seq.reshape(B * T, d)
        x_flat = self.ae.decode(z_flat)
        return x_flat.reshape(B, T, -1)

    def rollout(self, z0: torch.Tensor, v0: torch.Tensor, T: int, dt: float) -> torch.Tensor:
        return rollout_geodesic_resnet(
            self.christoffel, z0, v0, T=T, dt=dt, manifold=self.manifold, project_each_step=True
        )
