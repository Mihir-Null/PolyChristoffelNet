# christoffel_geodesic_train.py
# Minimal Christoffel-polynomial geodesic model + metric reconstruction (CUDA, batched)

import math
import os
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


"""
Single file sketch/reference that was refactored
"""

# ----------------------------
# 1) Utilities: device, seeds
# ----------------------------

def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------
# 2) Synthetic dataset: inertial in Cartesian,
#    observed in polar (r, theta)
# -----------------------------------------

class PolarInertialDataset(Dataset):
    """
    Generate trajectories by linear motion in Cartesian:
        x(t) = x0 + vx * t
        y(t) = y0 + vy * t
    then convert to polar (r, theta). This produces nontrivial Christoffels in (r, theta).
    """

    def __init__(
        self,
        n_traj: int = 4096,
        T: int = 50,
        dt: float = 0.05,
        r_min: float = 0.5,
        r_max: float = 2.0,
        v_scale: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.T = T
        self.dt = dt

        # Sample initial positions away from origin to avoid r ~ 0 singularity
        # sample angle and radius
        ang = 2 * math.pi * torch.rand(n_traj, generator=g)
        rad = r_min + (r_max - r_min) * torch.rand(n_traj, generator=g)
        x0 = rad * torch.cos(ang)
        y0 = rad * torch.sin(ang)

        # sample velocities
        vx = v_scale * torch.randn(n_traj, generator=g)
        vy = v_scale * torch.randn(n_traj, generator=g)

        # time grid
        t = torch.arange(T, dtype=torch.float32) * dt  # (T,)
        t = t[None, :]  # (1, T)

        x = x0[:, None] + vx[:, None] * t  # (N, T)
        y = y0[:, None] + vy[:, None] * t  # (N, T)

        r = torch.sqrt(x * x + y * y)
        theta = torch.atan2(y, x)

        # unwrap theta along time to avoid jumps at pi/-pi
        theta = torch.unwrap(theta, dim=1)

        z = torch.stack([r, theta], dim=-1)  # (N, T, 2)

        # approximate initial velocity in z-coordinates
        v0 = (z[:, 1] - z[:, 0]) / dt  # (N, 2)

        self.z0 = z[:, 0]        # (N, 2)
        self.v0 = v0             # (N, 2)
        self.z_true = z          # (N, T, 2)

    def __len__(self) -> int:
        return self.z0.shape[0]

    def __getitem__(self, idx: int):
        return self.z0[idx], self.v0[idx], self.z_true[idx]


# -----------------------------------------
# 3) Polynomial feature map phi(z)
#    (degree 2 default; easy to extend)
# -----------------------------------------

class PolyFeatures(nn.Module):
    """
    Build monomial features up to a given degree.
    For z in R^d:
      degree 0: 1
      degree 1: z_i
      degree 2: z_i z_j (with i<=j)
    Returns phi(z) shape (B, M).
    """

    def __init__(self, d: int, degree: int = 2):
        super().__init__()
        assert degree in (0, 1, 2), "Keep minimal; extend if needed."
        self.d = d
        self.degree = degree
        self.M = self._feature_dim(d, degree)

    @staticmethod
    def _feature_dim(d: int, degree: int) -> int:
        M = 1  # constant
        if degree >= 1:
            M += d
        if degree >= 2:
            M += d * (d + 1) // 2
        return M

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, d)
        B, d = z.shape
        feats = [torch.ones(B, 1, device=z.device, dtype=z.dtype)]
        if self.degree >= 1:
            feats.append(z)  # (B, d)
        if self.degree >= 2:
            # quadratic terms i<=j
            quads = []
            for i in range(d):
                for j in range(i, d):
                    quads.append((z[:, i] * z[:, j]).unsqueeze(-1))
            feats.append(torch.cat(quads, dim=-1))  # (B, d(d+1)/2)
        return torch.cat(feats, dim=-1)  # (B, M)


# -----------------------------------------
# 4) Christoffel polynomial model
#    Gamma^i_{jk}(z) = sum_a C[i,j,k,a] * phi_a(z)
#    with torsion-free constraint: Gamma^i_{jk} = Gamma^i_{kj}
# -----------------------------------------

class ChristoffelPoly(nn.Module):
    def __init__(self, d: int, degree: int = 2):
        super().__init__()
        self.d = d
        self.phi = PolyFeatures(d, degree)
        M = self.phi.M

        # Store only upper-triangular (j<=k) coefficients for symmetry
        # C_ut shape: (d, d, d, M) but only meaningful where j<=k
        # Minimal storage trick: store full but symmetrize in forward.
        self.C = nn.Parameter(0.01 * torch.randn(d, d, d, M))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, d)
        returns Gamma: (B, d, d, d) as Gamma[i,j,k] per batch.
        """
        B, d = z.shape
        phi = self.phi(z)  # (B, M)

        # Contract: (d,d,d,M) with (B,M) -> (B,d,d,d)
        Gamma = torch.einsum("ijkm,bm->bijk", self.C, phi)

        # Enforce symmetry in (j,k): torsion-free
        Gamma = 0.5 * (Gamma + Gamma.transpose(2, 3))
        return Gamma


# -----------------------------------------
# 5) Geodesic rollout: ResNet analogue
#    v_{n+1} = v_n + h * a(z_n,v_n)
#    z_{n+1} = z_n + h * v_{n+1}
#    where a^i = -Gamma^i_{jk}(z) v^j v^k
# -----------------------------------------

def geodesic_accel(Gamma: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Gamma: (B, d, d, d) where Gamma[b,i,j,k]
    v:     (B, d)
    returns a: (B, d) with a^i = -Gamma^i_{jk} v^j v^k
    """
    vv = torch.einsum("bj,bk->bjk", v, v)                    # (B, d, d)
    a = -torch.einsum("bijk,bjk->bi", Gamma, vv)             # (B, d)
    return a


def rollout_geodesic_resnet(
    model: ChristoffelPoly,
    z0: torch.Tensor,
    v0: torch.Tensor,
    T: int,
    dt: float,
) -> torch.Tensor:
    """
    Batched rollout.
    z0: (B, d)
    v0: (B, d)
    returns z_pred: (B, T, d)
    """
    z = z0
    v = v0
    traj = [z.unsqueeze(1)]
    for _ in range(1, T):
        Gamma = model(z)                     # (B,d,d,d)
        a = geodesic_accel(Gamma, v)         # (B,d)
        v = v + dt * a
        z = z + dt * v
        traj.append(z.unsqueeze(1))
    return torch.cat(traj, dim=1)            # (B,T,d)


# -----------------------------------------
# 6) Metric reconstruction from Gamma via metric-compatibility
#    ∂_i g = A_i^T g + g A_i,  where (A_i)_{j}^{ℓ} = Gamma^{ℓ}_{i j}
#    Integrate along a path z(s) from z0 -> z1
# -----------------------------------------

@dataclass
class Signature:
    p: int  # number of positive eigenvalues
    q: int  # number of negative eigenvalues


class BaseMetric(nn.Module):
    """
    Learnable base metric g(z0) with fixed signature (p,q) by construction:
      g0 = L diag(signs * exp(scales)) L^T
    where signs has q negatives then p positives.
    """

    def __init__(self, d: int, signature: Signature):
        super().__init__()
        assert signature.p + signature.q == d
        self.d = d
        self.signature = signature

        # lower-triangular unconstrained
        self.L_unconstrained = nn.Parameter(torch.randn(d, d) * 0.01)
        # log-scales
        self.log_scales = nn.Parameter(torch.zeros(d))

        signs = torch.tensor([-1.0] * signature.q + [1.0] * signature.p)
        self.register_buffer("signs", signs)

    def forward(self) -> torch.Tensor:
        L = torch.tril(self.L_unconstrained)
        # encourage diagonal not too small
        diag = torch.exp(self.log_scales)  # positive
        D = torch.diag(self.signs * diag)
        g0 = L @ D @ L.transpose(0, 1)
        # symmetrize for numerical stability
        g0 = 0.5 * (g0 + g0.transpose(0, 1))
        return g0


def metric_ode_step(
    model: ChristoffelPoly,
    z: torch.Tensor,    # (B,d) current point on path
    dz: torch.Tensor,   # (B,d) increment along path
    g: torch.Tensor,    # (B,d,d) current metric
) -> torch.Tensor:
    """
    One Euler step along the path: g <- g + sum_i (A_i^T g + g A_i) dz^i
    """
    B, d = z.shape
    Gamma = model(z)  # (B,d,d,d), Gamma[b,ell,i,j]? Actually Gamma[b,i,j,k] = Gamma^i_{jk}
    # Need A_i with entries A_i[ell, j] = Gamma^ell_{i j}
    # Our Gamma is Gamma^ell_{j k} with indices (ell, j, k) = (i,j,k) in code.
    # So A_i uses lower index = i and second lower index = j:
    # A_i[ell, j] = Gamma^ell_{i j}
    # That means take Gamma[:, ell, i, j] => Gamma[:, :, i, :]
    A = Gamma.transpose(1, 2)  # (B, j, ell, k)?? Let's do explicit gather below for clarity.

    # Build A_i matrices: shape (B, d, d, d) where Ai[b,i,ell,j] = Gamma^ell_{i j}
    # from Gamma[b,ell,i,j]
    Ai = Gamma.permute(0, 2, 1, 3).contiguous()  # (B, i, ell, j)

    # Compute update:
    # dg = sum_i (Ai_i^T g + g Ai_i) * dz_i
    dg = torch.zeros_like(g)
    for i in range(d):
        Ai_i = Ai[:, i, :, :]            # (B, ell, j)
        term = torch.matmul(Ai_i.transpose(1, 2), g) + torch.matmul(g, Ai_i)  # (B, d, d)
        dg = dg + term * dz[:, i].view(B, 1, 1)
    g_next = g + dg
    g_next = 0.5 * (g_next + g_next.transpose(1, 2))  # keep symmetric numerically
    return g_next


def reconstruct_metric_along_straight_path(
    model: ChristoffelPoly,
    z0: torch.Tensor,         # (B,d) basepoint (same for all in batch is OK)
    g0: torch.Tensor,         # (d,d) base metric at z0
    z1: torch.Tensor,         # (B,d) target point
    n_steps: int = 16,
) -> torch.Tensor:
    """
    Reconstruct g(z1) by integrating along straight line z(s) = z0 + s*(z1-z0).
    Returns g1: (B,d,d)
    """
    B, d = z1.shape
    g = g0.unsqueeze(0).expand(B, d, d).contiguous()
    delta = (z1 - z0)  # (B,d)
    for s in range(n_steps):
        # midpoint sampling improves stability slightly
        s0 = (s + 0.5) / n_steps
        z = z0 + s0 * delta
        dz = delta / n_steps
        g = metric_ode_step(model, z, dz, g)
    return g


def reconstruct_metric_two_segment(
    model: ChristoffelPoly,
    z0: torch.Tensor,      # (B,d)
    g0: torch.Tensor,      # (d,d)
    zm: torch.Tensor,      # (B,d) midpoint
    z1: torch.Tensor,      # (B,d) target
    n_steps: int = 16,
) -> torch.Tensor:
    g_m = reconstruct_metric_along_straight_path(model, z0, g0, zm, n_steps=n_steps)
    # Now integrate from zm -> z1, starting at g_m; reuse same stepper
    B, d = z1.shape
    g = g_m
    delta = (z1 - zm)
    for s in range(n_steps):
        s0 = (s + 0.5) / n_steps
        z = zm + s0 * delta
        dz = delta / n_steps
        g = metric_ode_step(model, z, dz, g)
    return g


# -----------------------------------------
# 7) Losses: trajectory + loop consistency + pseudo-Riemannian validity
# -----------------------------------------

def signature_loss(g: torch.Tensor, sig: Signature, beta: float = 10.0) -> torch.Tensor:
    """
    Penalize eigenvalues that violate desired signature.
    g: (B,d,d) assumed symmetric (we symmetrize anyway).
    """
    B, d, _ = g.shape
    g = 0.5 * (g + g.transpose(1, 2))
    eigs = torch.linalg.eigvalsh(g)  # (B,d), sorted ascending
    # desired signs align with sorted eigenvalues: first q negative, last p positive
    target = torch.tensor([-1.0] * sig.q + [1.0] * sig.p, device=g.device).view(1, d)
    # If sign matches, target*eigs is positive; if mismatch, negative -> penalize
    return torch.nn.functional.softplus(-beta * target * eigs).mean()


def logabsdet_loss(g: torch.Tensor, floor: float = -5.0) -> torch.Tensor:
    """
    Penalize near-singular metrics: encourage log|det(g)| >= floor.
    """
    g = 0.5 * (g + g.transpose(1, 2))
    sign, logabsdet = torch.linalg.slogdet(g)  # sign can be +/- for indefinite; 0 if singular
    # singular => sign=0; treat as huge penalty by pushing logabsdet to -inf
    logabsdet = torch.where(sign == 0, torch.full_like(logabsdet, -1e9), logabsdet)
    return torch.nn.functional.softplus(floor - logabsdet).mean()


def symmetry_loss(g: torch.Tensor) -> torch.Tensor:
    return ((g - g.transpose(1, 2)) ** 2).mean()


# -----------------------------------------
# 8) Training loop (CUDA + optional DataParallel)
# -----------------------------------------

def train():
    set_seed(0)
    device = get_device()
    print("Device:", device)
    if device.type == "cuda":
        print("CUDA devices:", torch.cuda.device_count())

    # Hyperparameters
    d = 2
    degree = 2
    T = 50
    dt = 0.05
    batch_size = 256
    epochs = 20
    lr = 2e-3

    # Metric reconstruction settings
    use_metric_losses = True
    metric_steps = 12
    sig = Signature(p=1, q=1)  # Lorentzian signature in 2D (one -, one +) as an example

    # Loss weights
    lam_loop = 0.1
    lam_sig = 0.1
    lam_det = 0.1
    lam_sym = 0.1
    lam_wd = 1e-4

    # Dataset
    ds = PolarInertialDataset(n_traj=8192, T=T, dt=dt, seed=0)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Model
    model = ChristoffelPoly(d=d, degree=degree).to(device)
    base_metric = BaseMetric(d=d, signature=sig).to(device)

    # Minimal multi-GPU support
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        # base_metric is small; leave it on one device (DataParallel won't wrap it)

    params = list(model.parameters()) + list(base_metric.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=0.0)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for ep in range(1, epochs + 1):
        model.train()
        base_metric.train()
        total = 0.0

        for z0, v0, z_true in dl:
            z0 = z0.to(device, non_blocking=True)
            v0 = v0.to(device, non_blocking=True)
            z_true = z_true.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                z_pred = rollout_geodesic_resnet(model, z0, v0, T=T, dt=dt)
                loss_traj = ((z_pred - z_true) ** 2).mean()

                loss = loss_traj

                if use_metric_losses:
                    # Choose basepoint z_base as the batch mean (or a fixed constant)
                    z_base = z0.mean(dim=0, keepdim=True)  # (1,d)
                    g0 = base_metric()                     # (d,d)

                    # Pick a target point z1 (e.g., last point of each trajectory)
                    z1 = z_true[:, -1, :]  # (B,d)

                    # Pick a midpoint zm (e.g., another timepoint)
                    zm = z_true[:, T // 2, :]  # (B,d)

                    gA = reconstruct_metric_along_straight_path(
                        model, z_base, g0, z1, n_steps=metric_steps
                    )
                    gB = reconstruct_metric_two_segment(
                        model, z_base, g0, zm, z1, n_steps=metric_steps
                    )

                    loss_loop = ((gA - gB) ** 2).mean()
                    loss_sig = signature_loss(gA, sig=sig)
                    loss_det = logabsdet_loss(gA, floor=-2.0)
                    loss_sym = symmetry_loss(gA)

                    loss = (
                        loss
                        + lam_loop * loss_loop
                        + lam_sig * loss_sig
                        + lam_det * loss_det
                        + lam_sym * loss_sym
                    )

                # coefficient regularization (explicit; Adam weight_decay off)
                wd = 0.0
                for p in model.parameters():
                    wd = wd + (p ** 2).mean()
                loss = loss + lam_wd * wd

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total += loss.item()

        avg = total / len(dl)
        print(f"Epoch {ep:03d} | loss {avg:.6f}")

    print("Done.")


if __name__ == "__main__":
    # For better CUDA determinism (optional)
    torch.backends.cudnn.benchmark = True
    train()
