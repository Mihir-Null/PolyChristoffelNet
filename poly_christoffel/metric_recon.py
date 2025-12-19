# metric_recon.py
import torch

"""
Reconstruct g(z) from the christoffels - uses metric coompatibility path integration + two path consistency
-> could also try using small loop consistency/tangent spaceness as addn signal
"""

def metric_ode_step(christoffel_model, z: torch.Tensor, dz: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """
    Euler step for metric compatibility along a path:
      dg = sum_i (A_i^T g + g A_i) dz^i
    where (A_i)_{ell,j} = Gamma^ell_{i j}.
    """
    B, d = z.shape
    Gamma = christoffel_model(z)  # (B, ell, j, k)?? actually (B, i, j, k) with i as upper index
    # Our Gamma[b,ell,i,j] is Gamma^ell_{i j} after permute:
    Ai = Gamma.permute(0, 2, 1, 3).contiguous()  # (B, i, ell, j)

    dg = torch.zeros_like(g)
    for i in range(d):
        Ai_i = Ai[:, i, :, :]  # (B, ell, j)
        term = Ai_i.transpose(1, 2) @ g + g @ Ai_i  # (B,d,d)
        dg = dg + term * dz[:, i].view(B, 1, 1)

    g_next = g + dg
    g_next = 0.5 * (g_next + g_next.transpose(1, 2))
    return g_next

def reconstruct_metric_straight_path(
    christoffel_model,
    z0: torch.Tensor,     # (1,d) or (B,d)
    g0: torch.Tensor,     # (d,d)
    z1: torch.Tensor,     # (B,d)
    n_steps: int = 16,
) -> torch.Tensor:
    """
    Integrate along z(s)=z0 + s(z1-z0) for s in [0,1].
    Returns g(z1): (B,d,d)
    """
    B, d = z1.shape
    if z0.shape[0] == 1:
        z0b = z0.expand(B, d)
    else:
        z0b = z0

    g = g0.unsqueeze(0).expand(B, d, d).contiguous()
    delta = (z1 - z0b)
    dz = delta / n_steps

    for s in range(n_steps):
        s_mid = (s + 0.5) / n_steps
        z = z0b + s_mid * delta
        g = metric_ode_step(christoffel_model, z, dz, g)
    return g

def reconstruct_metric_two_segment(
    christoffel_model,
    z0: torch.Tensor,   # (1,d) or (B,d)
    g0: torch.Tensor,   # (d,d)
    zm: torch.Tensor,   # (B,d)
    z1: torch.Tensor,   # (B,d)
    n_steps: int = 16,
) -> torch.Tensor:
    g_m = reconstruct_metric_straight_path(christoffel_model, z0, g0, zm, n_steps=n_steps)
    # second segment zm->z1
    B, d = z1.shape
    g = g_m
    delta = (z1 - zm)
    dz = delta / n_steps
    for s in range(n_steps):
        s_mid = (s + 0.5) / n_steps
        z = zm + s_mid * delta
        g = metric_ode_step(christoffel_model, z, dz, g)
    return g
