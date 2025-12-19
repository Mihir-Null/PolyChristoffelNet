# integrators.py
import torch
from manifold import Manifold

"""
Discrete geodesic "rollout" -> adapted from a ResNet but could probably use a more complex integrator
also has an optional manifold projection onto each step -> might want to see if this can be done without explicit manifold
"""

def geodesic_accel(Gamma: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Gamma: (B,d,d,d) Gamma^i_{jk}
    v:     (B,d)
    a^i = -Gamma^i_{jk} v^j v^k
    """
    vv = torch.einsum("bj,bk->bjk", v, v)         # (B,d,d)
    a = -torch.einsum("bijk,bjk->bi", Gamma, vv)  # (B,d)
    return a

def rollout_geodesic_resnet(
    christoffel_model,
    z0: torch.Tensor,
    v0: torch.Tensor,
    T: int,
    dt: float,
    manifold: Manifold,
    project_each_step: bool = True,
) -> torch.Tensor:
    """
    Batched rollout. Returns z_pred: (B,T,d).
    If manifold != none and project_each_step=True, we reproject (retraction) each step.
    """
    z = manifold.project(z0)
    v = manifold.tangent_project(z, v0)

    traj = [z.unsqueeze(1)]
    for _ in range(1, T):
        Gamma = christoffel_model(z)
        a = geodesic_accel(Gamma, v)
        v = v + dt * a
        z = z + dt * v

        if project_each_step:
            z = manifold.project(z)
            v = manifold.tangent_project(z, v)

        traj.append(z.unsqueeze(1))

    return torch.cat(traj, dim=1)
