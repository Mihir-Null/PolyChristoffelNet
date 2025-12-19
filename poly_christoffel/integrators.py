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
    z0, v0, T, dt,
    manifold,
    project_each_step=True,
    substeps: int = 4,
    vmax: float = 50.0,
    gamma_max: float = 50.0,
):
    z = manifold.project(z0)
    v = manifold.tangent_project(z, v0)

    traj = [z.unsqueeze(1)]
    h = dt / substeps

    for _ in range(1, T):
        for _s in range(substeps):
            Gamma = christoffel_model(z)
            # keep Christoffels from exploding early in training
            Gamma = torch.clamp(Gamma, -gamma_max, gamma_max)

            a = geodesic_accel(Gamma, v)
            v = v + h * a
            v = torch.clamp(v, -vmax, vmax)

            z = z + h * v

            if project_each_step:
                z = manifold.project(z)
                v = manifold.tangent_project(z, v)

        traj.append(z.unsqueeze(1))

    return torch.cat(traj, dim=1)

