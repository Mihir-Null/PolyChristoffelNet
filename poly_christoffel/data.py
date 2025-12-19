# data.py
import math
import torch
from torch.utils.data import Dataset
from utils import unwrap_angle

#Synthetic Dataset Generator
class PolarInertialDataset(Dataset):
    """
    Cartesian straight-line motion, represented in polar coordinates z=(r, theta).
    Returns full sequence x_seq (here x_seq=z_seq), and dt.
    """

    def __init__(
        self,
        n_traj: int,
        T: int,
        dt: float,
        r_min: float = 0.5,
        r_max: float = 2.0,
        v_scale: float = 1.0,
        seed: int = 0,
    ):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.T = T
        self.dt = dt

        ang0 = 2 * math.pi * torch.rand(n_traj, generator=g)
        rad0 = r_min + (r_max - r_min) * torch.rand(n_traj, generator=g)
        x0 = rad0 * torch.cos(ang0)
        y0 = rad0 * torch.sin(ang0)

        vx = v_scale * torch.randn(n_traj, generator=g)
        vy = v_scale * torch.randn(n_traj, generator=g)

        t = torch.arange(T, dtype=torch.float32) * dt  # (T,)
        t = t[None, :]  # (1,T)

        x = x0[:, None] + vx[:, None] * t
        y = y0[:, None] + vy[:, None] * t

        r = torch.sqrt(x * x + y * y)
        theta = torch.atan2(y, x)
        # theta = unwrap_angle(theta, dim=1) -> making training unstable, check math later

        z = torch.stack([r, theta], dim=-1)  # (N,T,2)
        self.x_seq = z

    def __len__(self):
        return self.x_seq.shape[0]

    def __getitem__(self, idx: int):
        # x_seq: (T,d)
        return self.x_seq[idx], self.dt
