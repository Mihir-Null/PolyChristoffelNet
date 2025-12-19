# utils.py
import math
import torch

def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unwrap_angle(theta: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Unwrap phase angles along `dim` to avoid discontinuities at +/- pi.
    theta: (..., T, ...) typically with T along dim.
    """
    # Compute incremental diffs
    dtheta = torch.diff(theta, dim=dim)
    # Wrap diffs to [-pi, pi]
    dtheta_wrapped = (dtheta + math.pi) % (2 * math.pi) - math.pi
    # Correction term
    corr = dtheta_wrapped - dtheta
    # Ignore small corrections (numerical noise)
    corr = torch.where(torch.abs(dtheta) < math.pi, torch.zeros_like(corr), corr)
    # Cumulative sum of corrections
    shape = [1] * theta.ndim
    shape[dim] = theta.size(dim)
    pad = torch.zeros_like(theta.select(dim, 0)).unsqueeze(dim)  # zeros at start
    corr_cum = torch.cumsum(corr, dim=dim)
    theta_unwrapped = theta + torch.cat([pad, corr_cum], dim=dim)
    return theta_unwrapped
