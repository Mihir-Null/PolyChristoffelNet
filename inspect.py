# inspect.py
import argparse
import math
import os
from dataclasses import asdict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import TrainConfig, Signature
from model import FullGeoModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/latest.pt", help="Path to checkpoint .pt")
    p.add_argument("--use_ae", type=int, default=0, help="Must match training (0/1)")
    p.add_argument("--manifold", type=str, default="none", choices=["none", "sphere", "hyperboloid"])
    p.add_argument("--p", type=int, default=2)
    p.add_argument("--q", type=int, default=0)

    p.add_argument("--d", type=int, default=2, help="Latent/state dimension d (polar demo uses d=2)")
    p.add_argument("--poly_degree", type=int, default=2)

    p.add_argument("--r_min", type=float, default=0.5)
    p.add_argument("--r_max", type=float, default=2.0)
    p.add_argument("--nr", type=int, default=80)
    p.add_argument("--ntheta", type=int, default=120)

    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--save_fig", type=str, default="", help="Optional: path to save figure (e.g. figs/gamma.png)")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    # Build model with the same structural config used in training
    cfg = TrainConfig()
    cfg.use_ae = bool(args.use_ae)
    cfg.manifold = args.manifold
    cfg.d = args.d
    cfg.poly_degree = args.poly_degree
    cfg.signature = Signature(p=args.p, q=args.q)

    # For the polar demo, x_dim==d. If you later use real observations, x_dim changes.
    x_dim = cfg.d

    model = FullGeoModel(
        x_dim=x_dim,
        d=cfg.d,
        poly_degree=cfg.poly_degree,
        use_ae=cfg.use_ae,
        ae_hidden=cfg.ae_hidden,
        ae_depth=cfg.ae_depth,
        manifold_kind=cfg.manifold,
        signature=cfg.signature,
    ).to(device)
    model.eval()

    # Load checkpoint
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {args.ckpt}")

    # Only inspect the Christoffel part (in polar demo, z=(r,theta))
    christoffel = model.christoffel

    # Create grid in (r, theta)
    r = torch.linspace(args.r_min, args.r_max, args.nr, device=device)
    theta = torch.linspace(-math.pi, math.pi, args.ntheta, device=device)
    R, TH = torch.meshgrid(r, theta, indexing="ij")  # (nr, ntheta)

    z = torch.stack([R.reshape(-1), TH.reshape(-1)], dim=-1)  # (N,2)
    Gamma = christoffel(z)  # (N,d,d,d), Gamma^i_{jk}

    # Indices: 0=r, 1=theta
    Gamma_r_thetatheta = Gamma[:, 0, 1, 1].reshape(args.nr, args.ntheta)
    Gamma_theta_rtheta = Gamma[:, 1, 0, 1].reshape(args.nr, args.ntheta)
    Gamma_theta_thetar = Gamma[:, 1, 1, 0].reshape(args.nr, args.ntheta)

    # Ground truth for polar coords on Euclidean plane:
    # Gamma^r_{θθ} = -r
    # Gamma^θ_{rθ} = Gamma^θ_{θr} = 1/r
    GT_r_thetatheta = (-R).detach()
    GT_theta_rtheta = (1.0 / R).detach()

    # Quantitative comparisons
    mse_r = torch.mean((Gamma_r_thetatheta - GT_r_thetatheta) ** 2).item()
    mse_tr = torch.mean((Gamma_theta_rtheta - GT_theta_rtheta) ** 2).item()
    mse_tsym = torch.mean((Gamma_theta_rtheta - Gamma_theta_thetar) ** 2).item()

    print("Diagnostics (averaged over grid):")
    print(f"  MSE[Gamma^r_{'{'}θθ{'}'}  vs  -r]     = {mse_r:.6e}")
    print(f"  MSE[Gamma^θ_{'{'}rθ{'}'}  vs  1/r]    = {mse_tr:.6e}")
    print(f"  MSE[Gamma^θ_{'{'}rθ{'}'}  vs  Gamma^θ_{'{'}θr{'}'}] = {mse_tsym:.6e}  (torsion-free symmetry check)")

    # Plot fields (no explicit colors set)
    fig = plt.figure(figsize=(14, 9))

    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(Gamma_r_thetatheta.detach().cpu().numpy(), aspect="auto", origin="lower")
    ax1.set_title(r"Learned $\Gamma^r_{\theta\theta}(r,\theta)$")
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$r$")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(GT_r_thetatheta.detach().cpu().numpy(), aspect="auto", origin="lower")
    ax2.set_title(r"Ground truth $-r$")
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel(r"$r$")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(Gamma_theta_rtheta.detach().cpu().numpy(), aspect="auto", origin="lower")
    ax3.set_title(r"Learned $\Gamma^\theta_{r\theta}(r,\theta)$")
    ax3.set_xlabel(r"$\theta$")
    ax3.set_ylabel(r"$r$")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, 2, 4)
    im4 = ax4.imshow(GT_theta_rtheta.detach().cpu().numpy(), aspect="auto", origin="lower")
    ax4.set_title(r"Ground truth $1/r$")
    ax4.set_xlabel(r"$\theta$")
    ax4.set_ylabel(r"$r$")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    fig.tight_layout()

    if args.save_fig:
        os.makedirs(os.path.dirname(args.save_fig), exist_ok=True)
        fig.savefig(args.save_fig, dpi=200)
        print(f"Saved figure to: {args.save_fig}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
