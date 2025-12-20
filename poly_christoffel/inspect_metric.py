# metric_diagnostics.py
import argparse
import math
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import TrainConfig, Signature
from model import FullGeoModel
from metric_recon import reconstruct_metric_straight_path, reconstruct_metric_two_segment


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/latest.pt")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # Must match the trained model structure
    p.add_argument("--use_ae", type=int, default=0)
    p.add_argument("--manifold", type=str, default="none", choices=["none", "sphere", "hyperboloid"])
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--poly_degree", type=int, default=2)
    p.add_argument("--p_sig", type=int, default=2)
    p.add_argument("--q_sig", type=int, default=0)

    # Grid for evaluation (polar demo uses x=(r,theta))
    p.add_argument("--r_min", type=float, default=0.5)
    p.add_argument("--r_max", type=float, default=2.0)
    p.add_argument("--nr", type=int, default=80)
    p.add_argument("--ntheta", type=int, default=120)

    # Metric reconstruction controls
    p.add_argument("--metric_steps", type=int, default=16)
    p.add_argument("--base_r", type=float, default=1.0)
    p.add_argument("--base_theta", type=float, default=0.0)
    p.add_argument("--use_gt_base", type=int, default=0,
                   help="If 1: set g0 to ground-truth polar metric at basepoint; "
                        "If 0: use learned base_metric() from checkpoint.")

    # Loop error map
    p.add_argument("--use_loop", type=int, default=1)
    p.add_argument("--mid_r", type=float, default=1.25)
    p.add_argument("--mid_theta", type=float, default=0.7)

    # Output
    p.add_argument("--save_fig", type=str, default="", help="Optional path to save figure, e.g. figs/metric.png")
    return p.parse_args()


def ground_truth_polar_metric(r: torch.Tensor) -> torch.Tensor:
    """
    Polar metric for Euclidean plane in coordinates (r, theta):
      g = [[1, 0],
           [0, r^2]]
    r: (...,)
    returns: (..., 2, 2)
    """
    g = torch.zeros(*r.shape, 2, 2, device=r.device, dtype=r.dtype)
    g[..., 0, 0] = 1.0
    g[..., 1, 1] = r * r
    return g


def best_fit_scale(g: torch.Tensor, g_gt: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Find scalar alpha minimizing ||g - alpha*g_gt||_F^2 per-sample.
    alpha = <g, g_gt> / <g_gt, g_gt>
    g, g_gt: (N,2,2)
    returns alpha: (N,1,1)
    """
    num = torch.sum(g * g_gt, dim=(1, 2))
    den = torch.sum(g_gt * g_gt, dim=(1, 2)).clamp_min(eps)
    alpha = (num / den).view(-1, 1, 1)
    return alpha


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    if args.d != 2:
        raise ValueError("This diagnostic currently assumes polar state x=(r,theta) with d=2.")

    if args.use_ae != 0:
        print("[Warn] use_ae=1 means x is latent. Ground-truth polar metric comparison may not be meaningful.")

    # Build model (structure must match training)
    cfg = TrainConfig()
    cfg.use_ae = bool(args.use_ae)
    cfg.manifold = args.manifold
    cfg.d = args.d
    cfg.poly_degree = args.poly_degree
    cfg.signature = Signature(p=args.p_sig, q=args.q_sig)

    x_dim = cfg.d  # polar demo

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

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {args.ckpt}")

    christoffel = model.christoffel

    # Basepoint and base metric
    z_base = torch.tensor([[args.base_r, args.base_theta]], device=device, dtype=torch.float32)  # (1,2)

    if args.use_gt_base:
        g0 = ground_truth_polar_metric(torch.tensor([args.base_r], device=device, dtype=torch.float32))[0]
        print("Using ground-truth base metric g0 at basepoint.")
    else:
        g0 = model.base_metric().detach()
        print("Using learned base_metric() as g0.")

    # Construct grid
    r = torch.linspace(args.r_min, args.r_max, args.nr, device=device)
    theta = torch.linspace(-math.pi, math.pi, args.ntheta, device=device)
    R, TH = torch.meshgrid(r, theta, indexing="ij")
    z = torch.stack([R.reshape(-1), TH.reshape(-1)], dim=-1)  # (N,2)
    N = z.shape[0]

    # Reconstruct g via straight path from basepoint
    gA = reconstruct_metric_straight_path(christoffel, z_base, g0, z, n_steps=args.metric_steps)  # (N,2,2)
    gA = 0.5 * (gA + gA.transpose(1, 2))

    # Ground truth metric on grid
    gGT = ground_truth_polar_metric(z[:, 0])  # depends only on r

    # Fit a single global scale alpha to account for constant scale ambiguity (and imperfect g0).
    # This avoids per-point rescaling that can make GT plots track the learned metric too closely.
    num = torch.sum(gA * gGT)
    den = torch.sum(gGT * gGT).clamp_min(1e-12)
    alpha = (num / den).view(1, 1, 1)
    gGT_scaled = alpha * gGT
    # Old per-point scaling (kept for easy reversion):
    # alpha = best_fit_scale(gA, gGT)  # (N,1,1)
    # gGT_scaled = alpha * gGT

    # Errors
    offdiag = gA[:, 0, 1]
    mse_scaled = torch.mean((gA - gGT_scaled) ** 2).item()
    mean_abs_offdiag = torch.mean(torch.abs(offdiag)).item()

    # Componentwise MSE (scaled GT)
    mse_rr = torch.mean((gA[:, 0, 0] - gGT_scaled[:, 0, 0]) ** 2).item()
    mse_tt = torch.mean((gA[:, 1, 1] - gGT_scaled[:, 1, 1]) ** 2).item()

    # Signature / positivity check (SPD expected for polar metric)
    eigs = torch.linalg.eigvalsh(gA)  # (N,2)
    min_eig = torch.min(eigs).item()
    frac_neg = torch.mean((eigs < 0).float()).item()

    print("\n=== Metric Diagnostics (polar GT: diag(1, r^2)) ===")
    print(f"  gA shape: {tuple(gA.shape)}  (N={N})")
    print(f"  mean|gA_rθ| (offdiag): {mean_abs_offdiag:.6e}")
    print(f"  MSE(gA vs scaled GT): {mse_scaled:.6e}")
    print(f"    MSE_rr: {mse_rr:.6e}")
    print(f"    MSE_θθ: {mse_tt:.6e}")
    print(f"  eig(gA): min eigenvalue = {min_eig:.6e}, fraction negative eigs = {frac_neg:.6f}")
    print("  Note: scaled GT uses per-point alpha=<gA,gGT>/<gGT,gGT> to handle scale mismatch.")

    # Optional loop/path dependence
    loop_map = None
    loop_mean = None
    if args.use_loop:
        z_mid = torch.tensor([[args.mid_r, args.mid_theta]], device=device, dtype=torch.float32).expand(N, 2)
        gB = reconstruct_metric_two_segment(christoffel, z_base, g0, z_mid, z, n_steps=args.metric_steps)
        gB = 0.5 * (gB + gB.transpose(1, 2))
        loop = torch.mean((gA - gB) ** 2, dim=(1, 2))  # (N,)
        loop_mean = loop.mean().item()
        loop_map = loop.reshape(args.nr, args.ntheta).detach().cpu().numpy()
        print(f"  Loop MSE mean (path dependence): {loop_mean:.6e}")

    # Reshape for plotting
    g_rr = gA[:, 0, 0].reshape(args.nr, args.ntheta).detach().cpu().numpy()
    g_rth = gA[:, 0, 1].reshape(args.nr, args.ntheta).detach().cpu().numpy()
    g_thth = gA[:, 1, 1].reshape(args.nr, args.ntheta).detach().cpu().numpy()

    gt_rr = gGT_scaled[:, 0, 0].reshape(args.nr, args.ntheta).detach().cpu().numpy()
    gt_thth = gGT_scaled[:, 1, 1].reshape(args.nr, args.ntheta).detach().cpu().numpy()

    # Plot: learned g components + scaled GT + (optional) loop map
    cols = 3 if args.use_loop else 2
    fig = plt.figure(figsize=(14, 8 if args.use_loop else 6))

    ax1 = fig.add_subplot(2, cols, 1)
    im1 = ax1.imshow(g_rr, aspect="auto", origin="lower")
    ax1.set_title(r"Learned $g_{rr}(r,\theta)$")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, cols, 2)
    im2 = ax2.imshow(g_rth, aspect="auto", origin="lower")
    ax2.set_title(r"Learned $g_{r\theta}(r,\theta)$")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    if args.use_loop:
        ax3 = fig.add_subplot(2, cols, 3)
        im3 = ax3.imshow(loop_map, aspect="auto", origin="lower")
        ax3.set_title(r"Loop error map $\|g^{(A)}-g^{(B)}\|^2$")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, cols, cols + 1)
    im4 = ax4.imshow(g_thth, aspect="auto", origin="lower")
    ax4.set_title(r"Learned $g_{\theta\theta}(r,\theta)$")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(2, cols, cols + 2)
    im5 = ax5.imshow(gt_rr, aspect="auto", origin="lower")
    ax5.set_title(r"Scaled GT $g_{rr}=1$")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    if args.use_loop:
        ax6 = fig.add_subplot(2, cols, cols + 3)
        im6 = ax6.imshow(gt_thth, aspect="auto", origin="lower")
        ax6.set_title(r"Scaled GT $g_{\theta\theta}=r^2$")
        fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    else:
        ax6 = fig.add_subplot(2, cols, cols + 2)
        im6 = ax6.imshow(gt_thth, aspect="auto", origin="lower")
        ax6.set_title(r"Scaled GT $g_{\theta\theta}=r^2$")
        fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    for ax in fig.axes:
        if hasattr(ax, "set_xlabel"):
            ax.set_xlabel(r"$\theta$ index")
            ax.set_ylabel(r"$r$ index")

    fig.tight_layout()

    if args.save_fig:
        os.makedirs(os.path.dirname(args.save_fig), exist_ok=True)
        fig.savefig(args.save_fig, dpi=200)
        print(f"Saved figure to: {args.save_fig}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
