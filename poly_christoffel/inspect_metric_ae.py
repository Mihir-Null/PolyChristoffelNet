# ae_metric_diagnostics.py
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

    # Model structure (must match training)
    p.add_argument("--use_ae", type=int, default=1)
    p.add_argument("--manifold", type=str, default="none", choices=["none", "sphere", "hyperboloid"])
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--poly_degree", type=int, default=2)
    p.add_argument("--p_sig", type=int, default=2)
    p.add_argument("--q_sig", type=int, default=0)

    # Observation grid y=(x,y) for diagnostics
    p.add_argument("--x_min", type=float, default=-2.0)
    p.add_argument("--x_max", type=float, default=2.0)
    p.add_argument("--y_min", type=float, default=-2.0)
    p.add_argument("--y_max", type=float, default=2.0)
    p.add_argument("--nx", type=int, default=120)
    p.add_argument("--ny", type=int, default=120)
    p.add_argument("--r_min", type=float, default=0.4, help="Exclude near-origin points to avoid polar singularities")

    # Metric reconstruction settings (in latent)
    p.add_argument("--metric_steps", type=int, default=16)

    # Basepoint specified in observation space; basepoint in latent is z_base = E(y_base)
    p.add_argument("--base_x", type=float, default=1.0)
    p.add_argument("--base_y", type=float, default=0.0)
    p.add_argument("--use_gt_base", type=int, default=0,
                   help="If 1: set g0 to latent polar GT metric at z_base (only meaningful if z ~ (r,theta)). "
                        "If 0: use learned base_metric().")

    # Optional loop/path dependence in latent and pulled-back space
    p.add_argument("--use_loop", type=int, default=1)
    p.add_argument("--mid_x", type=float, default=1.2)
    p.add_argument("--mid_y", type=float, default=0.7)

    # Jacobian computation
    p.add_argument("--chunk", type=int, default=2048, help="Batch chunk size for Jacobian computations")

    # Output
    p.add_argument("--save_fig", type=str, default="", help="Optional path to save figure, e.g. figs/ae_metric.png")
    return p.parse_args()


def polar_metric_latent(r: torch.Tensor) -> torch.Tensor:
    """
    Ground-truth polar metric on Euclidean plane in coordinates z=(r,theta):
      g = diag(1, r^2)
    r: (N,)
    returns: (N,2,2)
    """
    g = torch.zeros(r.shape[0], 2, 2, device=r.device, dtype=r.dtype)
    g[:, 0, 0] = 1.0
    g[:, 1, 1] = r * r
    return g


def normalize_metric_to_unit_trace(g: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize a 2x2 metric by its trace (removes a scale factor):
      g_norm = g / (tr(g)/2)
    g: (N,2,2)
    """
    tr = (g[:, 0, 0] + g[:, 1, 1]).clamp_min(eps)  # (N,)
    scale = (tr / 2.0).view(-1, 1, 1)
    return g / scale


def encode_points(model: FullGeoModel, y: torch.Tensor) -> torch.Tensor:
    """
    y: (B,2)
    returns z: (B,d)
    """
    # Use the AE encoder when present; otherwise fall back to identity.
    if hasattr(model, "ae") and model.ae is not None:
        z = model.ae.encode(y)
    else:
        z = y
    # Old encoder path (kept for easy reversion):
    # if hasattr(model, "encoder") and model.encoder is not None:
    #     z = model.encoder(y)
    # If manifold constraint exists, keep the same convention as training (encode then project)
    if hasattr(model, "manifold") and model.manifold is not None:
        z = model.manifold.project(z)
    return z


def encoder_jacobian(model: FullGeoModel, y: torch.Tensor) -> torch.Tensor:
    """
    Compute J = dE/dy for batch y.
    y: (B,2)
    returns J: (B,d,2)
    Uses torch.func.vmap + jacrev when available; falls back to loop.
    """
    # Preferred (fast) path: torch.func
    try:
        from torch.func import vmap, jacrev

        def enc_single(y_single):
            # y_single: (2,)
            z_single = encode_points(model, y_single.unsqueeze(0)).squeeze(0)  # (d,)
            return z_single

        J = vmap(jacrev(enc_single))(y)  # (B,d,2)
        return J
    except Exception:
        # Fallback: per-sample loop with autograd
        B = y.shape[0]
        d = model.d if hasattr(model, "d") else 2
        J = torch.zeros(B, d, 2, device=y.device, dtype=y.dtype)

        y_req = y.detach().clone().requires_grad_(True)
        z = encode_points(model, y_req)  # (B,d)
        for i in range(d):
            grads = torch.autograd.grad(
                z[:, i].sum(), y_req, create_graph=False, retain_graph=True, allow_unused=False
            )[0]  # (B,2)
            J[:, i, :] = grads
        return J


# @torch.no_grad()  # old behavior (disabled autograd Jacobians)
def main():
    args = parse_args()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    cfg = TrainConfig()
    cfg.use_ae = bool(args.use_ae)
    cfg.manifold = args.manifold
    cfg.d = args.d
    cfg.poly_degree = args.poly_degree
    cfg.signature = Signature(p=args.p_sig, q=args.q_sig)

    # Observation dimension for AE diagnostic: assume y is 2D coordinates (x,y).
    # If your observations are higher-dimensional (images), this diagnostic is not meaningful.
    x_dim = 2

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

    # Build observation grid in Cartesian y-space
    xs = torch.linspace(args.x_min, args.x_max, args.nx, device=device)
    ys = torch.linspace(args.y_min, args.y_max, args.ny, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")  # (nx,ny)

    y_grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)  # (N,2)
    r = torch.sqrt(y_grid[:, 0] ** 2 + y_grid[:, 1] ** 2)
    mask = r >= args.r_min
    y_grid = y_grid[mask]
    N = y_grid.shape[0]

    print(f"Grid points kept (r >= {args.r_min}): N={N}")

    # Basepoint in observation space, then encode to latent basepoint
    y_base = torch.tensor([[args.base_x, args.base_y]], device=device, dtype=torch.float32)
    z_base = encode_points(model, y_base)  # (1,d)

    if args.use_gt_base:
        if args.d != 2:
            raise ValueError("use_gt_base=1 assumes latent is z=(r,theta) with d=2.")
        g0 = polar_metric_latent(z_base[:, 0])[0]  # uses r component
        print("Using GT polar base metric at z_base (assumes latent coords ~ (r,theta)).")
    else:
        g0 = model.base_metric().detach()
        print("Using learned base_metric() as g0.")

    # We'll compute pulled-back metric g_y(y) = J^T g_z(z) J in chunks
    gyy_11 = []
    gyy_12 = []
    gyy_22 = []
    mse_to_I = []
    mse_to_I_norm = []
    offdiag_abs = []
    eig_min = []
    frac_neg = []

    # Optional loop error
    loop_vals = []

    # Midpoint (for loop) in observation space -> latent midpoint
    if args.use_loop:
        y_mid = torch.tensor([[args.mid_x, args.mid_y]], device=device, dtype=torch.float32)
        z_mid_single = encode_points(model, y_mid)  # (1,d)

    # Identity metric in y-space
    I = torch.eye(2, device=device, dtype=torch.float32).unsqueeze(0)  # (1,2,2)

    # Need grads for Jacobian; compute Jacobian chunks with autograd (outside no_grad)
    # We'll temporarily enable grad inside the loop.
    for start in range(0, N, args.chunk):
        end = min(N, start + args.chunk)
        y_chunk = y_grid[start:end].detach().clone().requires_grad_(True)  # (B,2)
        B = y_chunk.shape[0]

        # Encode
        z_chunk = encode_points(model, y_chunk)  # (B,d)

        # Reconstruct latent metric along straight path from z_base to z_chunk
        # (metric recon uses only Γ and integrates compatibility ODE)
        gA = reconstruct_metric_straight_path(christoffel, z_base, g0, z_chunk, n_steps=args.metric_steps)  # (B,d,d)
        gA = 0.5 * (gA + gA.transpose(1, 2))

        # Jacobian J = dE/dy
        # Use a separate function that may use torch.func or autograd; requires grads.
        J = encoder_jacobian(model, y_chunk)  # (B,d,2)

        # Pullback to observation space: g_y = J^T g_z J
        # (B,2,d) @ (B,d,d) @ (B,d,2) -> (B,2,2)
        JT = J.transpose(1, 2)  # (B,2,d)
        g_y = JT @ gA @ J  # (B,2,2)
        g_y = 0.5 * (g_y + g_y.transpose(1, 2))

        # Compare to Euclidean metric I (optionally scale-normalized)
        # Raw MSE (includes scale)
        mse_raw = torch.mean((g_y - I) ** 2, dim=(1, 2))  # (B,)

        # Scale-invariant comparison: normalize by trace so ideal is I
        g_y_norm = normalize_metric_to_unit_trace(g_y)
        mse_norm = torch.mean((g_y_norm - I) ** 2, dim=(1, 2))

        # Basic stats
        off = torch.abs(g_y[:, 0, 1])
        eigs = torch.linalg.eigvalsh(g_y)  # (B,2)

        gyy_11.append(g_y[:, 0, 0].detach())
        gyy_12.append(g_y[:, 0, 1].detach())
        gyy_22.append(g_y[:, 1, 1].detach())
        mse_to_I.append(mse_raw.detach())
        mse_to_I_norm.append(mse_norm.detach())
        offdiag_abs.append(off.detach())
        eig_min.append(eigs.min(dim=1).values.detach())
        frac_neg.append((eigs < 0).float().mean(dim=1).detach())

        # Loop / path dependence (latent), then also pull back and compare if desired
        if args.use_loop:
            z_mid = z_mid_single.expand(B, -1)  # (B,d)
            gB = reconstruct_metric_two_segment(christoffel, z_base, g0, z_mid, z_chunk, n_steps=args.metric_steps)
            gB = 0.5 * (gB + gB.transpose(1, 2))
            loop_latent = torch.mean((gA - gB) ** 2, dim=(1, 2))  # (B,)
            loop_vals.append(loop_latent.detach())

    # Concatenate
    g11 = torch.cat(gyy_11, dim=0).cpu().numpy()
    g12 = torch.cat(gyy_12, dim=0).cpu().numpy()
    g22 = torch.cat(gyy_22, dim=0).cpu().numpy()
    mse_raw = torch.cat(mse_to_I, dim=0).cpu().numpy()
    mse_norm = torch.cat(mse_to_I_norm, dim=0).cpu().numpy()
    off = torch.cat(offdiag_abs, dim=0).cpu().numpy()
    emin = torch.cat(eig_min, dim=0).cpu().numpy()
    fneg = torch.cat(frac_neg, dim=0).cpu().numpy()

    loop_latent = torch.cat(loop_vals, dim=0).cpu().numpy() if args.use_loop else None

    print("\n=== AE Metric Diagnostics (pullback to observation space) ===")
    print("Target in observation space (Cartesian y): Euclidean metric I.")
    print(f"  mean|g_y12|           = {off.mean():.6e}")
    print(f"  mean MSE(g_y vs I)    = {mse_raw.mean():.6e}  (scale-sensitive)")
    print(f"  mean MSE(g_y_norm vs I)= {mse_norm.mean():.6e}  (scale-invariant, normalized by trace)")
    print(f"  min eigenvalue mean   = {emin.mean():.6e} (should be > 0 for SPD pullback)")
    print(f"  frac negative eig mean= {fneg.mean():.6e}")
    if args.use_loop:
        print(f"  latent loop MSE mean  = {loop_latent.mean():.6e} (path dependence in reconstructed g_z)")

    # For plotting, we need to map values back to the (nx,ny) grid with mask.
    # We'll create full arrays with NaNs for masked-out points.
    full_shape = (args.nx, args.ny)
    def fill_full(vals_1d):
        full = torch.full((args.nx * args.ny,), float("nan"))
        full[mask.cpu()] = torch.tensor(vals_1d)
        return full.view(*full_shape).numpy()

    g11_full = fill_full(g11)
    g12_full = fill_full(g12)
    g22_full = fill_full(g22)
    mse_norm_full = fill_full(mse_norm)
    loop_full = fill_full(loop_latent) if args.use_loop else None

    # Plot
    cols = 3 if args.use_loop else 2
    fig = plt.figure(figsize=(14, 8 if args.use_loop else 6))

    ax1 = fig.add_subplot(2, cols, 1)
    im1 = ax1.imshow(g11_full, aspect="auto", origin="lower")
    ax1.set_title(r"Pulled-back $g_{xx}(x,y)$")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, cols, 2)
    im2 = ax2.imshow(g12_full, aspect="auto", origin="lower")
    ax2.set_title(r"Pulled-back $g_{xy}(x,y)$")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    if args.use_loop:
        ax3 = fig.add_subplot(2, cols, 3)
        im3 = ax3.imshow(loop_full, aspect="auto", origin="lower")
        ax3.set_title(r"Latent loop error $\|g^{(A)}-g^{(B)}\|^2$")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, cols, cols + 1)
    im4 = ax4.imshow(g22_full, aspect="auto", origin="lower")
    ax4.set_title(r"Pulled-back $g_{yy}(x,y)$")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(2, cols, cols + 2)
    im5 = ax5.imshow(mse_norm_full, aspect="auto", origin="lower")
    ax5.set_title(r"Scale-invariant error: $\|g_{\mathrm{norm}}-I\|^2$")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    if args.use_loop:
        ax6 = fig.add_subplot(2, cols, cols + 3)
        # just repeat error map for layout symmetry
        im6 = ax6.imshow(mse_norm_full, aspect="auto", origin="lower")
        ax6.set_title(r"(same) $\|g_{\mathrm{norm}}-I\|^2$")
        fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    for ax in fig.axes:
        if hasattr(ax, "set_xlabel"):
            ax.set_xlabel("y-index")
            ax.set_ylabel("x-index")

    fig.tight_layout()

    if args.save_fig:
        os.makedirs(os.path.dirname(args.save_fig), exist_ok=True)
        fig.savefig(args.save_fig, dpi=200)
        print(f"Saved figure to: {args.save_fig}")
    else:
        plt.show()


if __name__ == "__main__":
    # This script uses autograd Jacobians; do not wrap main() in no_grad.
    main()
