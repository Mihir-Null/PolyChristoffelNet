import argparse
import json
import math
import os
from typing import Dict

import torch
import torch.nn as nn

from config import TrainConfig, Signature
from data import PolarInertialDataset
from losses import symmetry_loss, logabsdet_loss, signature_loss
from metric_recon import reconstruct_metric_straight_path, reconstruct_metric_two_segment
from model import FullGeoModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/latest.pt")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # Model structure (must match training)
    p.add_argument("--use_ae", type=int, default=0)
    p.add_argument("--manifold", type=str, default="none", choices=["none", "sphere", "hyperboloid"])
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--poly_degree", type=int, default=2)
    p.add_argument("--p_sig", type=int, default=2)
    p.add_argument("--q_sig", type=int, default=0)

    # Evaluation controls
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_batches", type=int, default=16)
    p.add_argument("--seed", type=int, default=1)

    # Metric eval
    p.add_argument("--use_metric_eval", type=int, default=1)
    p.add_argument("--metric_steps", type=int, default=12)
    p.add_argument("--use_loop", type=int, default=1)

    # Output
    p.add_argument("--save_json", type=str, default="")
    return p.parse_args()


def polar_gt_gamma(z: torch.Tensor) -> torch.Tensor:
    """
    Ground-truth Christoffels for Euclidean plane in polar coordinates.
    z: (B,2) where z = (r, theta)
    returns Gamma_gt: (B,2,2,2)
    """
    r = z[:, 0].clamp_min(1e-6)
    B = z.shape[0]
    Gamma = torch.zeros(B, 2, 2, 2, device=z.device, dtype=z.dtype)
    # Gamma^r_{theta theta} = -r
    Gamma[:, 0, 1, 1] = -r
    # Gamma^theta_{r theta} = Gamma^theta_{theta r} = 1/r
    Gamma[:, 1, 0, 1] = 1.0 / r
    Gamma[:, 1, 1, 0] = 1.0 / r
    return Gamma


def polar_gt_accel(z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Accel from GT Christoffels in polar coordinates.
    z: (B,2), v: (B,2)
    returns a_gt: (B,2)
    """
    r = z[:, 0].clamp_min(1e-6)
    vr = v[:, 0]
    vth = v[:, 1]
    a_r = r * vth * vth
    a_th = -2.0 * vr * vth / r
    return torch.stack([a_r, a_th], dim=-1)


def compute_v0(z0: torch.Tensor, z1: torch.Tensor, dt: float, use_ae: bool) -> torch.Tensor:
    if use_ae:
        return (z1 - z0) / dt
    # Polar-safe angle wrap for theta
    dr = (z1[:, 0] - z0[:, 0]) / dt
    dtheta = torch.atan2(torch.sin(z1[:, 1] - z0[:, 1]), torch.cos(z1[:, 1] - z0[:, 1]))
    dtheta = dtheta / dt
    return torch.stack([dr, dtheta], dim=-1)


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

    x_dim = cfg.d if not cfg.use_ae else 2

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

    # Evaluation dataset (new seed to avoid training overlap)
    ds = PolarInertialDataset(n_traj=args.batch_size * args.n_batches, T=cfg.T, dt=cfg.dt, seed=args.seed)

    # Metrics accumulators
    traj_mse_sum = 0.0
    traj_mse_count = 0
    per_t_mse = torch.zeros(cfg.T, device=device)

    gamma_mse_r = 0.0
    gamma_mse_tr = 0.0
    accel_mse = 0.0
    gamma_count = 0

    christoffel = model.christoffel

    with torch.no_grad():
        for b in range(args.n_batches):
            x_seq = ds.x_seq[b * args.batch_size:(b + 1) * args.batch_size].to(device)

            z_true = model.encode_seq(x_seq)
            z0 = z_true[:, 0, :]
            z1 = z_true[:, 1, :]
            v0 = compute_v0(z0, z1, cfg.dt, cfg.use_ae)

            z_pred = model.rollout(z0, v0, cfg.T, cfg.dt)

            if cfg.use_ae:
                x_pred = model.decode_seq(z_pred)
                err = (x_pred - x_seq) ** 2
            else:
                err = (z_pred - x_seq) ** 2

            traj_mse_sum += float(err.mean().item())
            traj_mse_count += 1
            per_t_mse += err.mean(dim=(0, 2))

            # GT comparisons only valid for polar direct-state
            if (not cfg.use_ae) and cfg.d == 2:
                z_flat = z_true.reshape(-1, 2)
                v_flat = (z_true[:, 1:, :] - z_true[:, :-1, :]) / cfg.dt
                v_flat = torch.cat([v0.unsqueeze(1), v_flat], dim=1).reshape(-1, 2)

                Gamma_pred = christoffel(z_flat)
                Gamma_gt = polar_gt_gamma(z_flat)

                gamma_mse_r += torch.mean((Gamma_pred[:, 0, 1, 1] - Gamma_gt[:, 0, 1, 1]) ** 2).item()
                gamma_mse_tr += torch.mean((Gamma_pred[:, 1, 0, 1] - Gamma_gt[:, 1, 0, 1]) ** 2).item()

                # Acceleration error
                a_pred = -torch.einsum("bijk,bj,bk->bi", Gamma_pred, v_flat, v_flat)
                a_gt = polar_gt_accel(z_flat, v_flat)
                accel_mse += torch.mean((a_pred - a_gt) ** 2).item()
                gamma_count += 1

    per_t_mse = (per_t_mse / traj_mse_count).detach().cpu().tolist()

    results: Dict[str, float] = {
        "traj_mse": traj_mse_sum / max(1, traj_mse_count),
        "traj_mse_final": per_t_mse[-1] if len(per_t_mse) > 0 else float("nan"),
    }

    if gamma_count > 0:
        results["gamma_mse_r_thetatheta"] = gamma_mse_r / gamma_count
        results["gamma_mse_theta_rtheta"] = gamma_mse_tr / gamma_count
        results["accel_mse"] = accel_mse / gamma_count

    # Optional metric eval on a batch of endpoints
    if args.use_metric_eval:
        with torch.no_grad():
            x_seq = ds.x_seq[:args.batch_size].to(device)
            z_true = model.encode_seq(x_seq)
            z0 = z_true[:, 0, :]
            zT = z_true[:, -1, :]
            z_mid = z_true[:, cfg.T // 2, :]

            z_base = z0.mean(dim=0, keepdim=True)
            g0 = model.base_metric().detach()

            gA = reconstruct_metric_straight_path(christoffel, z_base, g0, zT, n_steps=args.metric_steps)
            gB = reconstruct_metric_two_segment(christoffel, z_base, g0, z_mid, zT, n_steps=args.metric_steps)

            loop = torch.mean((gA - gB) ** 2, dim=(1, 2))
            results["loop_mse_mean"] = float(loop.mean().item())

            results["symmetry_loss"] = float(symmetry_loss(gA).item())
            results["logabsdet_loss"] = float(logabsdet_loss(gA, floor=-2.0).item())
            results["signature_loss"] = float(signature_loss(gA, cfg.signature).item())

            eigs = torch.linalg.eigvalsh(0.5 * (gA + gA.transpose(1, 2)))
            results["metric_min_eig"] = float(eigs.min().item())
            results["metric_frac_neg_eig"] = float((eigs < 0).float().mean().item())

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump({"results": results, "per_t_mse": per_t_mse}, f, indent=2)
        print(f"Saved JSON to: {args.save_json}")


if __name__ == "__main__":
    main()