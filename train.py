# train.py
import argparse
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader

from config import TrainConfig, Signature
from utils import set_seed, get_device
from data import PolarInertialDataset
from model import FullGeoModel
from metric_recon import reconstruct_metric_straight_path, reconstruct_metric_two_segment
from losses import traj_mse, recon_mse, symmetry_loss, logabsdet_loss, signature_loss

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--use_ae", type=int, default=1)
    p.add_argument("--manifold", type=str, default="none", choices=["none", "sphere", "hyperboloid"])
    p.add_argument("--p", type=int, default=2)
    p.add_argument("--q", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = TrainConfig()
    cfg.use_ae = bool(args.use_ae)
    cfg.manifold = args.manifold
    cfg.signature = Signature(p=args.p, q=args.q)
    cfg.epochs = args.epochs

    set_seed(0)
    device = get_device()
    print("Device:", device, "| GPUs:", torch.cuda.device_count())

    # Data
    ds = PolarInertialDataset(n_traj=cfg.n_traj, T=cfg.T, dt=cfg.dt, seed=0)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # For this demo, x_dim == 2 (polar coords). If you use real observations, x_dim changes.
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

    # Optional multi-GPU
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    for ep in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0

        for x_seq, dt in dl:
            x_seq = x_seq.to(device, non_blocking=True)  # (B,T,d)
            dt = float(dt[0].item())  # same for all in batch

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                # 1) Encode to latent z(t)
                z_true = model.module.encode_seq(x_seq) if isinstance(model, nn.DataParallel) else model.encode_seq(x_seq)

                # 2) Initial conditions for geodesic integration
                z0 = z_true[:, 0, :]
                z1 = z_true[:, 1, :]
                v0 = (z1 - z0) / dt

                # 3) Rollout geodesic in latent
                z_pred = model.module.rollout(z0, v0, cfg.T, dt) if isinstance(model, nn.DataParallel) else model.rollout(z0, v0, cfg.T, dt)

                # 4) Primary loss:
                #    - if AE on: decode z_pred and compare in x-space
                #    - else: compare z directly (x=z)
                if cfg.use_ae:
                    x_pred = model.module.decode_seq(z_pred) if isinstance(model, nn.DataParallel) else model.decode_seq(z_pred)
                    loss_main = recon_mse(x_pred, x_seq)
                else:
                    loss_main = traj_mse(z_pred, x_seq)

                loss = loss_main

                # 5) Metric reconstruction + consistency losses
                if (cfg.use_metric_losses and (ep > round(cfg.metric_warmup_epochs * cfg.epochs))):
                    # Choose a basepoint in latent (batch mean)
                    z_base = z0.mean(dim=0, keepdim=True)  # (1,d)

                    g0 = model.module.base_metric() if isinstance(model, nn.DataParallel) else model.base_metric()

                    # Target points
                    zT = z_true[:, -1, :]
                    zm = z_true[:, cfg.T // 2, :]

                    christoffel = model.module.christoffel if isinstance(model, nn.DataParallel) else model.christoffel

                    gA = reconstruct_metric_straight_path(christoffel, z_base, g0, zT, n_steps=cfg.metric_steps)
                    gB = reconstruct_metric_two_segment(christoffel, z_base, g0, zm, zT, n_steps=cfg.metric_steps)

                    loss_loop = ((gA - gB) ** 2).mean()
                    loss_sig = signature_loss(gA, cfg.signature)
                    loss_det = logabsdet_loss(gA, floor=-2.0)
                    loss_sym = symmetry_loss(gA)

                    loss = loss + cfg.lam_loop * loss_loop + cfg.lam_sig * loss_sig + cfg.lam_det * loss_det + cfg.lam_sym * loss_sym

                # 6) Explicit weight decay on Christoffel coefficients for simplicity
                # (keeps polynomial small/interpretably sparse-ish)
                christoffel_params = (model.module.christoffel.parameters() if isinstance(model, nn.DataParallel) else model.christoffel.parameters())
                wd = 0.0
                for p in christoffel_params:
                    wd = wd + (p ** 2).mean()
                loss = loss + cfg.lam_wd * wd

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total += float(loss.item())

        print(f"Epoch {ep:03d} | loss {total / len(dl):.6f}")

        # save new checkpoints
        os.makedirs("checkpoints", exist_ok=True)
        to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save({"model_state": to_save}, "checkpoints/latest.pt")

    print("Training complete.")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
