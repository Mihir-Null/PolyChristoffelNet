# config.py
from dataclasses import dataclass

@dataclass
class Signature:
    # g has q negative eigenvalues and p positive eigenvalues
    p: int
    q: int

@dataclass
class TrainConfig:
    # Data
    n_traj: int = 8192
    T: int = 50
    dt: float = 0.05
    batch_size: int = 256
    num_workers: int = 2

    # Model
    d: int = 2
    poly_degree: int = 2

    # Autoencoder
    use_ae: bool = False
    ae_hidden: int = 128
    ae_depth: int = 3
    manifold: str = "none"   # "none" | "sphere" | "hyperboloid" - enforces latent embeddings remain valid statevectors, transport ensures that the vectors are pulled back onto latent space of the manifold.
    # If manifold="hyperboloid", latent dimension should be >=2 (we use d here)

    # Metric reconstruction / pseudo-Riemannian checks
    use_metric_losses: bool = True
    metric_steps: int = 12
    signature: Signature = Signature(p=2, q=0)  # default SPD for polar demo

    # Optimization
    epochs: int = 30
    lr: float = 2e-3
    amp: bool = True

    # Loss weights
    lam_loop: float = 0.1
    lam_sig: float = 0.1
    lam_det: float = 0.1
    lam_sym: float = 0.05
    lam_wd: float = 1e-4
