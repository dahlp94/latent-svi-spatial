from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Optional

import torch


Tensor = torch.Tensor
NormalizeMode = Literal["row", "spectral", "none"]


@dataclass
class SyntheticConfig:
    n: int = 50
    t: int = 10
    p: int = 3
    r: int = 4

    include_intercept: bool = True
    intercept_value: float = 1.0

    membership_prior: Literal["dirichlet", "softmax_normal"] = "dirichlet"
    dirichlet_alpha: float = 1.0
    membership_scale: float = 1.0

    c_scale: float = 1.0
    c_symmetric: bool = True
    c_positive: bool = True

    normalize_w: NormalizeMode = "row"
    zero_diagonal: bool = True

    rho: float = 0.4
    sigma2: float = 0.25

    beta_scale: float = 1.0
    unit_fe_scale: float = 0.0
    time_fe_scale: float = 0.0

    x_distribution: Literal["normal", "uniform"] = "normal"
    standardize_x: bool = False

    enforce_stability: bool = True
    stability_margin: float = 0.95

    seed: Optional[int] = None
    device: str = "cpu"
    dtype: str = "float64"


@dataclass
class SyntheticPanelData:
    config: dict
    X: Tensor                # (T, N, P_eff)
    y: Tensor                # (T, N)
    H: Tensor                # (N, r)
    C: Tensor                # (r, r)
    W_raw: Tensor            # (N, N)
    W: Tensor                # (N, N)
    beta: Tensor             # (P_eff,)
    rho: Tensor              # scalar tensor
    sigma2: Tensor           # scalar tensor
    unit_fe: Tensor          # (N,)
    time_fe: Tensor          # (T,)
    mean: Tensor             # (T, N), deterministic component before solve
    A: Tensor                # I - rho W
    rho_upper_bound: Tensor      # scalar tensor

    def to_dict(self) -> dict:
        return asdict(self)


def _get_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype}'. Use one of {list(mapping)}.")
    return mapping[dtype]


def _set_seed(seed: Optional[int]) -> None:
    if seed is not None:
        torch.manual_seed(seed)


def sample_memberships(
    n: int,
    r: int,
    *,
    prior: Literal["dirichlet", "softmax_normal"] = "dirichlet",
    dirichlet_alpha: float = 1.0,
    membership_scale: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """
    Sample latent membership matrix H of shape (n, r).

    Returns rows on the simplex when using either:
    - Dirichlet
    - softmax(Normal)
    """
    if r <= 0:
        raise ValueError("r must be positive.")
    if n <= 0:
        raise ValueError("n must be positive.")

    if prior == "dirichlet":
        alpha = torch.full((r,), float(dirichlet_alpha), device=device, dtype=dtype)
        dist = torch.distributions.Dirichlet(alpha)
        H = dist.sample((n,))
    elif prior == "softmax_normal":
        Z = membership_scale * torch.randn(n, r, device=device, dtype=dtype)
        H = torch.softmax(Z, dim=1)
    else:
        raise ValueError(f"Unknown membership prior '{prior}'.")

    return H


def sample_interaction_matrix(
    r: int,
    *,
    scale: float = 1.0,
    symmetric: bool = True,
    positive: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """
    Sample latent interaction matrix C of shape (r, r).
    """
    C = scale * torch.randn(r, r, device=device, dtype=dtype)

    if symmetric:
        C = 0.5 * (C + C.T)

    if positive:
        C = torch.nn.functional.softplus(C)

    return C


def build_low_rank_weight(
    H: Tensor,
    C: Tensor,
    *,
    zero_diagonal: bool = True,
) -> Tensor:
    """
    Build W_raw = H C H^T, then optionally enforce zero diagonal.
    """
    W = H @ C @ H.T
    if zero_diagonal:
        W = W - torch.diag(torch.diag(W))
    return W


def normalize_weight_matrix(
    W: Tensor,
    *,
    mode: NormalizeMode = "row",
    eps: float = 1e-8,
) -> Tensor:
    """
    Normalize W using one of:
    - row: row-stochastic normalization
    - spectral: divide by spectral radius
    - none: return unchanged
    """
    if mode == "none":
        return W

    if mode == "row":
        row_sums = W.sum(dim=1, keepdim=True)
        safe_row_sums = torch.where(row_sums.abs() < eps, torch.ones_like(row_sums), row_sums)
        return W / safe_row_sums

    if mode == "spectral":
        eigvals = torch.linalg.eigvals(W)
        radius = eigvals.abs().max().real
        scale = torch.clamp(radius, min=eps)
        return W / scale

    raise ValueError(f"Unknown normalize mode '{mode}'.")


def compute_system_matrix(W: Tensor, rho: float | Tensor) -> Tensor:
    n = W.shape[0]
    eye = torch.eye(n, device=W.device, dtype=W.dtype)
    return eye - rho * W


def stable_rho_upper_bound(W: Tensor, eps: float = 1e-6) -> float:
    """
    Conservative upper bound for rho based on spectral radius:
        rho < 1 / spectral_radius(W)
    """
    eigvals = torch.linalg.eigvals(W)
    radius = float(eigvals.abs().max().real.item())
    if radius < eps:
        return float("inf")
    return 1.0 / radius


def sample_beta(
    p_eff: int,
    *,
    scale: float = 1.0,
    include_intercept: bool = True,
    intercept_value: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    beta = scale * torch.randn(p_eff, device=device, dtype=dtype)
    if include_intercept:
        beta[0] = intercept_value
    return beta


def sample_covariates(
    t: int,
    n: int,
    p: int,
    *,
    include_intercept: bool = True,
    distribution: Literal["normal", "uniform"] = "normal",
    standardize: bool = False,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """
    Returns X with shape (T, N, P_eff).
    If include_intercept=True, the first column is 1.
    """
    if distribution == "normal":
        X_core = torch.randn(t, n, p, device=device, dtype=dtype)
    elif distribution == "uniform":
        X_core = 2.0 * torch.rand(t, n, p, device=device, dtype=dtype) - 1.0
    else:
        raise ValueError(f"Unknown x distribution '{distribution}'.")

    if standardize and p > 0:
        mean = X_core.mean(dim=(0, 1), keepdim=True)
        std = X_core.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)
        X_core = (X_core - mean) / std

    if include_intercept:
        intercept = torch.ones(t, n, 1, device=device, dtype=dtype)
        X = torch.cat([intercept, X_core], dim=2)
    else:
        X = X_core

    return X


def sample_fixed_effects(
    n: int,
    t: int,
    *,
    unit_scale: float = 0.0,
    time_scale: float = 0.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor]:
    unit_fe = unit_scale * torch.randn(n, device=device, dtype=dtype)
    time_fe = time_scale * torch.randn(t, device=device, dtype=dtype)
    return unit_fe, time_fe


def build_mean_process(
    X: Tensor,
    beta: Tensor,
    unit_fe: Optional[Tensor] = None,
    time_fe: Optional[Tensor] = None,
) -> Tensor:
    """
    Build deterministic mean for each t:
        mu_t = X_t beta + unit_fe + time_fe_t
    Returns shape (T, N).
    """
    mean = torch.einsum("tnp,p->tn", X, beta)

    if unit_fe is not None:
        mean = mean + unit_fe.unsqueeze(0)

    if time_fe is not None:
        mean = mean + time_fe.unsqueeze(1)

    return mean


def sample_panel_y(
    mean: Tensor,
    W: Tensor,
    rho: float,
    sigma2: float,
) -> tuple[Tensor, Tensor]:
    """
    Sample y_t independently from the SAR model:
        (I - rho W) y_t = mean_t + eps_t
        eps_t ~ N(0, sigma^2 I)

    Returns:
        y: (T, N)
        A: (N, N)
    """
    t, n = mean.shape
    A = compute_system_matrix(W, rho)

    eps = torch.sqrt(torch.tensor(sigma2, device=mean.device, dtype=mean.dtype)) * torch.randn(
        t, n, device=mean.device, dtype=mean.dtype
    )
    rhs = mean + eps

    # Solve A y_t = rhs_t^T for all t in batch
    # torch.linalg.solve expects (..., n, n) and (..., n, k)
    y = torch.linalg.solve(A, rhs.T).T
    return y, A


def generate_synthetic_panel(config: SyntheticConfig) -> SyntheticPanelData:
    """
    Main entry point for generating a synthetic panel dataset for the low-rank SAR model.
    """
    _set_seed(config.seed)
    dtype = _get_dtype(config.dtype)
    device = config.device

    p_eff = config.p + (1 if config.include_intercept else 0)

    H = sample_memberships(
        config.n,
        config.r,
        prior=config.membership_prior,
        dirichlet_alpha=config.dirichlet_alpha,
        membership_scale=config.membership_scale,
        device=device,
        dtype=dtype,
    )

    C = sample_interaction_matrix(
        config.r,
        scale=config.c_scale,
        symmetric=config.c_symmetric,
        positive=config.c_positive,
        device=device,
        dtype=dtype,
    )

    W_raw = build_low_rank_weight(
        H,
        C,
        zero_diagonal=config.zero_diagonal,
    )

    W = normalize_weight_matrix(
        W_raw,
        mode=config.normalize_w,
    )
    
    rho_upper = stable_rho_upper_bound(W)

    if rho_upper != float("inf") and abs(config.rho) >= rho_upper:
        if config.enforce_stability and config.normalize_w == "none":
            # Since W is linear in C, rescaling C rescales W and preserves structure.
            # We shrink just enough so that target rho sits safely inside the stable region.
            scale = config.stability_margin * rho_upper / abs(config.rho)

            C = C * scale
            W_raw = W_raw * scale
            W = W * scale

            rho_upper = stable_rho_upper_bound(W)

            if rho_upper != float("inf") and abs(config.rho) >= rho_upper:
                raise ValueError(
                    f"Auto-rescaling failed: rho={config.rho:.4f} still not stable. "
                    f"Need |rho| < {rho_upper:.4f}."
                )
        else:
            raise ValueError(
                f"rho={config.rho:.4f} is not stable for generated W. "
                f"Need |rho| < {rho_upper:.4f}."
            )



    X = sample_covariates(
        config.t,
        config.n,
        config.p,
        include_intercept=config.include_intercept,
        distribution=config.x_distribution,
        standardize=config.standardize_x,
        device=device,
        dtype=dtype,
    )

    beta = sample_beta(
        p_eff,
        scale=config.beta_scale,
        include_intercept=config.include_intercept,
        intercept_value=config.intercept_value,
        device=device,
        dtype=dtype,
    )

    unit_fe, time_fe = sample_fixed_effects(
        config.n,
        config.t,
        unit_scale=config.unit_fe_scale,
        time_scale=config.time_fe_scale,
        device=device,
        dtype=dtype,
    )

    mean = build_mean_process(X, beta, unit_fe=unit_fe, time_fe=time_fe)
    y, A = sample_panel_y(mean, W, config.rho, config.sigma2)

    return SyntheticPanelData(
        config=asdict(config),
        X=X,
        y=y,
        H=H,
        C=C,
        W_raw=W_raw,
        W=W,
        beta=beta,
        rho=torch.tensor(config.rho, device=device, dtype=dtype),
        sigma2=torch.tensor(config.sigma2, device=device, dtype=dtype),
        unit_fe=unit_fe,
        time_fe=time_fe,
        mean=mean,
        A=A,
        rho_upper_bound=torch.tensor(rho_upper, device=device, dtype=dtype),
    )


__all__ = [
    "SyntheticConfig",
    "SyntheticPanelData",
    "sample_memberships",
    "sample_interaction_matrix",
    "build_low_rank_weight",
    "normalize_weight_matrix",
    "compute_system_matrix",
    "stable_rho_upper_bound",
    "sample_beta",
    "sample_covariates",
    "sample_fixed_effects",
    "build_mean_process",
    "sample_panel_y",
    "generate_synthetic_panel",
]