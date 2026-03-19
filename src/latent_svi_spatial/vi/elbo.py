from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from latent_svi_spatial.models.sar import (
    apply_A_to_y,
    safe_logdet_A_lowrank,
    stability_penalty,
)
from latent_svi_spatial.vi.variational_family import VariationalFamily, VariationalSample


Tensor = torch.Tensor


@dataclass
class ELBOResult:
    elbo: Tensor
    loss: Tensor
    expected_log_likelihood: Tensor
    kl_total: Tensor
    stability_penalty: Tensor
    stability_rate: Tensor
    min_reduced_eig: Tensor
    mc_logdet: Tensor
    mc_quadratic: Tensor
    mc_log_sigma2: Tensor
    n_mc_samples: int

    def as_dict(self) -> dict[str, float]:
        return {
            "elbo": float(self.elbo.item()),
            "loss": float(self.loss.item()),
            "expected_log_likelihood": float(self.expected_log_likelihood.item()),
            "kl_total": float(self.kl_total.item()),
            "stability_penalty": float(self.stability_penalty.item()),
            "stability_rate": float(self.stability_rate.item()),
            "min_reduced_eig": float(self.min_reduced_eig.item()),
            "mc_logdet": float(self.mc_logdet.item()),
            "mc_quadratic": float(self.mc_quadratic.item()),
            "mc_log_sigma2": float(self.mc_log_sigma2.item()),
            "n_mc_samples": self.n_mc_samples,
        }


def _validate_inputs(X: Tensor, y: Tensor) -> tuple[int, int, int]:
    """
    X: (T, N, P)
    y: (T, N)
    """
    if X.ndim != 3:
        raise ValueError(f"X must have shape (T, N, P), got {tuple(X.shape)}")
    if y.ndim != 2:
        raise ValueError(f"y must have shape (T, N), got {tuple(y.shape)}")

    T_x, N_x, P = X.shape
    T_y, N_y = y.shape

    if T_x != T_y or N_x != N_y:
        raise ValueError(
            f"Incompatible shapes: X has {(T_x, N_x, P)} while y has {(T_y, N_y)}"
        )

    return T_x, N_x, P


def _single_sample_log_likelihood(
    sample: VariationalSample,
    X: Tensor,
    y: Tensor,
    *,
    unstable_logdet_value: float = -1e6,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute one-sample SAR log-likelihood contribution safely.

    Returns:
        log_lik
        logdet_term
        quadratic_term
        is_stable
        min_reduced_eig
    """
    T, N, P = _validate_inputs(X, y)

    H = sample.H
    C = sample.C
    rho = sample.rho
    beta = sample.beta
    sigma2 = sample.sigma2

    if beta.shape[0] != P:
        raise ValueError(
            f"beta dimension mismatch: beta has shape {tuple(beta.shape)}, expected ({P},)"
        )

    sigma2 = torch.clamp(sigma2, min=1e-8)

    logdet_A, is_stable, min_reduced_eig = safe_logdet_A_lowrank(
        H,
        C,
        rho,
        unstable_logdet_value=unstable_logdet_value,
    )

    Xbeta = torch.einsum("tnp,p->tn", X, beta)
    Ay = apply_A_to_y(H, C, y, rho, zero_diagonal=True)

    residual = Ay - Xbeta
    quadratic = torch.sum(residual.pow(2))

    log_sigma2 = torch.log(sigma2)
    log_lik = (
        T * logdet_A
        - 0.5 * N * T * torch.log(
            torch.tensor(2.0 * torch.pi, device=X.device, dtype=X.dtype)
        )
        - 0.5 * N * T * log_sigma2
        - 0.5 * quadratic / sigma2
    )

    return log_lik, logdet_A, quadratic, is_stable, min_reduced_eig


def estimate_elbo(
    variational_family: VariationalFamily,
    X: Tensor,
    y: Tensor,
    *,
    n_mc_samples: int = 1,
    stability_penalty_weight: float = 0.0,
    stability_margin: float = 0.05,
    hard_stability_penalty_weight: float = 100.0,
    unstable_logdet_value: float = -1e6,
) -> ELBOResult:
    """
    Monte Carlo estimate of the ELBO.

    ELBO = E_q[log p(Y | latent)] - KL(q || p)

    This MVP version includes:
      - SAR log-likelihood via Monte Carlo
      - KL from the variational family

    It does NOT yet separate analytic expectations over q(beta) or q(sigma²).
    Everything is sampled jointly for simplicity and plumbing correctness.
    """
    if n_mc_samples <= 0:
        raise ValueError("n_mc_samples must be positive.")

    _validate_inputs(X, y)

    log_lik_terms = []
    logdet_terms = []
    quadratic_terms = []
    log_sigma2_terms = []
    stability_penalty_terms = []
    stability_indicator_terms = []
    min_reduced_eig_terms = []

    for _ in range(n_mc_samples):
        sample = variational_family.rsample()

        log_lik, logdet_A, quadratic, is_stable, min_reduced_eig = _single_sample_log_likelihood(
            sample,
            X,
            y,
            unstable_logdet_value=unstable_logdet_value,
        )

        penalty, _, _ = stability_penalty(
            sample.H,
            sample.C,
            sample.rho,
            margin=stability_margin,
            soft_weight=1.0,
            hard_weight=hard_stability_penalty_weight,
        )

        log_lik_terms.append(log_lik)
        logdet_terms.append(logdet_A)
        quadratic_terms.append(quadratic)
        log_sigma2_terms.append(torch.log(sample.sigma2))
        stability_penalty_terms.append(penalty)
        stability_indicator_terms.append(is_stable)
        min_reduced_eig_terms.append(min_reduced_eig)

    expected_log_likelihood = torch.stack(log_lik_terms).mean()
    mc_logdet = torch.stack(logdet_terms).mean()
    mc_quadratic = torch.stack(quadratic_terms).mean()
    mc_log_sigma2 = torch.stack(log_sigma2_terms).mean()
    mean_stability_penalty = torch.stack(stability_penalty_terms).mean()
    mean_stability_rate = torch.stack(stability_indicator_terms).mean()
    mean_min_reduced_eig = torch.stack(min_reduced_eig_terms).mean()

    kl_total = variational_family.kl_total()

    elbo = (
        expected_log_likelihood
        - kl_total
        - stability_penalty_weight * mean_stability_penalty
    )
    loss = -elbo

    return ELBOResult(
        elbo=elbo,
        loss=loss,
        expected_log_likelihood=expected_log_likelihood,
        kl_total=kl_total,
        stability_penalty=mean_stability_penalty,
        stability_rate=mean_stability_rate,
        min_reduced_eig=mean_min_reduced_eig,
        mc_logdet=mc_logdet,
        mc_quadratic=mc_quadratic,
        mc_log_sigma2=mc_log_sigma2,
        n_mc_samples=n_mc_samples,
    )


__all__ = [
    "ELBOResult",
    "estimate_elbo",
]