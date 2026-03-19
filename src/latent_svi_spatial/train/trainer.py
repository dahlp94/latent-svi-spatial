from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from latent_svi_spatial.models.sar import compute_A_dense
from latent_svi_spatial.vi.elbo import ELBOResult, estimate_elbo
from latent_svi_spatial.vi.variational_family import VariationalFamily


Tensor = torch.Tensor


def build_weight_from_factors(H: Tensor, C: Tensor) -> Tensor:
    W_raw = H @ C @ H.T
    return W_raw - torch.diag(torch.diag(W_raw))


@dataclass
class TrainConfig:
    n_steps: int = 500
    lr: float = 1e-2
    n_mc_samples: int = 1
    stability_penalty_weight: float = 0.0
    hard_stability_penalty_weight: float = 100.0
    stability_margin: float = 0.05
    unstable_logdet_value: float = -1e6
    clip_grad_norm: Optional[float] = 5.0
    log_every: int = 50
    verbose: bool = True


@dataclass
class TrainState:
    step: int
    elbo: float
    loss: float
    expected_log_likelihood: float
    kl_total: float
    stability_penalty: float
    stability_rate: float
    min_reduced_eig: float
    mc_logdet: float
    mc_quadratic: float
    mc_log_sigma2: float
    rho_mean: float
    sigma2_mean: float
    W_fro_error: float | None = None
    A_fro_error: float | None = None
    predictive_rmse: float | None = None

@dataclass
class TrainHistory:
    records: list[TrainState] = field(default_factory=list)

    def append(self, state: TrainState) -> None:
        self.records.append(state)

    def to_dict(self) -> dict[str, list[float]]:
        out = {
            "step": [],
            "elbo": [],
            "loss": [],
            "expected_log_likelihood": [],
            "kl_total": [],
            "stability_penalty": [],
            "stability_rate": [],
            "min_reduced_eig": [],
            "mc_logdet": [],
            "mc_quadratic": [],
            "mc_log_sigma2": [],
            "rho_mean": [],
            "sigma2_mean": [],
            "W_fro_error": [],
            "A_fro_error": [],
            "predictive_rmse": [],
        }
        for r in self.records:
            out["step"].append(r.step)
            out["elbo"].append(r.elbo)
            out["loss"].append(r.loss)
            out["expected_log_likelihood"].append(r.expected_log_likelihood)
            out["kl_total"].append(r.kl_total)
            out["stability_penalty"].append(r.stability_penalty)
            out["stability_rate"].append(r.stability_rate)
            out["min_reduced_eig"].append(r.min_reduced_eig)
            out["mc_logdet"].append(r.mc_logdet)
            out["mc_quadratic"].append(r.mc_quadratic)
            out["mc_log_sigma2"].append(r.mc_log_sigma2)
            out["rho_mean"].append(r.rho_mean)
            out["sigma2_mean"].append(r.sigma2_mean)
            out["W_fro_error"].append(r.W_fro_error)
            out["A_fro_error"].append(r.A_fro_error)
            out["predictive_rmse"].append(r.predictive_rmse)
        return out

    def last(self) -> Optional[TrainState]:
        if not self.records:
            return None
        return self.records[-1]


def _build_train_state(
    step: int,
    result: ELBOResult,
    variational_family: VariationalFamily,
    X: Tensor,
    y: Tensor,
    true_W: Tensor | None = None,
) -> TrainState:
    summary = variational_family.summary()

    W_fro_error = None
    A_fro_error = None
    predictive_rmse = None

    with torch.no_grad():
        H_mean = variational_family.mean_H()
        C_mean = variational_family.mean_C()
        rho_mean_tensor = variational_family.mean_rho()
        beta_mean = variational_family.mean_beta()

        W_mean = build_weight_from_factors(H_mean, C_mean)

        if true_W is not None:
            W_fro_error = float(torch.norm(W_mean - true_W).item())

            A_true = compute_A_dense(true_W, rho_mean_tensor)
            A_mean = compute_A_dense(W_mean, rho_mean_tensor)
            A_fro_error = float(torch.norm(A_mean - A_true).item())

        Xbeta_mean = torch.einsum("tnp,p->tn", X, beta_mean)
        A_mean = compute_A_dense(W_mean, rho_mean_tensor)
        y_hat_mean = torch.linalg.solve(A_mean, Xbeta_mean.T).T
        predictive_rmse = float(torch.sqrt(torch.mean((y_hat_mean - y) ** 2)).item())

    return TrainState(
        step=step,
        elbo=float(result.elbo.item()),
        loss=float(result.loss.item()),
        expected_log_likelihood=float(result.expected_log_likelihood.item()),
        kl_total=float(result.kl_total.item()),
        stability_penalty=float(result.stability_penalty.item()),
        stability_rate=float(result.stability_rate.item()),
        min_reduced_eig=float(result.min_reduced_eig.item()),
        mc_logdet=float(result.mc_logdet.item()),
        mc_quadratic=float(result.mc_quadratic.item()),
        mc_log_sigma2=float(result.mc_log_sigma2.item()),
        rho_mean=summary["rho_mean"],
        sigma2_mean=summary["sigma2_mean"],
        W_fro_error=W_fro_error,
        A_fro_error=A_fro_error,
        predictive_rmse=predictive_rmse,
    )

def train_variational_model(
    variational_family: VariationalFamily,
    X: Tensor,
    y: Tensor,
    *,
    true_W: Tensor | None = None,
    config: Optional[TrainConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> TrainHistory:
    """
    Minimal training loop for the low-rank SAR VI model.

    Parameters
    ----------
    variational_family:
        The variational family to optimize.
    X:
        Covariates of shape (T, N, P).
    y:
        Responses of shape (T, N).
    config:
        Training hyperparameters.
    optimizer:
        Optional optimizer. If None, Adam is created internally.

    Returns
    -------
    TrainHistory
        Logged training history.
    """
    if config is None:
        config = TrainConfig()

    if optimizer is None:
        optimizer = torch.optim.Adam(variational_family.parameters(), lr=config.lr)

    history = TrainHistory()

    for step in range(1, config.n_steps + 1):
        optimizer.zero_grad()

        result = estimate_elbo(
            variational_family=variational_family,
            X=X,
            y=y,
            n_mc_samples=config.n_mc_samples,
            stability_penalty_weight=config.stability_penalty_weight,
            stability_margin=config.stability_margin,
            hard_stability_penalty_weight=config.hard_stability_penalty_weight,
            unstable_logdet_value=config.unstable_logdet_value,
        )

        result.loss.backward()

        if config.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                variational_family.parameters(),
                max_norm=config.clip_grad_norm,
            )

        optimizer.step()

        state = _build_train_state(
            step,
            result,
            variational_family,
            X=X,
            y=y,
            true_W=true_W,
        )
        history.append(state)

        if config.verbose and (
            step == 1
            or step % config.log_every == 0
            or step == config.n_steps
        ):
            msg = (
                f"[step {step:4d}] "
                f"ELBO={state.elbo: .3f}  "
                f"Loss={state.loss: .3f}  "
                f"LogLik={state.expected_log_likelihood: .3f}  "
                f"KL={state.kl_total: .3f}  "
                f"StabPen={state.stability_penalty: .3f}  "
                f"StableRate={state.stability_rate: .3f}  "
                f"MinEig={state.min_reduced_eig: .3f}  "
                f"rho_mean={state.rho_mean: .4f}  "
                f"sigma2_mean={state.sigma2_mean: .4f}"
            )

            if state.W_fro_error is not None:
                msg += f"  W_err={state.W_fro_error: .4f}"

            if state.predictive_rmse is not None:
                msg += f"  RMSE={state.predictive_rmse: .4f}"

            print(msg)

    return history


__all__ = [
    "TrainConfig",
    "TrainState",
    "TrainHistory",
    "train_variational_model",
]