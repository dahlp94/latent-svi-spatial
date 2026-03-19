from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from latent_svi_spatial.vi.elbo import ELBOResult, estimate_elbo
from latent_svi_spatial.vi.variational_family import VariationalFamily


Tensor = torch.Tensor


@dataclass
class TrainConfig:
    n_steps: int = 500
    lr: float = 1e-2
    n_mc_samples: int = 1
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
    mc_logdet: float
    mc_quadratic: float
    mc_log_sigma2: float
    rho_mean: float
    sigma2_mean: float


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
            "mc_logdet": [],
            "mc_quadratic": [],
            "mc_log_sigma2": [],
            "rho_mean": [],
            "sigma2_mean": [],
        }
        for r in self.records:
            out["step"].append(r.step)
            out["elbo"].append(r.elbo)
            out["loss"].append(r.loss)
            out["expected_log_likelihood"].append(r.expected_log_likelihood)
            out["kl_total"].append(r.kl_total)
            out["mc_logdet"].append(r.mc_logdet)
            out["mc_quadratic"].append(r.mc_quadratic)
            out["mc_log_sigma2"].append(r.mc_log_sigma2)
            out["rho_mean"].append(r.rho_mean)
            out["sigma2_mean"].append(r.sigma2_mean)
        return out

    def last(self) -> Optional[TrainState]:
        if not self.records:
            return None
        return self.records[-1]


def _build_train_state(
    step: int,
    result: ELBOResult,
    variational_family: VariationalFamily,
) -> TrainState:
    summary = variational_family.summary()
    return TrainState(
        step=step,
        elbo=float(result.elbo.item()),
        loss=float(result.loss.item()),
        expected_log_likelihood=float(result.expected_log_likelihood.item()),
        kl_total=float(result.kl_total.item()),
        mc_logdet=float(result.mc_logdet.item()),
        mc_quadratic=float(result.mc_quadratic.item()),
        mc_log_sigma2=float(result.mc_log_sigma2.item()),
        rho_mean=summary["rho_mean"],
        sigma2_mean=summary["sigma2_mean"],
    )


def train_variational_model(
    variational_family: VariationalFamily,
    X: Tensor,
    y: Tensor,
    *,
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
        )

        result.loss.backward()

        if config.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                variational_family.parameters(),
                max_norm=config.clip_grad_norm,
            )

        optimizer.step()

        state = _build_train_state(step, result, variational_family)
        history.append(state)

        if config.verbose and (
            step == 1
            or step % config.log_every == 0
            or step == config.n_steps
        ):
            print(
                f"[step {step:4d}] "
                f"ELBO={state.elbo: .3f}  "
                f"Loss={state.loss: .3f}  "
                f"LogLik={state.expected_log_likelihood: .3f}  "
                f"KL={state.kl_total: .3f}  "
                f"rho_mean={state.rho_mean: .4f}  "
                f"sigma2_mean={state.sigma2_mean: .4f}"
            )

    return history


__all__ = [
    "TrainConfig",
    "TrainState",
    "TrainHistory",
    "train_variational_model",
]