from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


def _softplus_positive(x: Tensor, min_value: float = 1e-6) -> Tensor:
    return F.softplus(x) + min_value


@dataclass
class VariationalSample:
    H: Tensor
    C: Tensor
    rho: Tensor
    beta: Tensor
    sigma2: Tensor

    U: Tensor
    C_raw: Tensor
    rho_raw: Tensor
    beta_raw: Tensor
    log_sigma2_raw: Tensor


class GaussianParameter(nn.Module):
    """
    Diagonal Gaussian variational block.
    """
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        init_mean: float = 0.0,
        init_mean_noise: float = 0.01,
        init_log_scale: float = -3.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()

        mean = (
            torch.full(shape, init_mean, device=device, dtype=dtype)
            + init_mean_noise * torch.randn(*shape, device=device, dtype=dtype)
        )
        log_scale = torch.full(shape, init_log_scale, device=device, dtype=dtype)

        self.mean = nn.Parameter(mean)
        self.log_scale = nn.Parameter(log_scale)

    @property
    def scale(self) -> Tensor:
        return _softplus_positive(self.log_scale)

    def rsample(self) -> tuple[Tensor, Tensor]:
        eps = torch.randn_like(self.mean)
        sample = self.mean + self.scale * eps
        return sample, eps

    def kl_to_standard_normal(self) -> Tensor:
        scale2 = self.scale.pow(2)
        mean2 = self.mean.pow(2)
        return 0.5 * torch.sum(mean2 + scale2 - 1.0 - torch.log(scale2))


class VariationalFamily(nn.Module):
    """
    Variational family for the low-rank SAR MVP.

    Variant 1 structured VI:
      deterministic_C=True makes C deterministic while keeping H stochastic.
    """
    def __init__(
        self,
        n: int,
        r: int,
        p_eff: int,
        *,
        rho_max: float = 0.10,
        symmetric_c: bool = True,
        deterministic_C: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()

        if n <= 0:
            raise ValueError("n must be positive.")
        if r <= 0:
            raise ValueError("r must be positive.")
        if p_eff <= 0:
            raise ValueError("p_eff must be positive.")
        if not (0.0 < rho_max <= 1.0):
            raise ValueError("rho_max must be in (0, 1].")

        self.n = n
        self.r = r
        self.p_eff = p_eff
        self.rho_max = rho_max
        self.symmetric_c = symmetric_c
        self.deterministic_C = deterministic_C
        self.device = device
        self.dtype = dtype

        # U -> H = softmax(U)
        self.q_U = GaussianParameter(
            (n, r),
            init_mean=0.0,
            init_mean_noise=0.01,
            init_log_scale=-3.0,
            device=device,
            dtype=dtype,
        )

        # C parameterization
        if not self.deterministic_C:
            self.q_C_raw = GaussianParameter(
                (r, r),
                init_mean=-2.0,
                init_mean_noise=0.01,
                init_log_scale=-3.5,
                device=device,
                dtype=dtype,
            )
        else:
            c_mean = (
                torch.full((r, r), -2.0, device=device, dtype=dtype)
                + 0.01 * torch.randn(r, r, device=device, dtype=dtype)
            )
            self.C_mean_raw = nn.Parameter(c_mean)

        # rho_raw -> rho = rho_max * sigmoid(rho_raw)
        self.q_rho_raw = GaussianParameter(
            (1,),
            init_mean=-4.0,
            init_mean_noise=0.01,
            init_log_scale=-4.0,
            device=device,
            dtype=dtype,
        )

        # beta
        self.q_beta = GaussianParameter(
            (p_eff,),
            init_mean=0.0,
            init_mean_noise=0.01,
            init_log_scale=-3.0,
            device=device,
            dtype=dtype,
        )

        # log sigma2 -> sigma2 = exp(log sigma2)
        self.q_log_sigma2 = GaussianParameter(
            (1,),
            init_mean=-1.0,
            init_mean_noise=0.01,
            init_log_scale=-4.0,
            device=device,
            dtype=dtype,
        )

    def sample_H(self) -> tuple[Tensor, Tensor]:
        U, _ = self.q_U.rsample()
        H = torch.softmax(U, dim=1)
        return H, U

    def sample_C(self) -> tuple[Tensor, Tensor]:
        """
        Sample or deterministically construct C.
        """
        if not self.deterministic_C:
            C_raw, _ = self.q_C_raw.rsample()
        else:
            C_raw = self.C_mean_raw

        C = _softplus_positive(C_raw)

        if self.symmetric_c:
            C = 0.5 * (C + C.T)

        return C, C_raw

    def sample_rho(self) -> tuple[Tensor, Tensor]:
        rho_raw, _ = self.q_rho_raw.rsample()
        rho = self.rho_max * torch.sigmoid(rho_raw)
        return rho.squeeze(), rho_raw.squeeze()

    def sample_beta(self) -> tuple[Tensor, Tensor]:
        beta, _ = self.q_beta.rsample()
        return beta, beta

    def sample_sigma2(self) -> tuple[Tensor, Tensor]:
        log_sigma2_raw, _ = self.q_log_sigma2.rsample()
        sigma2 = torch.exp(log_sigma2_raw)
        return sigma2.squeeze(), log_sigma2_raw.squeeze()

    def rsample(self) -> VariationalSample:
        H, U = self.sample_H()
        C, C_raw = self.sample_C()
        rho, rho_raw = self.sample_rho()
        beta, beta_raw = self.sample_beta()
        sigma2, log_sigma2_raw = self.sample_sigma2()

        return VariationalSample(
            H=H,
            C=C,
            rho=rho,
            beta=beta,
            sigma2=sigma2,
            U=U,
            C_raw=C_raw,
            rho_raw=rho_raw,
            beta_raw=beta_raw,
            log_sigma2_raw=log_sigma2_raw,
        )

    def mean_H(self) -> Tensor:
        return torch.softmax(self.q_U.mean, dim=1)

    def mean_C(self) -> Tensor:
        if not self.deterministic_C:
            C_raw = self.q_C_raw.mean
        else:
            C_raw = self.C_mean_raw

        C = _softplus_positive(C_raw)

        if self.symmetric_c:
            C = 0.5 * (C + C.T)

        return C

    def mean_rho(self) -> Tensor:
        return self.rho_max * torch.sigmoid(self.q_rho_raw.mean).squeeze()

    def mean_beta(self) -> Tensor:
        return self.q_beta.mean

    def mean_sigma2(self) -> Tensor:
        return torch.exp(self.q_log_sigma2.mean).squeeze()

    def kl_H(self) -> Tensor:
        return self.q_U.kl_to_standard_normal()

    def kl_C(self) -> Tensor:
        if not self.deterministic_C:
            return self.q_C_raw.kl_to_standard_normal()
        return torch.zeros((), device=self.q_U.mean.device, dtype=self.q_U.mean.dtype)

    def kl_rho(self) -> Tensor:
        return self.q_rho_raw.kl_to_standard_normal()

    def kl_beta(self) -> Tensor:
        return self.q_beta.kl_to_standard_normal()

    def kl_sigma2(self) -> Tensor:
        return self.q_log_sigma2.kl_to_standard_normal()

    def kl_total(self) -> Tensor:
        return (
            self.kl_H()
            + self.kl_C()
            + self.kl_rho()
            + self.kl_beta()
            + self.kl_sigma2()
        )

    def summary(self) -> dict[str, float]:
        with torch.no_grad():
            return {
                "rho_mean": float(self.mean_rho().item()),
                "sigma2_mean": float(self.mean_sigma2().item()),
                "kl_H": float(self.kl_H().item()),
                "kl_C": float(self.kl_C().item()),
                "kl_rho": float(self.kl_rho().item()),
                "kl_beta": float(self.kl_beta().item()),
                "kl_sigma2": float(self.kl_sigma2().item()),
                "kl_total": float(self.kl_total().item()),
                "deterministic_C": float(self.deterministic_C),
            }


__all__ = [
    "VariationalSample",
    "GaussianParameter",
    "VariationalFamily",
]