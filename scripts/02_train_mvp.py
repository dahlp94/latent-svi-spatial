from __future__ import annotations

from pathlib import Path

import torch

from latent_svi_spatial.data.synthetic import SyntheticConfig, generate_synthetic_panel
from latent_svi_spatial.train.trainer import TrainConfig, train_variational_model
from latent_svi_spatial.vi.variational_family import VariationalFamily


def frobenius_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.norm(a - b).item())


def build_weight_from_factors(H: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    W_raw = H @ C @ H.T
    return W_raw - torch.diag(torch.diag(W_raw))


def main() -> None:
    # ---------------------------------------------------------
    # 1. Synthetic data config
    # ---------------------------------------------------------
    data_config = SyntheticConfig(
        n=30,
        t=8,
        p=3,
        r=3,
        seed=123,
        normalize_w="none",
        rho=0.01,
        sigma2=0.25,
        c_scale=0.5,
        membership_prior="dirichlet",
        include_intercept=True,
    )

    data = generate_synthetic_panel(data_config)

    T, N, p_eff = data.X.shape
    r = data.H.shape[1]

    print("=" * 72)
    print("Synthetic dataset generated")
    print("=" * 72)
    print(f"T={T}, N={N}, p_eff={p_eff}, r={r}")
    print(f"True rho     : {float(data.rho):.6f}")
    print(f"True sigma2  : {float(data.sigma2):.6f}")
    print(f"||W||_F      : {float(torch.norm(data.W).item()):.6f}")
    print()

    # ---------------------------------------------------------
    # 2. Variational family
    # ---------------------------------------------------------
    vf = VariationalFamily(
        n=N,
        r=r,
        p_eff=p_eff,
        rho_max=0.05,
        symmetric_c=True,
        dtype=data.X.dtype,
        device=data.X.device,
    )

    print("=" * 72)
    print("Initial variational summary")
    print("=" * 72)
    for k, v in vf.summary().items():
        print(f"{k:15s}: {v:.6f}")
    print()

    # ---------------------------------------------------------
    # 3. Train config
    # ---------------------------------------------------------
    train_config = TrainConfig(
        n_steps=300,
        lr=1e-2,
        n_mc_samples=1,
        clip_grad_norm=5.0,
        log_every=25,
        verbose=True,
    )

    # ---------------------------------------------------------
    # 4. Train
    # ---------------------------------------------------------
    print("=" * 72)
    print("Training variational model")
    print("=" * 72)

    history = train_variational_model(
        variational_family=vf,
        X=data.X,
        y=data.y,
        config=train_config,
    )

    print()
    print("=" * 72)
    print("Training complete")
    print("=" * 72)

    last = history.last()
    if last is None:
        raise RuntimeError("Training history is empty.")

    print(f"Final ELBO    : {last.elbo:.6f}")
    print(f"Final Loss    : {last.loss:.6f}")
    print(f"Final LogLik  : {last.expected_log_likelihood:.6f}")
    print(f"Final KL      : {last.kl_total:.6f}")
    print(f"Mean rho      : {last.rho_mean:.6f}")
    print(f"Mean sigma2   : {last.sigma2_mean:.6f}")
    print()

    # ---------------------------------------------------------
    # 5. Compare learned means to truth
    # ---------------------------------------------------------
    H_mean = vf.mean_H().detach()
    C_mean = vf.mean_C().detach()
    rho_mean = vf.mean_rho().detach()
    beta_mean = vf.mean_beta().detach()
    sigma2_mean = vf.mean_sigma2().detach()

    W_mean = build_weight_from_factors(H_mean, C_mean)

    # Deterministic fitted mean under posterior means:
    # Solve (I - rho W) y_hat_t = X_t beta
    Xbeta_mean = torch.einsum("tnp,p->tn", data.X, beta_mean)
    A_mean = torch.eye(N, dtype=data.X.dtype, device=data.X.device) - rho_mean * W_mean
    y_hat_mean = torch.linalg.solve(A_mean, Xbeta_mean.T).T

    predictive_rmse = torch.sqrt(torch.mean((y_hat_mean - data.y) ** 2)).item()

    print("=" * 72)
    print("Comparison to ground truth")
    print("=" * 72)
    print(f"rho error         : {abs(float(rho_mean) - float(data.rho)):.6f}")
    print(f"sigma2 error      : {abs(float(sigma2_mean) - float(data.sigma2)):.6f}")
    print(f"beta Frobenius err: {frobenius_error(beta_mean, data.beta):.6f}")
    print(f"H Frobenius err   : {frobenius_error(H_mean, data.H):.6f}")
    print(f"C Frobenius err   : {frobenius_error(C_mean, data.C):.6f}")
    print(f"W Frobenius err   : {frobenius_error(W_mean, data.W):.6f}")
    print(f"Predictive RMSE   : {predictive_rmse:.6f}")
    print()

    # ---------------------------------------------------------
    # 6. Save lightweight artifacts
    # ---------------------------------------------------------
    output_dir = Path("outputs") / "runs" / "mvp_train"
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / "history.pt"
    torch.save(history.to_dict(), history_path)

    summary_path = output_dir / "summary.pt"
    torch.save(
        {
            "true_rho": data.rho.detach().cpu(),
            "true_sigma2": data.sigma2.detach().cpu(),
            "true_beta": data.beta.detach().cpu(),
            "true_H": data.H.detach().cpu(),
            "true_C": data.C.detach().cpu(),
            "true_W": data.W.detach().cpu(),
            "mean_rho": rho_mean.detach().cpu(),
            "mean_sigma2": sigma2_mean.detach().cpu(),
            "mean_beta": beta_mean.detach().cpu(),
            "mean_H": H_mean.detach().cpu(),
            "mean_C": C_mean.detach().cpu(),
            "mean_W": W_mean.detach().cpu(),
            "y_hat_mean": y_hat_mean.detach().cpu(),
            "predictive_rmse": predictive_rmse,
            "rho_error": abs(float(rho_mean) - float(data.rho)),
            "sigma2_error": abs(float(sigma2_mean) - float(data.sigma2)),
            "beta_error": frobenius_error(beta_mean, data.beta),
            "H_error": frobenius_error(H_mean, data.H),
            "C_error": frobenius_error(C_mean, data.C),
            "W_error": frobenius_error(W_mean, data.W),
        },
        summary_path,
    )

    print("=" * 72)
    print("Artifacts saved")
    print("=" * 72)
    print(f"History saved to: {history_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()