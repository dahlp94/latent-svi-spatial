from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch

from latent_svi_spatial.data.synthetic import SyntheticConfig, generate_synthetic_panel
from latent_svi_spatial.train.trainer import TrainConfig, train_variational_model
from latent_svi_spatial.vi.variational_family import VariationalFamily


def frobenius_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.norm(a - b).item())


def build_weight(H: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    W_raw = H @ C @ H.T
    return W_raw - torch.diag(torch.diag(W_raw))


def compute_rmse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((y_hat - y) ** 2)).item())


def run_single_condition(
    *,
    seed: int,
    stability_penalty_weight: float,
) -> dict:
    N = 100
    T = 8
    r = 3
    rho_max = 0.05
    n_mc_samples = 5

    try:
        data = generate_synthetic_panel(
            SyntheticConfig(
                n=N,
                t=T,
                p=3,
                r=r,
                seed=seed,
                normalize_w="none",
                rho=0.01,
                sigma2=0.25,
                include_intercept=True,
                enforce_stability=True,
                stability_margin=0.95,
            )
        )

        _, _, p_eff = data.X.shape

        vf = VariationalFamily(
            n=N,
            r=r,
            p_eff=p_eff,
            rho_max=rho_max,
            symmetric_c=True,
            dtype=data.X.dtype,
            device=data.X.device,
        )

        start_time = time.time()

        history = train_variational_model(
            variational_family=vf,
            X=data.X,
            y=data.y,
            true_W=data.W,
            config=TrainConfig(
                n_steps=300,
                lr=1e-2,
                n_mc_samples=n_mc_samples,
                stability_penalty_weight=stability_penalty_weight,
                hard_stability_penalty_weight=100.0,
                stability_margin=0.05,
                unstable_logdet_value=-1e4,
                verbose=False,
            ),
        )

        runtime = time.time() - start_time

        H_mean = vf.mean_H().detach()
        C_mean = vf.mean_C().detach()
        rho_mean = vf.mean_rho().detach()
        beta_mean = vf.mean_beta().detach()
        sigma2_mean = vf.mean_sigma2().detach()

        W_mean = build_weight(H_mean, C_mean)

        Xbeta = torch.einsum("tnp,p->tn", data.X, beta_mean)
        A = torch.eye(N, dtype=data.X.dtype, device=data.X.device) - rho_mean * W_mean
        y_hat = torch.linalg.solve(A, Xbeta.T).T
        rmse = compute_rmse(y_hat, data.y)

        last = history.last()
        if last is None:
            raise RuntimeError("Training history is empty.")

        return {
            "seed": seed,
            "N": N,
            "T": T,
            "r": r,
            "rho_max": rho_max,
            "n_mc_samples": n_mc_samples,
            "stability_penalty_weight": stability_penalty_weight,
            "status": "ok",
            "error_msg": "",
            "rho_true": float(data.rho),
            "rho_est": float(rho_mean),
            "rho_error": abs(float(rho_mean) - float(data.rho)),
            "sigma2_true": float(data.sigma2),
            "sigma2_est": float(sigma2_mean),
            "sigma2_error": abs(float(sigma2_mean) - float(data.sigma2)),
            "beta_error": frobenius_error(beta_mean, data.beta),
            "W_error": frobenius_error(W_mean, data.W),
            "rmse": rmse,
            "final_elbo": float(last.elbo),
            "final_loglik": float(last.expected_log_likelihood),
            "final_kl": float(last.kl_total),
            "final_stability_penalty": float(last.stability_penalty),
            "final_stability_rate": float(last.stability_rate),
            "final_min_reduced_eig": float(last.min_reduced_eig),
            "runtime_sec": runtime,
        }

    except Exception as e:
        return {
            "seed": seed,
            "N": N,
            "T": T,
            "r": r,
            "rho_max": 0.05,
            "n_mc_samples": 5,
            "stability_penalty_weight": stability_penalty_weight,
            "status": "failed",
            "error_msg": str(e),
            "rho_true": None,
            "rho_est": None,
            "rho_error": None,
            "sigma2_true": None,
            "sigma2_est": None,
            "sigma2_error": None,
            "beta_error": None,
            "W_error": None,
            "rmse": None,
            "final_elbo": None,
            "final_loglik": None,
            "final_kl": None,
            "final_stability_penalty": None,
            "final_stability_rate": None,
            "final_min_reduced_eig": None,
            "runtime_sec": None,
        }


def main() -> None:
    seeds = [0, 1, 2, 3, 4]
    penalty_grid = [0.0, 10.0, 50.0]

    results: list[dict] = []

    print("=" * 80)
    print("Running Experiment 3: stability penalty ablation at N=100")
    print("=" * 80)

    for penalty_weight in penalty_grid:
        print(f"\n--- condition: stability_penalty_weight={penalty_weight:.1f} ---")

        for seed in seeds:
            print(f"Running seed {seed}...")

            res = run_single_condition(
                seed=seed,
                stability_penalty_weight=penalty_weight,
            )
            results.append(res)

            if res["status"] == "ok":
                print(
                    f"rho_err={res['rho_error']:.4f}  "
                    f"W_err={res['W_error']:.2f}  "
                    f"RMSE={res['rmse']:.4f}  "
                    f"StabPen={res['final_stability_penalty']:.4f}  "
                    f"time={res['runtime_sec']:.2f}s"
                )
            else:
                print(f"FAILED: {res['error_msg']}")

    df = pd.DataFrame(results)

    output_dir = Path("outputs") / "experiments" / "stability_penalty_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("Experiment complete")
    print("=" * 80)
    print(f"Saved to: {csv_path}")

    df_ok = df[df["status"] == "ok"].copy()

    if len(df_ok) > 0:
        summary = (
            df_ok.groupby("stability_penalty_weight")[
                [
                    "rho_error",
                    "W_error",
                    "rmse",
                    "runtime_sec",
                    "final_elbo",
                    "final_stability_penalty",
                    "final_stability_rate",
                    "final_min_reduced_eig",
                ]
            ]
            .mean()
            .reset_index()
        )

        print("\nMean metrics by penalty weight:")
        print(summary.to_string(index=False))

    failure_summary = (
        df.groupby("stability_penalty_weight")["status"]
        .apply(lambda s: (s == "failed").sum())
        .reset_index(name="n_failed")
    )

    print("\nFailure counts by penalty weight:")
    print(failure_summary.to_string(index=False))


if __name__ == "__main__":
    main()