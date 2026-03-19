from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch

from latent_svi_spatial.data.synthetic import SyntheticConfig, generate_synthetic_panel
from latent_svi_spatial.train.trainer import TrainConfig, train_variational_model
from latent_svi_spatial.vi.variational_family import VariationalFamily


OUTPUT_DIR = Path("outputs/experiments/structured_vi_variant")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]

N = 100
T = 8
R = 3
P_EFF = 4
RHO_MAX = 0.05
N_MC_SAMPLES = 5


def frobenius_error(A: torch.Tensor, B: torch.Tensor) -> float:
    return float(torch.norm(A - B, p="fro").item())


def run_single(seed: int, deterministic_C: bool) -> dict:
    torch.manual_seed(seed)

    try:
        data = generate_synthetic_panel(
            SyntheticConfig(
                n=N,
                t=T,
                p=P_EFF - 1,   # include_intercept=True => total effective dim is P_EFF
                r=R,
                seed=seed,
                normalize_w="none",
                rho=0.01,
                sigma2=0.25,
                include_intercept=True,
                enforce_stability=True,
                stability_margin=0.95,
            )
        )

        vf = VariationalFamily(
            n=N,
            r=R,
            p_eff=P_EFF,
            rho_max=RHO_MAX,
            deterministic_C=deterministic_C,
            symmetric_c=True,
            dtype=data.X.dtype,
            device=data.X.device,
        )

        start = time.time()

        history = train_variational_model(
            variational_family=vf,
            X=data.X,
            y=data.y,
            true_W=data.W,
            config=TrainConfig(
                n_steps=300,
                lr=1e-2,
                n_mc_samples=N_MC_SAMPLES,
                stability_penalty_weight=10.0,
                hard_stability_penalty_weight=100.0,
                stability_margin=0.05,
                unstable_logdet_value=-1e4,
                verbose=False,
            ),
        )

        runtime = time.time() - start
        last = history.records[-1]

        rho_mean = vf.mean_rho()
        sigma2_mean = vf.mean_sigma2()
        beta_mean = vf.mean_beta()
        H_mean = vf.mean_H()
        C_mean = vf.mean_C()

        W_mean = H_mean @ C_mean @ H_mean.T
        W_mean = W_mean - torch.diag(torch.diag(W_mean))

        rmse = float(last.predictive_rmse)

        return {
            "seed": seed,
            "deterministic_C": deterministic_C,
            "status": "ok",
            "error_msg": "",
            "rho_error": abs(float(rho_mean) - float(data.rho)),
            "sigma2_error": abs(float(sigma2_mean) - float(data.sigma2)),
            "beta_error": frobenius_error(beta_mean, data.beta),
            "H_error": frobenius_error(H_mean, data.H),
            "C_error": frobenius_error(C_mean, data.C),
            "W_error": frobenius_error(W_mean, data.W),
            "rmse": rmse,
            "final_elbo": float(last.elbo),
            "final_stability_rate": float(last.stability_rate),
            "final_min_reduced_eig": float(last.min_reduced_eig),
            "runtime_sec": runtime,
        }

    except Exception as e:
        return {
            "seed": seed,
            "deterministic_C": deterministic_C,
            "status": "failed",
            "error_msg": str(e),
        }


def run_experiment() -> None:
    print("=" * 80)
    print("Running Experiment 4: Structured VI (deterministic C)")
    print("=" * 80)

    results = []

    for deterministic_C in [False, True]:
        label = "deterministic_C=True" if deterministic_C else "baseline (stochastic C)"
        print(f"\n--- condition: {label} ---")

        for seed in SEEDS:
            print(f"Running seed {seed}...")

            res = run_single(seed, deterministic_C)

            if res["status"] == "ok":
                print(
                    f"W_err={res['W_error']:.2f}  "
                    f"RMSE={res['rmse']:.4f}  "
                    f"rho_err={res['rho_error']:.4f}  "
                    f"time={res['runtime_sec']:.2f}s"
                )
            else:
                print(f"FAILED: {res['error_msg']}")

            results.append(res)

    df = pd.DataFrame(results)
    out_path = OUTPUT_DIR / "results.csv"
    df.to_csv(out_path, index=False)

    print("\nSaved results to:", out_path)

    df_ok = df[df["status"] == "ok"].copy()

    if len(df_ok) > 0:
        summary = (
            df_ok.groupby("deterministic_C")[
                [
                    "rho_error",
                    "W_error",
                    "rmse",
                    "runtime_sec",
                    "final_elbo",
                    "final_stability_rate",
                    "final_min_reduced_eig",
                ]
            ]
            .mean()
            .reset_index()
        )

        print("\nMean metrics:")
        print(summary.to_string(index=False))

    failure_summary = (
        df.groupby("deterministic_C")["status"]
        .apply(lambda s: (s == "failed").sum())
        .reset_index(name="n_failed")
    )

    print("\nFailure counts:")
    print(failure_summary.to_string(index=False))


if __name__ == "__main__":
    run_experiment()