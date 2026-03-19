
# Structured Variational Inference for Spatial Econometric Models

### (Latent Weight Matrix Learning via Low-Rank Factorization)

---

## Project Overview

This project develops a **variational inference (VI) framework for spatial econometric models** where the spatial weight matrix $W$ is **not fixed**, but **learned endogenously** from the data.

Instead of assuming a pre-specified adjacency matrix, we model:

$$
W = H C H^\top - \mathrm{diag}(H C H^\top)
$$

where:

* $H \in \mathbb{R}^{N \times r}$: latent community memberships (simplex-constrained)
* $C \in \mathbb{R}^{r \times r}$: community interaction matrix

This enables:

* **data-driven spatial structure learning**
* **interpretable latent communities**
* **low-rank scalability**

---

## Core Contributions

This work makes the following contributions:

---

### 1. Endogenous Spatial Weight Learning

* Unlike classical SAR/CAR models, we **learn the spatial weight matrix $W$** from data via a latent low-rank structure.

---

### 2. Low-Rank Structural Parameterization

We parameterize:

$$
W = H C H^\top - \mathrm{diag}(H C H^\top)
$$

* enabling **$O(N r^2)$** computation instead of $O(N^3)$

---

### 3. Scalable Variational Inference for SAR Models

* A VI framework that combines:

  * **low-rank algebra**
  * **Woodbury identities**
  * **Monte Carlo ELBO**

---

### 4. Stability-Aware Generative Modeling

Synthetic data generation enforces:

$$
|\rho| < \frac{1}{\lambda_{\max}(W)}
$$

* ensuring valid SAR processes at scale

---

### 5. Empirical Identification of VI Failure Modes

We show:

* accurate recovery of global parameters
* systematic degradation in **latent structure recovery**
* emergence of **numerical instability at scale**

---

## Model

We consider a **Spatial Autoregressive (SAR) model**:

$$
(I - \rho W) y_t = X_t \beta + \varepsilon_t, \quad
\varepsilon_t \sim \mathcal{N}(0, \sigma^2 I)
$$

with:

* $\rho$: spatial dependence parameter
* $\beta$: regression coefficients
* $\sigma^2$: noise variance

---

## Variational Inference

We use a **reparameterized variational family**:

* $q(H)$: softmax-transformed Gaussian (simplex)
* $q(C)$: positive matrix (softplus)
* $q(\rho)$: bounded via sigmoid
* $q(\beta), q(\sigma^2)$: Gaussian / log-normal

The ELBO is estimated via **Monte Carlo sampling**, with:

* low-rank log-determinant identity
* Woodbury-based inverse
* stochastic gradients (PyTorch)

---

## Project Structure

```
latent-svi-spatial/
├── README.md
├── pyproject.toml
├── requirements.txt
├── scripts/
├── src/
│   └── latent_svi_spatial/
│       ├── data/
│       │   └── synthetic.py
│       ├── models/
│       │   └── sar.py
│       ├── train/
│       │   └── trainer.py
│       └── vi/
│           ├── elbo.py
│           └── variational_family.py
└── tests/
    ├── test_elbo_shapes.py
    ├── test_logdet_identity.py
    ├── test_smoke_train.py
    ├── test_stability_bound.py
    ├── test_variational_family.py
    └── test_woodbury_inverse.py
```

---

## What Has Been Implemented

### Core Modeling

* Low-rank spatial weight construction $W = H C H^\top$
* SAR likelihood with learned $W$

### Efficient Linear Algebra

* Log-determinant via low-rank identity
* Matrix inverse via Woodbury formula

### Variational Inference

* Reparameterized sampling
* KL divergence decomposition
* Monte Carlo ELBO

### Training Pipeline

* Stable optimization with gradient clipping
* ELBO tracking
* Parameter summaries

### Testing

* Algebraic correctness (logdet, inverse)
* Variational sampling constraints
* ELBO finiteness
* End-to-end smoke test

---

## MVP Results (Synthetic Data)

Example run:

```
rho error         : 0.0012
sigma2 error      : 0.0094
beta error        : 0.1256
W error           : 8.46
Predictive RMSE   : 0.48
```

---

## Interpretation

* Scalar parameters ($\rho$, $\sigma^2$, $\beta$) are well recovered
* Learned spatial structure $W$ has significant error
* Predictive performance is reasonable (RMSE ≈ noise level)

---

## Key Insight

> Variational inference successfully recovers global parameters but struggles to accurately reconstruct the latent spatial dependency structure $W$.

This suggests:

* VI is **statistically efficient**
* but **structurally biased under mean-field assumptions**

---

## Experiment 1 — Recovery vs Problem Size

We evaluate how model performance scales with the number of spatial units (N).

### Setup

* (N \in {20, 50, 100})
* Fixed latent dimension (r = 3)
* Fixed true parameters
* 5 random seeds per setting

### Results

| N   | ρ Error | W Error | RMSE | Failures |
| --- | ------- | ------- | ---- | -------- |
| 20  | 0.0014  | 5.70    | 0.50 | 0        |
| 50  | 0.0032  | 9.29    | 0.52 | 0        |
| 100 | 0.0024  | 14.04   | 0.69 | 2        |

### Key Findings

* **$\rho$ recovery is stable across scale**
* **W recovery degrades significantly as N increases**
* **Predictive error increases moderately**
* **Numerical instability appears at larger N**

---

### Interpretation

> Mean-field VI captures global dependence strength but fails to recover complex spatial structure as dimensionality increases.


---

## Current Limitations

1. **Mean-field assumption**

   * Ignores dependence between $H$, $C$, $\rho$

2. **Non-identifiability**

   * Multiple $(H, C)$ pairs yield similar $W$

3. **ELBO prioritizes likelihood**

   * Structural accuracy is not directly optimized

---

## Observed Failure Modes

The following issues emerge empirically:

---

### 1. Structural Underfitting

* The variational family cannot capture posterior dependence between $H$, $C$, and $\rho$
* Leads to poor reconstruction of $W$

---

### 2. Identifiability Issues

* Different $(H, C)$ combinations yield similar $W$
* VI collapses toward easier modes

---

### 3. Numerical Instability

* At larger $N$, samples of $(H, C, \rho)$ violate:

$$
\det(I - \rho W) > 0
$$

* Causes ELBO failures during training

---

### 4. Objective Mismatch

* ELBO prioritizes likelihood fit over structural accuracy
* No direct penalty on $W$

---

## Phase Separation: Stability vs Structure

A key insight from our experiments is that two distinct challenges arise:

1. **Numerical feasibility**
   - Variational samples can violate SAR stability:
     $$
     \det(I - \rho W) \le 0
     $$
   - This leads to ELBO failures and undefined likelihoods

2. **Variational misspecification**
   - Even when stable, the mean-field family fails to recover $W$

---

### Important Design Principle

> Stability must be enforced before improving variational expressiveness.

If structured VI is introduced prematurely:

- optimization becomes harder
- instability persists
- improvements become difficult to interpret

---

### Our Approach

We therefore proceed in two stages:

---

### Stage 1 — Stability-Aware VI (Current Focus)

We modify the ELBO to ensure:

* valid SAR systems
* stable log-determinant computation
* well-defined gradients

This provides a **reliable optimization baseline**.

---

### Stage 2 — Structured Variational Inference (Next Step)

Once stability is ensured, we introduce:

* dependencies between $H$ and $C$
* richer posterior families

to improve **structural recovery of $W$**.

---

## Why This Problem Is Difficult

Learning spatial dependence structures is fundamentally challenging because:

* The likelihood depends on (W) **nonlinearly through log-determinants**
* Stability constraints impose **non-convex feasible regions**
* Latent factorization introduces **non-identifiability**
* Posterior dependencies are **strong and global**


---

## Next Steps (Research Directions)

### 1. Stability-Aware Variational Inference (Immediate Next Step)

* Add stability constraints to ELBO
* Prevent invalid SAR samples
* Improve numerical robustness at large N

---

### 2. Structured Variational Inference (Next Phase)

* Introduce dependencies:

  * $q(H, C) \neq q(H) q(C)$
* Improve structural recovery of $W$
---

### 3. W-aware Objectives

* Add penalties:

  * $|W|$, spectral constraints, smoothness

---

### 4. Spectral Regularization

* Control eigenstructure of $W$
* Connect to graph filters (SDM-CAR)

---

### 5. Improved Variational Families

* Normalizing flows
* Low-rank covariance
* Hierarchical priors

---

### 6. Experiment Suite

Planned experiments:

* Parameter recovery vs $N, T, r$
* Comparison:

  * Mean-field VI vs structured VI
* Sensitivity to spatial strength $\rho$
* Runtime scalability

---

## Goal

Develop a **scalable, interpretable, and statistically robust VI framework** for:

* spatial econometrics
* network-based dependence modeling
* endogenous graph learning

---

## Why This Matters

* Moves beyond fixed spatial weights
* Bridges:

  * econometrics
  * Bayesian inference
  * network modeling
* Enables **data-driven discovery of spatial structure**

---

## Toward Structured Variational Inference

This work motivates the development of:

* **Structured VI with dependent factors**
* **Stability-constrained variational families**
* **Spectral regularization of learned graphs**
* **Flow-based posterior approximations**

The goal is to move from:

> Mean-field VI → Structure-aware inference for learned spatial graphs

---

## How to Run

```bash
pip install -e .
pytest
python -m scripts.02_train_mvp
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{latent_svi_spatial2026,
  title        = {Structured Variational Inference for Spatial Econometric Models with Endogenously Learned Weight Matrices},
  author       = {Pratik Dahal},
  year         = {2026},
  note   = {Contact: pd006@uark.edu, mapratikdahal@gmail.com},
}
```
