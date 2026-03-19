# Structured Variational Inference for Spatial Econometric Models

### (Latent Weight Matrix Learning via Low-Rank Factorization)

---

## Project Overview

This project develops a **variational inference (VI) framework for spatial econometric models** where the spatial weight matrix $$ W $$ is **not fixed**, but **learned endogenously** from the data.

Instead of assuming a pre-specified adjacency matrix, we model:

$$
W = H C H^\top - \operatorname{diag}(H C H^\top)
$$

where:

* $$ H \in \mathbb{R}^{N \times r} $$: latent community memberships (simplex-constrained)
* $$ C \in \mathbb{R}^{r \times r} $$: community interaction matrix

This enables:

* **data-driven spatial structure learning**
* **interpretable latent communities**
* **low-rank scalability**

---

## Model

We consider a **Spatial Autoregressive (SAR) model**:

$$
(I - \rho W) y_t = X_t \beta + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I)
$$

with:

* $$ \rho $$: spatial dependence parameter
* $$ \beta $$: regression coefficients
* $$ \sigma^2 $$: noise variance

---

## Variational Inference

We use a **reparameterized variational family**:

* $$ q(H) $$: softmax-transformed Gaussian (simplex)
* $$ q(C) $$: positive matrix (softplus)
* $$ q(\rho) $$: bounded via sigmoid
* $$ q(\beta), q(\sigma^2) $$: Gaussian / log-normal

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

* Low-rank spatial weight construction $$ W = H C H^\top $$
* SAR likelihood with learned $$ W $$

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

* Scalar parameters ($$ \rho, \sigma^2, \beta $$) are well recovered
* Learned spatial structure $$ W $$ has significant error
* Predictive performance is reasonable (RMSE ≈ noise level)

---

## Key Insight

> Variational inference successfully recovers global parameters but struggles to accurately reconstruct the latent spatial dependency structure $$ W $$.

This suggests:

* VI is **statistically efficient**
* but **structurally biased under mean-field assumptions**

---

## Current Limitations

1. **Mean-field assumption**

   * Ignores dependence between $$ H, C, \rho $$

2. **Non-identifiability**

   * Multiple $$ (H, C) $$ pairs yield similar $$ W $$

3. **ELBO prioritizes likelihood**

   * Structural accuracy is not directly optimized

---

## Next Steps (Research Directions)

### 1. Structured Variational Inference

* Introduce dependencies:

  * $$ q(H, C) \neq q(H) q(C) $$
* Block covariance or conditional VI

---

### 2. W-aware Objectives

* Add penalties:

  * $$ |W| $$, spectral constraints, smoothness

---

### 3. Spectral Regularization

* Control eigenstructure of $$ W $$
* Connect to graph filters (SDM-CAR)

---

### 4. Improved Variational Families

* Normalizing flows
* Low-rank covariance
* Hierarchical priors

---

### 5. Experiment Suite

Planned experiments:

* Parameter recovery vs $$ N, T, r $$
* Comparison:

  * Mean-field VI vs structured VI
* Sensitivity to spatial strength $$ \rho $$
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
