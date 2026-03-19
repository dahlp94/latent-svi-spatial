from __future__ import annotations

import torch

Tensor = torch.Tensor


# ============================================================
# Core system matrix pieces
# ============================================================

def compute_A_dense(W: Tensor, rho: float | Tensor) -> Tensor:
    """
    Compute A = I - rho W (dense version, for testing/debug only)
    """
    n = W.shape[0]
    I = torch.eye(n, device=W.device, dtype=W.dtype)
    return I - rho * W


def apply_A_to_y(
    H: Tensor,
    C: Tensor,
    y: Tensor,
    rho: float | Tensor,
    *,
    zero_diagonal: bool = True,
) -> Tensor:
    """
    Compute A y WITHOUT forming A explicitly.

    A = D - rho H C H^T
    where D = I + rho diag(H C H^T)

    y: (..., N)

    Returns:
        Ay: (..., N)
    """
    # Step 1: g = H^T y
    g = torch.matmul(y, H)  # (..., r)

    # Step 2: H C g
    HCg = torch.matmul(g, C.T)  # (..., r)
    HCHTy = torch.matmul(HCg, H.T)  # (..., N)

    # Step 3: diagonal correction
    if zero_diagonal:
        diag_vals = torch.sum(H * (H @ C), dim=1)  # (N,)
    else:
        diag_vals = torch.zeros(H.shape[0], device=H.device, dtype=H.dtype)

    D_y = y + rho * diag_vals * y

    Ay = D_y - rho * HCHTy
    return Ay


# ============================================================
# Low-rank determinant
# ============================================================

def compute_D(
    H: Tensor,
    C: Tensor,
    rho: float | Tensor,
) -> Tensor:
    """
    Compute diagonal D = I + rho diag(H C H^T)
    Returns vector of size (N,)
    """
    HC = H @ C  # (N, r)
    diag_vals = torch.sum(H * HC, dim=1)  # (N,)
    return 1.0 + rho * diag_vals


def compute_reduced_stability_matrix(
    H: Tensor,
    C: Tensor,
    rho: float | Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute the reduced r x r matrix

        M = I_r - rho C H^T D^{-1} H

    where
        D = I + rho diag(H C H^T)

    This is the matrix whose determinant appears in the low-rank logdet identity.
    """
    _, r = H.shape

    D = compute_D(H, C, rho)
    D_safe = torch.clamp(D, min=eps)

    D_inv = 1.0 / D_safe
    HD = H * D_inv.unsqueeze(1)
    Ht_Dinv_H = torch.matmul(H.T, HD)

    M = torch.eye(r, device=H.device, dtype=H.dtype) - rho * (C @ Ht_Dinv_H)
    return M


def logdet_A_lowrank(
    H: Tensor,
    C: Tensor,
    rho: float | Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute log|A| using low-rank identity:

        log|A| = sum log D_ii + log|I - rho C H^T D^{-1} H|
    """
    D = compute_D(H, C, rho)
    D_safe = torch.clamp(D, min=eps)

    logdet_D = torch.sum(torch.log(D_safe))

    M = compute_reduced_stability_matrix(H, C, rho, eps=eps)

    sign, logabsdet = torch.linalg.slogdet(M)
    if torch.any(sign <= 0):
        raise ValueError("Low-rank inner matrix has non-positive determinant.")

    return logdet_D + logabsdet

def safe_logdet_A_lowrank(
    H: Tensor,
    C: Tensor,
    rho: float | Tensor,
    *,
    eps: float = 1e-8,
    unstable_logdet_value: float = -1e6,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Safe version of the low-rank logdet.

    Returns
    -------
    logdet_value:
        If stable, the true log|A|. If unstable, a large negative surrogate.
    is_stable:
        Scalar tensor in {0,1} indicating whether the reduced matrix had positive determinant.
    min_real_eig:
        Minimum real part of eigenvalues of the reduced matrix M.

    Notes
    -----
    This function never raises on non-positive determinant. It is designed for training-time use.
    """
    D = compute_D(H, C, rho)
    D_safe = torch.clamp(D, min=eps)

    logdet_D = torch.sum(torch.log(D_safe))

    M = compute_reduced_stability_matrix(H, C, rho, eps=eps)

    sign, logabsdet = torch.linalg.slogdet(M)

    eigvals = torch.linalg.eigvals(M)
    min_real_eig = eigvals.real.min()

    is_stable = (sign > 0).to(H.dtype)

    fallback = torch.tensor(
        unstable_logdet_value,
        device=H.device,
        dtype=H.dtype,
    )

    logdet_small = torch.where(sign > 0, logabsdet, fallback)
    logdet_value = logdet_D + logdet_small

    return logdet_value, is_stable, min_real_eig

def stability_penalty(
    H: Tensor,
    C: Tensor,
    rho: float | Tensor,
    *,
    margin: float = 0.05,
    soft_weight: float = 1.0,
    hard_weight: float = 100.0,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Stability penalty for the reduced matrix M = I_r - rho C H^T D^{-1} H.

    Returns
    -------
    penalty:
        soft_weight * relu(margin - min_real_eig)^2
        + hard_weight * 1{sign <= 0}
    is_stable:
        Scalar tensor in {0,1}
    min_real_eig:
        Minimum real part of eigenvalues of M

    Notes
    -----
    The hard term is essential because instability occurs exactly at the logdet boundary.
    """
    M = compute_reduced_stability_matrix(H, C, rho, eps=eps)

    sign, _ = torch.linalg.slogdet(M)
    eigvals = torch.linalg.eigvals(M)
    min_real_eig = eigvals.real.min()

    margin_t = torch.tensor(margin, device=M.device, dtype=M.dtype)
    soft_gap = torch.relu(margin_t - min_real_eig)
    soft_term = soft_weight * soft_gap.pow(2)

    hard_indicator = (sign <= 0).to(M.dtype)
    hard_term = hard_weight * hard_indicator

    penalty = soft_term + hard_term
    is_stable = (sign > 0).to(M.dtype)

    return penalty, is_stable, min_real_eig

# ============================================================
# Woodbury inverse application
# ============================================================

def solve_A_inv_y(
    H: Tensor,
    C: Tensor,
    y: Tensor,
    rho: float | Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    Compute x = A^{-1} y using Woodbury identity.

    A^{-1} = D^{-1} + D^{-1} H (rho^{-1} C^{-1} - H^T D^{-1} H)^{-1} H^T D^{-1}
    """
    N, r = H.shape

    # D
    D = compute_D(H, C, rho)
    D_safe = torch.clamp(D, min=eps)
    D_inv = 1.0 / D_safe

    # Step 1: D^{-1} y
    Dy = D_inv * y  # (N,)

    # Step 2: compute middle matrix
    HD = H * D_inv.unsqueeze(1)
    Ht_Dinv_H = H.T @ HD  # (r, r)

    C_inv = torch.inverse(C)
    middle = (1.0 / rho) * C_inv - Ht_Dinv_H

    middle_inv = torch.inverse(middle)

    # Step 3: compute correction
    Hy = H.T @ Dy  # (r,)
    correction = H @ (middle_inv @ Hy)

    x = Dy + D_inv * correction
    return x


# ============================================================
# Utility checks (VERY IMPORTANT)
# ============================================================

def check_logdet_consistency(
    W: Tensor,
    H: Tensor,
    C: Tensor,
    rho: float,
) -> float:
    """
    Compare dense logdet vs low-rank logdet
    """
    A_dense = compute_A_dense(W, rho)
    sign_dense, logabsdet_dense = torch.linalg.slogdet(A_dense)
    if sign_dense <= 0:
        raise ValueError("Dense A has non-positive determinant.")
    logdet_dense = logabsdet_dense

    logdet_lr = logdet_A_lowrank(H, C, rho)

    return float((logdet_dense - logdet_lr).abs().item())


def check_inverse_consistency(
    W: Tensor,
    H: Tensor,
    C: Tensor,
    rho: float,
    y: Tensor,
) -> float:
    """
    Compare dense inverse vs Woodbury inverse
    """
    A_dense = compute_A_dense(W, rho)
    x_dense = torch.linalg.solve(A_dense, y)

    x_lr = solve_A_inv_y(H, C, y, rho)

    return float(torch.norm(x_dense - x_lr).item())

__all__ = [
    "apply_A_to_y",
    "check_inverse_consistency",
    "check_logdet_consistency",
    "compute_A_dense",
    "compute_D",
    "compute_reduced_stability_matrix",
    "logdet_A_lowrank",
    "safe_logdet_A_lowrank",
    "solve_A_inv_y",
    "stability_penalty",
]