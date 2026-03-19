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
    N, r = H.shape

    # Compute D
    D = compute_D(H, C, rho)  # (N,)
    D_safe = torch.clamp(D, min=eps)

    logdet_D = torch.sum(torch.log(D_safe))

    # Compute H^T D^{-1} H
    D_inv = 1.0 / D_safe  # (N,)
    HD = H * D_inv.unsqueeze(1)  # (N, r)
    Ht_Dinv_H = torch.matmul(H.T, HD)  # (r, r)

    # Inner matrix
    M = torch.eye(r, device=H.device, dtype=H.dtype) - rho * (C @ Ht_Dinv_H)

    sign, logabsdet = torch.linalg.slogdet(M)
    if torch.any(sign <= 0):
        raise ValueError("Low-rank inner matrix has non-positive determinant.")
    return logdet_D + logabsdet


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