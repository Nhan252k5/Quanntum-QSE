import numpy as np

from qse.config import NumericalConfig
from qse.linalg import margins


def numerical_rank(A, rtol: float, atol: float) -> int:
    A = np.asarray(A)
    if A.size == 0:
        return 0
    s = np.linalg.svd(A, compute_uv=False)
    if len(s) == 0:
        return 0
    return int(np.sum(s > max(atol, rtol * s[0])))


def psd_support(M, config: NumericalConfig, tag="rank"):
    """
    Support số của M (0 <= M <= I nên dùng ngưỡng tuyệt đối):
        lambda > tol_rank(n) = 1e3 * noise.
    tol_rank >= 100 x tol_psd, nên nhiễu solver (|lambda| <~ tol_psd)
    không bao giờ bị tính là hạng.
    """
    M = (M + M.conj().T) / 2
    n = M.shape[0]
    eigvals, eigvecs = np.linalg.eigh(M)
    tol = config.tol_rank(n)
    margins.check(tag, eigvals, tol, config)
    mask = eigvals > tol
    return eigvecs[:, mask], int(mask.sum()), eigvals


def min_kept_eigenvalue(Ms, config: NumericalConfig) -> float:
    """lambda nhỏ nhất còn được giữ trong support (dùng cho ngưỡng Davis–Kahan)."""
    n = Ms[0].shape[0]
    tol = config.tol_rank(n)
    kept = [l for M in Ms for l in np.linalg.eigvalsh((M + M.conj().T) / 2) if l > tol]
    return min(kept) if kept else 1.0


def renormalize(Ms):
    """
    M_i <- S^{-1/2} M_i S^{-1/2},  S = sum M_i.
    Khôi phục sum M_i = I chính xác mà KHÔNG đổi hạng từng khối
    (khác với PSD-clip từng khối, vốn phá completeness).
    """
    S = sum(Ms)
    S = (S + S.conj().T) / 2
    w, V = np.linalg.eigh(S)
    if w.min() <= 0.5:
        # S quá xa I: chuẩn hóa sẽ khuếch đại sai số -> không dùng được
        raise ValueError(f"renormalize: lambda_min(sum M) = {w.min():.2e} quá nhỏ")
    T = V @ np.diag(w ** -0.5) @ V.conj().T
    out = []
    for M in Ms:
        X = T @ M @ T
        out.append((X + X.conj().T) / 2)
    return out
