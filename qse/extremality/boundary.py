import numpy as np

from qse.config import NumericalConfig


def block_step_interval(
    M: np.ndarray,
    Delta: np.ndarray,
    QS: np.ndarray,
    config: NumericalConfig,
):
    """
    Tìm interval t sao cho:

        M + t Delta >= 0

    trên support S = range(M).

    Ta nén:

        B = QS^dag M QS > 0
        H = QS^dag Delta QS

    rồi xét:

        B + t H >= 0

    tương đương:

        I + t C >= 0

    với:

        C = B^{-1/2} H B^{-1/2}

    Nếu mu là eigenvalue của C thì:

        1 + t mu >= 0.

    Boundary:
        t = -1/mu.
    """

    r = QS.shape[1]

    if r == 0:
        return -np.inf, np.inf

    B = ( QS.conj().T @ M @ QS)

    H = ( QS.conj().T @ Delta @ QS)

    B = ( B + B.conj().T) / 2

    H = ( H + H.conj().T) / 2

    eig_B, U = np.linalg.eigh(B)

    if eig_B.min() <= config.rank_atol:
        raise FloatingPointError(
            "M no positive definite "
            "in numerical support."
        )

    inv_sqrt = (U @ np.diag(1.0 / np.sqrt(eig_B)) @ U.conj().T)

    C = (inv_sqrt @ H@ inv_sqrt )

    C = (C + C.conj().T) / 2

    mus = np.linalg.eigvalsh(C)

    scale = max(1.0, 
                np.max(np.abs(mus))if len(mus) else 1.0)

    mu_tol = max(
        config.boundary_atol,
        config.boundary_rtol * scale
    )

    lower = -np.inf
    upper = np.inf

    for mu in mus:

        if abs(mu) <= mu_tol:
            continue

        t = -1.0 / mu

        if t < 0:
            lower = max(lower, t)

        elif t > 0:
            upper = min(upper,t)

    return lower, upper


def global_step_interval(
    Ms,
    Deltas,
    support_bases,
    config,
):
    """
    Vì tất cả block phải cùng PSD:

        interval global
            = giao interval_i.
    """

    lower = -np.inf
    upper = np.inf

    for M, Delta, QS in zip(
        Ms,
        Deltas,
        support_bases,
    ):

        lo, hi = block_step_interval(M,Delta,QS,config)

        lower = max(lower,lo)

        upper = min(upper,hi)

    if lower > upper:
        raise RuntimeError(
            "Không tồn tại interval PSD."
        )

    return lower, upper


def choose_boundary_step(
    lower: float,
    upper: float,
    config: NumericalConfig,
) -> float:
    """
    Chọn boundary gần t=0 nhất.

    Mục đích:
        đưa ít nhất một eigenvalue về 0
        => giảm rank.
    """

    candidates = []

    if np.isfinite(lower):
        candidates.append(lower)

    if np.isfinite(upper):
        candidates.append(upper)

    if len(candidates) == 0:
        raise RuntimeError(
            "Direction không có PSD boundary."
        )

    t = min( candidates,key=abs)

    if abs(t) <= config.boundary_step_tol:
        raise FloatingPointError(
            "Boundary step quá nhỏ."
        )

    return float(t)