import numpy as np

from qse.config import NumericalConfig
from qse.linalg.hermitian import hermitian_basis,vec_hermitian


from qse.linalg.numerical_rank import numerical_rank



def operator_span_dimension(
    subspace_bases: list[np.ndarray],
    config: NumericalConfig,
    lam_min_kept: float = 1.0,
) -> int:
    """
    Tính:

        dim_R sum_i L(S_i)

    hoặc:

        dim_R sum_i L(K_i).
    """

    if len(subspace_bases) == 0:
        return 0

    columns = []

    for Q in subspace_bases:

        d = Q.shape[1]

        for E in hermitian_basis(d):

            A = (Q @ E @ Q.conj().T)

            columns.append(vec_hermitian(A))

    if len(columns) == 0:
        return 0

    B = np.column_stack(
        columns
    )

    s = np.linalg.svd(B, compute_uv=False)
    if len(s) == 0:
        return 0
    n = int(round(np.sqrt(B.shape[0])))
    return int(np.sum(s > config.tol_null(n, s[0], lam_min_kept)))


def compute_D_S(support_bases, config, lam_min_kept=1.0):
    return operator_span_dimension(support_bases, config, lam_min_kept)


def compute_D_K(
    kernel_bases,
    config,
):
    return operator_span_dimension(
        kernel_bases,
        config
    )