import numpy as np

from qse.config import NumericalConfig
from qse.linalg.numerical_rank import numerical_rank

from dataclasses import dataclass


@dataclass
class IntersectionReport:
    max_dimension: int
    worst_pair: tuple[int, int] | None
    pair_dimensions: dict[tuple[int, int], int]

def intersection_dimension(
    QA: np.ndarray,
    QB: np.ndarray,
    config: NumericalConfig,
    lam_min_kept: float = 1.0,
) -> int:
    """
    dim(A intersection B)

      = dim(A)
      + dim(B)
      - dim(A + B).
    """

    dA = QA.shape[1]
    dB = QB.shape[1]

    if dA == 0 or dB == 0:
        return 0

    QAB = np.hstack( [QA, QB])

    # sigma_min([QA QB]) ~ sin(góc chính) -> cùng ngưỡng Davis–Kahan như Phi
    sv = np.linalg.svd(QAB, compute_uv=False)
    n = QA.shape[0]
    dim_sum = int(np.sum(sv > config.tol_null(n, sv[0], lam_min_kept)))

    return int(
        dA + dB - dim_sum
    )


def max_pairwise_intersection(
    support_bases,
    config,
    lam_min_kept=1.0,
):
    """
    max_{i != j} dim(S_i intersection S_j).
    """
    pair_dims = {}

    max_dim = 0
    worst_pair = None
    

    k = len(support_bases)

    for i in range(k):
        for j in range(i + 1, k):

            d = intersection_dimension(
                support_bases[i],
                support_bases[j],
                config,
                lam_min_kept,
            )

            pair_dims[(i, j)] = d
            if d > max_dim:
                max_dim = d
                worst_pair = (i, j)

    return IntersectionReport(
        max_dimension=max_dim,
        worst_pair=worst_pair,
        pair_dimensions=pair_dims
    )