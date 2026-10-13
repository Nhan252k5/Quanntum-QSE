import numpy as np
from qse.config import NumericalConfig
from qse.types import VerificationReport
from qse.states.kernels import kernel_projector

from qse.extremality.supports import compute_supports
from qse.extremality.phi import build_phi
from qse.extremality.nullspace import compute_nullspace
from qse.invariants.dimensions import compute_D_S, compute_D_K
from qse.invariants.intersections import max_pairwise_intersection


def verify_povm(
    rhos,
    kernel_bases,
    Ms,
    config: NumericalConfig,
):
    """
    Kiểm tra toàn bộ invariant cuối pipeline.
    """

    n = Ms[0].shape[0]
    k = len(Ms)

    # -----------------------
    # PSD
    # -----------------------

    from qse.linalg.numerical_rank import min_kept_eigenvalue
    min_eig = min(np.linalg.eigvalsh((M + M.conj().T) / 2).min()for M in Ms)

    # -----------------------
    # completeness
    # -----------------------

    completeness_residual = ( np.linalg.norm(sum(Ms) - np.eye(n),ord="fro")/ np.sqrt(n))

    # -----------------------
    # perfect exclusion
    # -----------------------

    exclusion_residual = max(abs(np.trace(rho @ M))for rho, M in zip(rhos, Ms))


    # -----------------------
    # kernel support
    # -----------------------

    support_errors = []

    for QK, M in zip(kernel_bases,Ms):

        P = kernel_projector(QK)

        err = np.linalg.norm( M - P @ M @ P,ord="fro")

        denom = max( 1.0, np.linalg.norm( M,ord="fro"))

        support_errors.append(err / denom)

    support_residual = max(support_errors)

    # -----------------------
    # supports / ranks
    # -----------------------

    support_bases, ranks = (compute_supports(Ms, config))

    active_count = sum(r > 0 for r in ranks)

    sum_rank_squared = sum(r * r for r in ranks)

    # -----------------------
    # Phi / extremality
    # -----------------------

    phi_data = build_phi(
        support_bases
    )

    lam_kept = min_kept_eigenvalue(Ms, config)
    null_info = compute_nullspace(
        phi_data.matrix,
        config,
        lam_kept,
    )

    extreme = (
        null_info.nullity == 0
    )

    # -----------------------
    # dimensions
    # -----------------------

    # đường tính ĐỘC LẬP: dim span của sum L(S_i), không suy từ nullity
    D_S = compute_D_S(support_bases, config, lam_kept)

    D_K = compute_D_K(
        kernel_bases,
        config
    )

    # -----------------------
    # intersections
    # -----------------------

    max_intersection = max_pairwise_intersection(
        support_bases, config, lam_kept
    ).max_dimension

    # -----------------------
    # physical validity
    # -----------------------

    physical_valid = (
        min_eig >= -config.tol_psd(n)
        and
        completeness_residual<= config.tol_completeness(n)
        and
        exclusion_residual<= config.tol_exclusion(n)
        and
        support_residual<= config.tol_support(n)
    )

    # -----------------------
    # theoretical invariants
    # -----------------------

    invariant_valid = (
        extreme

        and

        sum_rank_squared == D_S

        and

        D_S <= D_K

        and

        D_K <= min(
            n * n,
            sum(
                Q.shape[1] ** 2
                for Q in kernel_bases
            )
        )

        and

        active_count
        <= min(k, D_K)

        and

        max_intersection == 0
    )

    return VerificationReport(
        valid=(
            physical_valid
            and invariant_valid
        ),
        physical_valid=physical_valid,
        extreme=extreme,

        completeness_residual=float(
            completeness_residual
        ),
        exclusion_residual=float(
            exclusion_residual
        ),
        support_residual=float(
            support_residual
        ),
        min_eigenvalue=float(
            min_eig
        ),

        ranks=ranks,
        active_count=active_count,

        sum_rank_squared=(
            sum_rank_squared
        ),

        D_S=D_S,
        D_K=D_K,

        nullity_phi=(
            null_info.nullity
        ),

        max_pairwise_intersection=(
            max_intersection
        ),
    )