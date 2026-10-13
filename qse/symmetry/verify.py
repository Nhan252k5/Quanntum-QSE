import numpy as np

from qse.extremality.supports import (
    compute_supports,
)

from qse.extremality.phi import (
    build_phi,
)

from qse.extremality.nullspace import (
    compute_nullspace,
)

from qse.invariants.dimensions import (
    compute_D_K,
)

from qse.symmetry.group_action import (
    is_transitive,
    state_covariance_residual,
)

from qse.symmetry.covariant import (
    povm_covariance_residual,
)

from qse.symmetry.commutant import (
    commutant_dimension,
)

from qse.symmetry.types import (
    SymmetryVerificationReport,
)


def verify_covariant_povm(
    rhos,
    kernel_bases,
    Ms,
    spec,
    covariant_extreme_report,
    config,
):
    """
    Kiểm tra nghiệm nhánh symmetry.
    """

    n = Ms[0].shape[0]
    k = len(Ms)

    # ------------------
    # transitivity
    # ------------------

    transitive = is_transitive(
        spec,
        k
    )

    # ------------------
    # state covariance
    # ------------------

    state_cov_err = (
        state_covariance_residual(
            rhos,
            spec
        )
    )

    state_covariant = (
        state_cov_err
        <= config.support_tol
    )

    # ------------------
    # POVM covariance
    # ------------------

    povm_cov_err = (
        povm_covariance_residual(
            Ms,
            spec
        )
    )

    povm_covariant = (
        povm_cov_err
        <= config.support_tol
    )

    # ------------------
    # PSD
    # ------------------

    min_eig = min(
        np.linalg.eigvalsh(
            (M + M.conj().T) / 2
        ).min()
        for M in Ms
    )

    psd_valid = (
        min_eig
        >= -config.psd_tol
    )

    # ------------------
    # completeness
    # ------------------

    completeness_residual = (
        np.linalg.norm(
            sum(Ms)
            - np.eye(n),
            ord="fro"
        )
        / np.sqrt(n)
    )

    completeness_valid = (
        completeness_residual
        <= config.completeness_tol
    )

    # ------------------
    # exclusion
    # ------------------

    exclusion_residual = max(
        abs(
            np.trace(
                rho @ M
            )
        )
        for rho, M
        in zip(rhos, Ms)
    )

    exclusion_valid = (
        exclusion_residual
        <= config.exclusion_tol
    )

    # ------------------
    # ranks
    # ------------------

    support_bases, ranks = (
        compute_supports(
            Ms,
            config
        )
    )

    # ------------------
    # full SQSE extremality
    # ------------------

    phi = build_phi(
        support_bases
    )

    null_info = compute_nullspace(
        phi.matrix,
        config
    )

    full_extreme = (
        null_info.nullity == 0
    )

    # ------------------
    # D_K
    # ------------------

    D_K = compute_D_K(
        kernel_bases,
        config
    )

    # ------------------
    # commutant
    # ------------------

    comm_dim = (
        commutant_dimension(
            spec.unitaries,
            config
        )
    )

    valid = (
        transitive
        and state_covariant
        and povm_covariant
        and psd_valid
        and completeness_valid
        and exclusion_valid
    )

    return SymmetryVerificationReport(
        valid=valid,

        transitive=transitive,
        state_covariant=state_covariant,
        povm_covariant=povm_covariant,

        psd_valid=psd_valid,
        completeness_valid=(
            completeness_valid
        ),
        exclusion_valid=(
            exclusion_valid
        ),

        state_covariance_residual=float(
            state_cov_err
        ),
        povm_covariance_residual=float(
            povm_cov_err
        ),
        completeness_residual=float(
            completeness_residual
        ),
        exclusion_residual=float(
            exclusion_residual
        ),
        min_eigenvalue=float(
            min_eig
        ),

        ranks=ranks,

        extreme_in_covariant_set=(
            covariant_extreme_report
            .extreme_in_covariant_set
        ),

        extreme_in_full_sqse=(
            full_extreme
        ),

        D_K=D_K,

        commutant_dimension=(
            comm_dim
        ),
    )