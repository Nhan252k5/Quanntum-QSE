from qse.states.validation import (
    validate_states,
)

from qse.states.kernels import (
    compute_kernel_bases,
)

from qse.symmetry.group_action import (
    check_unitaries,
    build_symmetry_structure,
    state_covariance_residual,
)

from qse.symmetry.seed import (
    solve_seed_sdp,
)

from qse.symmetry.covariant import (
    build_covariant_povm,
)

from qse.symmetry.covariant_extremality import (
    covariant_extremality,
)

from qse.symmetry.verify import (
    verify_covariant_povm,
)

from qse.symmetry.types import (
    SymmetryPipelineResult,
)


def run_symmetry_pipeline(
    rhos,
    spec,
    config,
):
    """
    Nhánh symmetry:

        states
        -> validate symmetry
        -> kernels
        -> stabilizer H
        -> coset representatives
        -> seed SDP
        -> orbit POVM
        -> covariant extremality
        -> verification
    """

    # STEP 0
    validate_states(
        rhos,
        config
    )

    k = len(rhos)

    # STEP 1
    unitary_ok, unitary_error = (
        check_unitaries(
            spec,
            config
        )
    )

    if not unitary_ok:
        raise ValueError(
            f"U_g không unitary. "
            f"max residual="
            f"{unitary_error:.3e}"
        )

    # STEP 2
    structure = (
        build_symmetry_structure(
            spec,
            k
        )
    )

    # STEP 3
    state_cov_error = (
        state_covariance_residual(
            rhos,
            spec
        )
    )

    if (
        state_cov_error
        > config.support_tol
    ):
        raise ValueError(
            "States không thỏa covariance. "
            f"residual="
            f"{state_cov_error:.3e}"
        )

    # STEP 4
    kernel_bases = (
        compute_kernel_bases(
            rhos,
            config
        )
    )

    QK_base = kernel_bases[
        spec.base_index
    ]

    # STEP 5
    seed_result = solve_seed_sdp(
        QK_base,
        spec,
        structure,
        config
    )

    if not seed_result.feasible:

        return SymmetryPipelineResult(
            feasible=False,
            seed_result=seed_result,
        )

    seed = seed_result.seed

    # STEP 6
    Ms = build_covariant_povm(
        seed,
        spec,
        structure
    )

    # STEP 7
    cov_ext = (
        covariant_extremality(
            seed,
            spec,
            structure,
            config
        )
    )

    # STEP 8
    report = verify_covariant_povm(
        rhos,
        kernel_bases,
        Ms,
        spec,
        cov_ext,
        config
    )

    return SymmetryPipelineResult(
        feasible=True,
        seed=seed,
        effects=Ms,
        seed_result=seed_result,
        covariant_extremality=(
            cov_ext
        ),
        verification=report,
    )