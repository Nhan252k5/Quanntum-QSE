import numpy as np

from qse.config import NumericalConfig

from qse.linalg.hermitian import (
    hermitian_basis,
    vec_hermitian,
)

from qse.linalg.numerical_rank import (
    psd_support,
)

from qse.symmetry.types import (
    SymmetrySpec,
    SymmetryStructure,
    CovariantExtremalityReport,
)


def _nullspace(
    A,
    config,
):
    """
    Nullspace generic bằng SVD.
    """

    m, d = A.shape

    if d == 0:
        return np.zeros((0, 0))

    U, s, Vh = np.linalg.svd(
        A,
        full_matrices=True
    )

    if len(s) == 0:
        rank = 0

    else:

        tol = max(
            config.null_atol,
            config.null_rtol
            * s[0]
        )

        rank = int(
            np.sum(s > tol)
        )

    return (
        Vh[rank:, :]
        .conj()
        .T
    )
def stabilizer_invariant_support_basis(
    seed,
    spec,
    structure,
    config,
):
    """
    Tìm basis của:

        Herm_H(S)

    = {Delta = Delta^dagger:
       range(Delta) ⊆ S,
       U_h Delta U_h^dagger = Delta
       ∀h∈H}

    với S = range(seed).
    """

    QS, r, _ = psd_support(
        seed,
        config
    )

    if r == 0:
        return []

    local = hermitian_basis(r)

    embedded = [
        QS
        @ E
        @ QS.conj().T
        for E in local
    ]

    # Nếu H chỉ có identity,
    # mọi perturbation đều invariant.
    if len(structure.stabilizer) == 0:
        return embedded

    constraint_blocks = []

    for h in structure.stabilizer:

        U = spec.unitaries[h]

        columns = []

        for A in embedded:

            diff = (
                U
                @ A
                @ U.conj().T
                - A
            )

            columns.append(
                vec_hermitian(diff)
            )

        constraint_blocks.append(
            np.column_stack(
                columns
            )
        )

    C = np.vstack(
        constraint_blocks
    )

    Z = _nullspace(
        C,
        config
    )

    invariant_basis = []

    for j in range(
        Z.shape[1]
    ):

        coeffs = Z[:, j]

        B = np.zeros_like(
            seed,
            dtype=complex
        )

        for c, A in zip(
            coeffs,
            embedded
        ):
            B += c * A

        B = (
            B + B.conj().T
        ) / 2

        invariant_basis.append(B)

    return invariant_basis
def covariant_extremality(
    seed,
    spec,
    structure,
    config,
):
    """
    Kiểm tra extreme trong S_cov.

    Map:

       T(Delta)
         = sum_{gH}
             U_g Delta U_g^dagger.

    Domain:
       Herm_H(S_seed)

    Seed extreme trong S_cov
    iff ker(T) = {0}.
    """

    invariant_basis = (
        stabilizer_invariant_support_basis(
            seed,
            spec,
            structure,
            config,
        )
    )

    d = len(
        invariant_basis
    )

    n = seed.shape[0]

    if d == 0:

        return CovariantExtremalityReport(
            extreme_in_covariant_set=True,
            domain_dimension=0,
            map_rank=0,
            nullity=0,
            invariant_support_dimension=0,
        )

    columns = []

    for Delta in invariant_basis:

        total = np.zeros(
            (n, n),
            dtype=complex
        )

        for i in sorted(
            structure.representatives
        ):

            g = (
                structure
                .representatives[i]
            )

            U = spec.unitaries[g]

            total += (
                U
                @ Delta
                @ U.conj().T
            )

        columns.append(
            vec_hermitian(total)
        )

    T_matrix = np.column_stack(
        columns
    )

    U, s, Vh = np.linalg.svd(
        T_matrix,
        full_matrices=True
    )

    if len(s) == 0:

        rank = 0

    else:

        tol = max(
            config.null_atol,
            config.null_rtol
            * s[0]
        )

        rank = int(
            np.sum(s > tol)
        )

    nullity = (
        T_matrix.shape[1]
        - rank
    )

    return CovariantExtremalityReport(
        extreme_in_covariant_set=(
            nullity == 0
        ),
        domain_dimension=(
            T_matrix.shape[1]
        ),
        map_rank=rank,
        nullity=nullity,
        invariant_support_dimension=d,
    )