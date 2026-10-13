import numpy as np

from qse.linalg.hermitian import (
    hermitian_basis,
    vec_hermitian,
)

from qse.linalg.numerical_rank import (
    numerical_rank,
)


def commutant_dimension(
    unitaries,
    config,
):
    """
    Tính dimension thực của:

        Herm_G(n)
          = {A = A^dagger:
             U_g A U_g^dagger = A
             ∀g∈G}

    Không cần decomposition irreducible.
    """

    first_U = next(
        iter(unitaries.values())
    )

    n = first_U.shape[0]

    basis = hermitian_basis(n)

    constraints = []

    for U in unitaries.values():

        columns = []

        for E in basis:

            diff = (
                U
                @ E
                @ U.conj().T
                - E
            )

            columns.append(
                vec_hermitian(diff)
            )

        constraints.append(
            np.column_stack(
                columns
            )
        )

    C = np.vstack(
        constraints
    )

    rank_C = numerical_rank(
        C,
        config.null_rtol,
        config.null_atol,
    )

    return (
        n * n
        - rank_C
    )