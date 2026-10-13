import numpy as np

from qse.symmetry.types import (
    SymmetrySpec,
    SymmetryStructure,
)


def build_covariant_povm(
    seed: np.ndarray,
    spec: SymmetrySpec,
    structure: SymmetryStructure,
):
    """
    Sinh:

        M_i
          = U_{g_i}
            N
            U_{g_i}^dagger

    với g_i·base = i.
    """

    k = len(
        structure.representatives
    )

    Ms = [None] * k

    for i, g in (
        structure
        .representatives
        .items()
    ):

        U = spec.unitaries[g]

        M = (
            U
            @ seed
            @ U.conj().T
        )

        Ms[i] = (
            M + M.conj().T
        ) / 2

    return Ms
def povm_covariance_residual(
    Ms,
    spec: SymmetrySpec,
):
    """
    Kiểm tra:

        U_g M_i U_g^dagger
          = M_{g·i}.
    """

    max_error = 0.0

    for g in spec.elements:

        U = spec.unitaries[g]

        for i, Mi in enumerate(Ms):

            j = spec.action(
                g,
                i
            )

            error = np.linalg.norm(
                U
                @ Mi
                @ U.conj().T
                - Ms[j],
                ord="fro"
            )

            max_error = max(
                max_error,
                float(error)
            )

    return max_error