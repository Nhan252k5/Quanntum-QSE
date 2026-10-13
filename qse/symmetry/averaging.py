import numpy as np

from qse.symmetry.types import (
    SymmetrySpec,
)


def twirl_povm(
    Ms,
    spec: SymmetrySpec,
):
    """
    Group-average một POVM bất kỳ.

    Thay vì dùng g^{-1}, ta loop:

        j -> i = g·j

    và cộng

        U_g M_j U_g^dagger

    vào block i.

    Kết quả là POVM hiệp biến.
    """

    k = len(Ms)
    n = Ms[0].shape[0]

    averaged = [
        np.zeros(
            (n, n),
            dtype=complex
        )
        for _ in range(k)
    ]

    G_size = len(
        spec.elements
    )

    for g in spec.elements:

        U = spec.unitaries[g]

        for j, Mj in enumerate(Ms):

            i = spec.action(
                g,
                j
            )

            averaged[i] += (
                U
                @ Mj
                @ U.conj().T
            ) / G_size

    # numerical Hermitian cleanup
    averaged = [
        (
            M + M.conj().T
        ) / 2
        for M in averaged
    ]

    return averaged