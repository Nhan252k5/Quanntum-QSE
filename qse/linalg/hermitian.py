import numpy as np


def hermitian_basis(
    d: int
) -> list[np.ndarray]:
    """
    Basis trực chuẩn Hilbert-Schmidt
    của Herm(d).

    dim_R Herm(d) = d^2.
    """

    basis = []

    # diagonal
    for i in range(d):
        E = np.zeros(
            (d, d),
            dtype=complex
        )

        E[i, i] = 1.0

        basis.append(E)

    # real off-diagonal
    for i in range(d):
        for j in range(i + 1, d):

            E = np.zeros(
                (d, d),
                dtype=complex
            )

            E[i, j] = 1 / np.sqrt(2)
            E[j, i] = 1 / np.sqrt(2)

            basis.append(E)

    # imaginary off-diagonal
    for i in range(d):
        for j in range(i + 1, d):

            E = np.zeros(
                (d, d),
                dtype=complex
            )

            E[i, j] = -1j / np.sqrt(2)
            E[j, i] = 1j / np.sqrt(2)

            basis.append(E)

    return basis


def vec_hermitian(
    A: np.ndarray
) -> np.ndarray:
    """
    Vector hóa Hermitian matrix
    thành vector thực dimension n^2.

    Dùng:
        diagonal
        sqrt(2) Re(A_ij)
        sqrt(2) Im(A_ij)
    """

    A = (
        A + A.conj().T
    ) / 2

    n = A.shape[0]

    out = []

    # diagonal
    for i in range(n):
        out.append(
            float(A[i, i].real)
        )

    # real off-diagonal
    for i in range(n):
        for j in range(i + 1, n):
            out.append(
                np.sqrt(2)
                * A[i, j].real
            )

    # imaginary off-diagonal
    for i in range(n):
        for j in range(i + 1, n):
            out.append(
                np.sqrt(2)
                * A[i, j].imag
            )

    return np.asarray(
        out,
        dtype=float
    )