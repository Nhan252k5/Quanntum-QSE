from dataclasses import dataclass

import numpy as np

from qse.linalg.hermitian import hermitian_basis, vec_hermitian



@dataclass
class PhiData: 
    matrix: np.ndarray #Ma trận biểu diễn ánh xạ Phi. Cột là bậc tự do của các perturbation Delta_i, hàng là bậc tự do của Herm(n).

    block_slices: list[slice] # Danh sách các slice, mỗi slice chỉ ra vị trí của các cột trong ma trận Phi tương ứng với perturbation Delta_i. 

    local_bases: list[list[np.ndarray]] 


def build_phi(support_bases: list[np.ndarray]) -> PhiData:
    """
    Xây ma trận biểu diễn ánh xạ

        Phi(H_1,...,H_k)
          = sum_i QS_i H_i QS_i^dagger.

    Domain có dimension:

        sum_i r_i^2

    Codomain Herm(n) có dimension n^2.
    """

    if len(support_bases) == 0:
        raise ValueError(
            "support_bases rỗng."
        )

    n = support_bases[0].shape[0]

    columns = []

    block_slices = []

    local_bases_all = []

    cursor = 0

    for QS in support_bases:

        r = QS.shape[1]

        local_basis = (hermitian_basis(r))

        local_bases_all.append(local_basis)

        start = cursor

        for E in local_basis:
            A = ( QS @ E @ QS.conj().T)
            columns.append(vec_hermitian(A))
            cursor += 1

        block_slices.append(slice(start,cursor))

    if len(columns) == 0:
        Phi = np.zeros((n * n, 0),dtype=float)

    else:

        Phi = np.column_stack(columns)

    return PhiData(
        matrix=Phi,
        block_slices=block_slices,
        local_bases=local_bases_all,
    )


def decode_direction(
    z: np.ndarray,
    phi_data: PhiData,
    support_bases: list[np.ndarray],
):
    """
    Chuyển vector z trong ker(Phi)
    thành các perturbation Delta_i.
    """

    deltas = []

    for QS, slc, local_basis in zip(
        support_bases,
        phi_data.block_slices,
        phi_data.local_bases,
    ):

        coeffs = z[slc]

        r = QS.shape[1]

        H = np.zeros((r, r),dtype=complex)
        for c, E in zip(coeffs, local_basis):
            H += c * E

        Delta = (QS @ H @ QS.conj().T)

        Delta = (Delta+ Delta.conj().T) / 2

        deltas.append(Delta)

    return deltas