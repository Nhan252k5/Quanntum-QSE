from dataclasses import dataclass

import numpy as np

from qse.config import NumericalConfig
from qse.linalg import margins


@dataclass
class NullspaceInfo:
    rank: int 
    nullity: int

    singular_values: np.ndarray

    tolerance: float

    basis: np.ndarray


def compute_nullspace(
    Phi: np.ndarray,
    config: NumericalConfig,
    lam_min_kept: float = 1.0,
) -> NullspaceInfo:
    """
    Tìm ker(Phi) bằng SVD.

        Phi = U Sigma V^T

    Các cột V tương ứng singular value 0
    tạo basis của ker(Phi).
    """

    m, d = Phi.shape

    if d == 0:
        return NullspaceInfo(
            rank=0,
            nullity=0,
            singular_values=np.array([]),
            tolerance=0.0,
            basis=np.zeros((0, 0)),
        )

    U, s, Vh = np.linalg.svd(Phi,full_matrices=True)

    n = int(round(np.sqrt(m)))
    if len(s) == 0:
        tol = config.tol_null(n, 1.0, lam_min_kept)
    else:
        # Davis–Kahan: basis support sai lệch ~ noise / lambda_min_kept,
        # nên singular value của Phi nhỏ hơn mức đó không phân biệt được với 0.
        tol = config.tol_null(n, s[0], lam_min_kept)
    # singular value bằng 0 "ẩn" khi d > m: chỉ kiểm tra các giá trị thật
    margins.check("null", s, tol, config, shape=Phi.shape)

    rank = int( np.sum(s > tol))    
    nullity = d - rank #dim(ker(Phi)) = d - rank(Phi), d số cột của ma trân Phi


    if nullity == 0:
        basis = np.zeros( (d, 0) )
    else:
        basis = (Vh[rank:, :].T.copy()) # do np trả về theo hàng nên phai chuyển sang cột, basis có dim d x nullity

    return NullspaceInfo(
        rank=rank,
        nullity=nullity,
        singular_values=s,
        tolerance=tol,
        basis=basis,
    )


def choose_null_direction(
    info: NullspaceInfo,
) -> np.ndarray:
    """
    Chọn một hướng khác 0 trong ker(Phi).

    Hiện tại lấy vector basis đầu tiên.

    Sau này có thể thay bằng chiến lược
    chọn hướng tối ưu hơn.
    """

    if info.nullity == 0:
        raise ValueError(
            "ker(Phi) = {0}."
        )

    z = info.basis[:, 0]

    norm = np.linalg.norm(z)

    if norm == 0:
        raise RuntimeError(
            "Null direction bằng 0."
        )

    return z / norm #giúp tính toán ổn định và dễ so sánh