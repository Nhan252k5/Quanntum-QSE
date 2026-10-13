import numpy as np

from qse.config import NumericalConfig
from qse.symmetry.types import (
    SymmetrySpec,
    SymmetryStructure,
)


def check_unitaries(
    spec: SymmetrySpec,
    config: NumericalConfig,
):
    """
    Kiểm tra U_g có unitary không:

        U_g^dagger U_g = I.
    """

    max_error = 0.0

    for g in spec.elements:

        U = spec.unitaries[g]

        n = U.shape[0]

        error = np.linalg.norm(
            U.conj().T @ U
            - np.eye(n),
            ord="fro"
        ) / np.sqrt(n)

        max_error = max(
            max_error,
            float(error)
        )

    return (
        max_error
        <= config.completeness_tol,
        max_error
    )


def orbit(
    spec: SymmetrySpec,
    index: int,
):
    """
    Orbit của index:

        Orb(index)
          = {g·index : g ∈ G}.
    """

    return {
        spec.action(g, index)
        for g in spec.elements
    }


def is_transitive(
    spec: SymmetrySpec,
    k: int,
) -> bool:
    """
    G tác động bắc cầu khi orbit
    của base chứa toàn bộ outcomes.
    """

    orb = orbit(
        spec,
        spec.base_index
    )

    return orb == set(range(k))


def stabilizer(
    spec: SymmetrySpec,
):
    """
    H = Stab(base)

      = {g ∈ G : g·base = base}.
    """

    base = spec.base_index

    return [
        g
        for g in spec.elements
        if spec.action(g, base) == base
    ]


def orbit_representatives(
    spec: SymmetrySpec,
    k: int,
):
    """
    Chọn một g_i cho mỗi outcome i sao cho

        g_i · base = i.

    Đây đóng vai trò đại diện coset gH.
    """

    base = spec.base_index

    reps = {}

    for g in spec.elements:

        i = spec.action(
            g,
            base
        )

        if i not in reps:
            reps[i] = g

    if set(reps.keys()) != set(range(k)):
        raise ValueError(
            "Action không transitive: "
            "không tìm được representative "
            "cho mọi outcome."
        )

    return reps


def build_symmetry_structure(
    spec: SymmetrySpec,
    k: int,
):
    """
    Tạo H và các đại diện g_i.
    """

    if not is_transitive(
        spec,
        k
    ):
        raise ValueError(
            "Nhánh symmetry hiện tại "
            "yêu cầu action transitive."
        )

    H = stabilizer(spec)

    reps = orbit_representatives(
        spec,
        k
    )

    return SymmetryStructure(
        stabilizer=H,
        representatives=reps,
    )
def state_covariance_residual(
    rhos,
    spec: SymmetrySpec,
):
    """
    Kiểm tra:

        rho_{g·i}
          = U_g rho_i U_g^dagger.

    Trả về residual lớn nhất.
    """

    max_error = 0.0

    for g in spec.elements:

        U = spec.unitaries[g]

        for i, rho_i in enumerate(rhos):

            j = spec.action(
                g,
                i
            )

            lhs = rhos[j]

            rhs = (
                U
                @ rho_i
                @ U.conj().T
            )

            error = np.linalg.norm(
                lhs - rhs,
                ord="fro"
            )

            max_error = max(
                max_error,
                float(error)
            )

    return max_error