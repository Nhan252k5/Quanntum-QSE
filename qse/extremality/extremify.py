import numpy as np

from qse.config import NumericalConfig
from qse.types import ExtremalityResult

from qse.extremality.supports import (
    compute_supports,
)
from qse.extremality.phi import (
    build_phi,
    decode_direction,
)
from qse.extremality.nullspace import (
    compute_nullspace,
    choose_null_direction,
)
from qse.linalg.numerical_rank import min_kept_eigenvalue, renormalize
from qse.extremality.boundary import (
    global_step_interval,
    choose_boundary_step,
)


def extremify(
    Ms: list[np.ndarray],
    config: NumericalConfig,
) -> ExtremalityResult:
    """
    Đưa một POVM khả thi về một điểm cực biên.

    Loop:
        support
        -> Phi
        -> ker(Phi)
        -> Delta
        -> PSD boundary
        -> rank giảm
        -> repeat
    """

    Ms = [(M + M.conj().T) / 2 for M in Ms]

    history = []

    for iteration in range(config.max_extremify_iter):
        support_bases, ranks = (compute_supports(Ms,config))

        phi_data = build_phi(support_bases)

        null_info = compute_nullspace(phi_data.matrix, config,
                                      min_kept_eigenvalue(Ms, config))

        history.append({
            "iteration": iteration,
            "ranks": ranks.copy(),
            "nullity": null_info.nullity,
        })

        # Điều kiện cực biên
        if null_info.nullity == 0:

            return ExtremalityResult(
                effects=Ms,
                extreme=True,
                ranks=ranks,
                nullity=0,
                iterations=iteration,
                history=history,
            )

        # chọn z thuộc ker(Phi)
        z = choose_null_direction(null_info)

        # z -> Delta_i
        Deltas = decode_direction(z,phi_data,support_bases)

        # check sum Delta_i ~ 0
        delta_sum = sum(Deltas)

        null_residual = np.linalg.norm(delta_sum,ord="fro")

        lower, upper = (
            global_step_interval(
                Ms,
                Deltas,
                support_bases,
                config,
            )
        )

        t = choose_boundary_step(
            lower,
            upper,
            config
        )

        # move to PSD boundary
        new_Ms = []

        for M, Delta in zip(
            Ms,
            Deltas
        ):

            M_new = (
                M + t * Delta
            )

            M_new = (
                M_new
                + M_new.conj().T
            ) / 2

            new_Ms.append(
                M_new
            )

        # Hướng "null số" chỉ có sum Delta ~ sigma, nên bước t làm lệch
        # completeness cỡ |t| * sigma. Chuẩn hóa lại bằng S^{-1/2} (giữ hạng).
        n = Ms[0].shape[0]
        if abs(t) * null_residual > config.tol_completeness(n):
            history[-1]["completeness_drift"] = float(abs(t) * null_residual)
        Ms = renormalize(new_Ms)  # sum Delta ~ 0 nên S ~ I, luôn an toàn

    support_bases, ranks = (
        compute_supports(
            Ms,
            config
        )
    )

    phi_data = build_phi(
        support_bases
    )

    null_info = compute_nullspace(
        phi_data.matrix,
        config,
        min_kept_eigenvalue(Ms, config),
    )

    return ExtremalityResult(
        effects=Ms,
        extreme=False,
        ranks=ranks,
        nullity=null_info.nullity,
        iterations=config.max_extremify_iter,
        history=history,
    )