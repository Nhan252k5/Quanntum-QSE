import cvxpy as cp
import numpy as np

from qse.config import NumericalConfig
from qse.symmetry.types import (
    SymmetrySpec,
    SymmetryStructure,
    SeedResult,
)


def solve_seed_sdp(
    QK_base,
    spec: SymmetrySpec,
    structure: SymmetryStructure,
    config: NumericalConfig,
):
    """
    Tìm seed N thỏa:

        N >= 0

        range(N) ⊆ K_base

        U_h N U_h^dagger = N
        cho mọi h ∈ H

        sum_i
            U_{g_i} N U_{g_i}^dagger
        = I
    """

    n = QK_base.shape[0]
    d = QK_base.shape[1]

    if d == 0:
        return SeedResult(
            decision="INFEASIBLE",
            status="ZERO_KERNEL",
        )

    # X sống trực tiếp trong kernel space
    X = cp.Variable(
        (d, d),
        hermitian=True
    )

    N = (
        QK_base
        @ X
        @ QK_base.conj().T
    )

    constraints = [
        X >> 0
    ]

    # ---------------------
    # stabilizer invariance
    # ---------------------

    for h in structure.stabilizer:

        Uh = spec.unitaries[h]

        constraints.append(
            Uh
            @ N
            @ Uh.conj().T
            == N
        )

    # ---------------------
    # orbit completeness
    # ---------------------

    total = cp.Constant(
        np.zeros(
            (n, n),
            dtype=complex
        )
    )

    for i in sorted(
        structure.representatives
    ):

        g = (
            structure
            .representatives[i]
        )

        Ug = spec.unitaries[g]

        total = (
            total
            + Ug
            @ N
            @ Ug.conj().T
        )

    constraints.append(
        total == np.eye(n)
    )

    problem = cp.Problem(
        cp.Minimize(0),
        constraints
    )

    installed = set(
        cp.installed_solvers()
    )

    last_status = "NO_SOLVER"
    last_solver = None

    for solver in config.solver_order:

        if solver not in installed:
            continue

        last_solver = solver

        try:

            if solver == "SCS":

                problem.solve(
                    solver=solver,
                    eps=1e-8,
                    max_iters=100000,
                    verbose=(
                        config.solver_verbose
                    ),
                )

            else:

                problem.solve(
                    solver=solver,
                    verbose=(
                        config.solver_verbose
                    ),
                )

        except Exception:
            continue

        last_status = str(
            problem.status
        )

        if problem.status in {
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        }:
            break

        if problem.status == cp.INFEASIBLE:

            return SeedResult(
                decision="INFEASIBLE",
                status=last_status,
                solver=solver,
            )

    if problem.status not in {
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    }:

        return SeedResult(
            decision="UNKNOWN",
            status=last_status,
            solver=last_solver,
        )

    X_value = X.value

    if X_value is None:

        return SeedResult(
            decision="UNKNOWN",
            status="NO_SOLUTION",
            solver=last_solver,
        )

    X_value = (
        X_value
        + X_value.conj().T
    ) / 2

    N_value = (
        QK_base
        @ X_value
        @ QK_base.conj().T
    )

    N_value = (
        N_value
        + N_value.conj().T
    ) / 2

    # ---------------------
    # post verification
    # ---------------------

    min_eig = float(
        np.linalg.eigvalsh(
            N_value
        ).min()
    )

    stab_error = 0.0

    for h in structure.stabilizer:

        Uh = spec.unitaries[h]

        error = np.linalg.norm(
            Uh
            @ N_value
            @ Uh.conj().T
            - N_value,
            ord="fro"
        )

        stab_error = max(
            stab_error,
            float(error)
        )

    total_value = np.zeros(
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

        Ug = spec.unitaries[g]

        total_value += (
            Ug
            @ N_value
            @ Ug.conj().T
        )

    completeness_error = (
        np.linalg.norm(
            total_value
            - np.eye(n),
            ord="fro"
        )
        / np.sqrt(n)
    )

    passes = (
        min_eig
        >= -config.psd_tol

        and

        stab_error
        <= config.support_tol

        and

        completeness_error
        <= config.completeness_tol
    )

    if not passes:

        return SeedResult(
            decision="UNKNOWN",
            status="POSTCHECK_FAILED",
            seed=N_value,
            solver=last_solver,
            psd_residual=min_eig,
            stabilizer_residual=(
                stab_error
            ),
            completeness_residual=(
                completeness_error
            ),
        )

    return SeedResult(
        decision="FEASIBLE",
        status=last_status,
        seed=N_value,
        solver=last_solver,
        psd_residual=min_eig,
        stabilizer_residual=(
            stab_error
        ),
        completeness_residual=(
            completeness_error
        ),
    )