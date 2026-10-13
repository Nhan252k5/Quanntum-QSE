import cvxpy as cp
import numpy as np

from qse.config import NumericalConfig
from qse.types import FeasibilityResult
from qse.linalg import margins
from qse.linalg.numerical_rank import renormalize


def _completeness_residual(Ms, n) -> float:
    return float(np.linalg.norm(sum(Ms) - np.eye(n), ord="fro") / np.sqrt(n))


def _leak(Ms, kernel_bases, X_vars):
    out = 0.0
    for i, (M, QK) in enumerate(zip(Ms, kernel_bases)):
        if i not in X_vars:
            continue
        P = QK @ QK.conj().T
        out = max(out, float(np.real(np.trace(M - P @ M @ P))))
    return out


def _solve(problem, solver, config):
    acc = config.sdp_accuracy
    if solver == "CLARABEL":
        problem.solve(solver=solver, verbose=config.solver_verbose,
                      tol_feas=acc, tol_gap_abs=acc, tol_gap_rel=acc, max_iter=400)
    elif solver == "SCS":
        problem.solve(solver=solver, verbose=config.solver_verbose,
                      eps_abs=acc, eps_rel=acc, max_iters=20000)
    else:
        problem.solve(solver=solver, verbose=config.solver_verbose)


def solve_feasible_povm(kernel_bases, n, config: NumericalConfig, active_indices=None):
    """
    Bước A dạng ĐỊNH LƯỢNG:

        r* = min || sum_i QK_i X_i QK_i^dagger - I ||_F / sqrt(n),   X_i >= 0

    Bài này luôn khả thi (X = 0), nên solver không phải tự quyết
    "infeasible" — quyết định dựa vào r* so với ngưỡng suy ra từ config:

    Sau đó chuẩn hóa M' = S^{-1/2} M S^{-1/2} (sum M' = I chính xác).
    Vì rho_i M_i = 0, phần rò ra ngoài kernel của M' chỉ cỡ O(r*^2):
        leak_i = Tr((I - P_Ki) M'_i)  >=  Tr(rho_i M'_i)
    nên tiêu chí chấp nhận là leak (đại lượng VẬT LÝ), không phải r*:

        max leak <= tol_exclusion           -> FEASIBLE
        max leak >= 1e4 * tol_exclusion     -> INFEASIBLE (r* là khoảng cách tới SQSE)
        ở giữa                              -> UNKNOWN (AMBIGUOUS)
    """
    k = len(kernel_bases)
    active = set(range(k) if active_indices is None else active_indices)

    X_vars, terms = {}, []
    for i, QK in enumerate(kernel_bases):
        d = QK.shape[1]
        if i not in active or d == 0:
            continue
        X = cp.Variable((d, d), hermitian=True)
        X_vars[i] = X
        terms.append(QK @ X @ QK.conj().T)

    if not X_vars:
        return FeasibilityResult(decision="INFEASIBLE", status="NO_ACTIVE_KERNEL")

    residual_expr = cp.norm(cp.sum(terms) - np.eye(n), "fro") / np.sqrt(n)
    problem = cp.Problem(cp.Minimize(residual_expr), [X >> 0 for X in X_vars.values()])

    installed = set(cp.installed_solvers())
    last_status, last_solver = "NO_SOLVER", None
    best = None
    for solver in config.solver_order:
        if solver not in installed:
            continue
        last_solver = solver
        try:
            _solve(problem, solver, config)
        except Exception:
            continue
        last_status = str(problem.status)
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            continue
        Ms = []
        for i, QK in enumerate(kernel_bases):
            if i not in X_vars or X_vars[i].value is None:
                Ms.append(np.zeros((n, n), dtype=complex))
                continue
            X = X_vars[i].value
            X = (X + X.conj().T) / 2
            # bỏ phần âm do nhiễu solver NGAY trên tọa độ kernel;
            # completeness được khôi phục bởi renormalize phía sau
            w, V = np.linalg.eigh(X)
            X = (V * np.clip(w, 0, None)) @ V.conj().T
            Ms.append(QK @ X @ QK.conj().T)
        r = _completeness_residual(Ms, n)
        try:
            leak = _leak(renormalize(Ms), kernel_bases, X_vars)
        except ValueError:
            leak = np.inf
        if best is None or leak < best[0]:
            best = (leak, Ms, solver, r)
        if leak <= config.tol_exclusion(n):
            break                      # đủ tốt, không cần solver tiếp theo

    if best is None:
        return FeasibilityResult(decision="UNKNOWN", status=last_status, solver=last_solver)

    leak, Ms, solver, r = best
    tol = config.tol_exclusion(n)
    margins.check("sdp_leak", [leak], tol, config, active=sorted(active))

    min_eig = min(np.linalg.eigvalsh((M + M.conj().T) / 2).min() for M in Ms)
    if leak <= tol:
        Ms = renormalize(Ms)
        return FeasibilityResult(decision="FEASIBLE", status=last_status, effects=Ms,
                                 solver=solver, completeness_residual=r,
                                 min_eigenvalue=float(min_eig))
    decision = "INFEASIBLE" if leak >= 1e4 * tol else "UNKNOWN"
    return FeasibilityResult(decision=decision, status=f"RESIDUAL={r:.2e},LEAK={leak:.2e}", effects=Ms,
                             solver=solver, completeness_residual=r,
                             min_eigenvalue=float(min_eig))
