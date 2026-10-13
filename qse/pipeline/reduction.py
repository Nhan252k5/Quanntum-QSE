import dataclasses
import numpy as np
import pandas as pd
from qse.config import NumericalConfig
from qse.types import PipelineResult
from qse.states.validation import validate_states
from qse.states.kernels import compute_kernel_bases
from qse.feasibility.sdp import solve_feasible_povm
from qse.extremality.extremify import extremify
from qse.elimination.dbrge import dbrge
from qse.invariants.verify import verify_povm
from qse.linalg import margins


def run_reduction_pipeline(rhos, config: NumericalConfig):
    """
    rho -> kernel -> SDP (định lượng) -> Extremify -> DBRGE -> Verify

    status: OK | INFEASIBLE | UNKNOWN | EXTREMIFY_FAILED
    ambiguities: các quyết định số học nằm sát ngưỡng (xem qse/linalg/margins.py).
    Nếu ambiguities khác rỗng, kết luận nên được đánh dấu AMBIGUOUS.
    """
    margins.reset()
    validate_states(rhos, config)
    n = rhos[0].shape[0]
    kernel_bases = compute_kernel_bases(rhos, config)

    feas = solve_feasible_povm(kernel_bases, n, config)
    if not feas.feasible:
        return PipelineResult(feasible=False, status=feas.decision,
                              initial_feasibility=feas, ambiguities=margins.entries())

    # Hiệu chỉnh mức nhiễu THỰC ĐO của solver: nếu lời giải Bước A có
    # residual / trị riêng âm lớn hơn sdp_accuracy yêu cầu thì nâng noise lên,
    # để rank_tol (= 1e3 x noise) luôn nằm trên nhiễu thật của solver.
    measured = max(feas.completeness_residual or 0.0, -(feas.min_eigenvalue or 0.0))
    if measured > config.sdp_accuracy:
        config = dataclasses.replace(config, sdp_accuracy=measured)

    first = extremify(feas.effects, config)
    if not first.extreme:
        return PipelineResult(feasible=True, status="EXTREMIFY_FAILED", effects=first.effects,
                              initial_feasibility=feas, initial_extreme=first,
                              ambiguities=margins.entries())

    reduced = dbrge(kernel_bases, first.effects, config)
    final = extremify(reduced, config)
    report = verify_povm(rhos, kernel_bases, final.effects, config)
    result = PipelineResult(feasible=True, status="OK", effects=final.effects,
                          verification=report, initial_feasibility=feas,
                          initial_extreme=first, ambiguities=margins.entries())
    result.effective_config = config
    return result
def export_povm(result, path):
    Ms = result.effects
    n = Ms[0].shape[0]
    rows = [[None] * n]
    for M in Ms:
        rows += [[f"{z.real:.17g}{z.imag:+.17g}j" for z in row] for row in M]
        rows += [[None] * n, [None] * n]
    pd.DataFrame(rows).to_excel(path, sheet_name="Output_M",
                                header=False, index=False)
    np.save(path.replace(".xlsx", ".npy"), np.stack(Ms))   # shape (k, n, n)