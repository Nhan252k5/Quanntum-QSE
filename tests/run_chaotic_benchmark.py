"""
Chạy pipeline trên bộ dữ liệu hỗn loạn và so với ground truth.
    python -m tests.run_chaotic_benchmark
"""
import time, warnings, traceback
import numpy as np
warnings.filterwarnings("ignore")

from qse.config import DEFAULT_CONFIG as C
from qse.pipeline.reduction import run_reduction_pipeline
from qse.states.kernels import compute_kernel_bases
from qse.feasibility.sdp import solve_feasible_povm
from tests.chaotic_data import build_suite, min_subset_size


def exhaustive_truth(rhos, n):
    K = compute_kernel_bases(rhos, C)
    full = solve_feasible_povm(K, n, C)
    if not full.feasible:
        return dict(feasible=False, s_min=None)
    s, A = min_subset_size(len(rhos),
                           lambda A: solve_feasible_povm(K, n, C, active_indices=A).feasible)
    return dict(feasible=True, s_min=s, witness=A)


def run_case(case):
    t0 = time.time()
    try:
        res = run_reduction_pipeline(case["rhos"], C)
    except Exception as e:
        return dict(status="CRASH", error=f"{type(e).__name__}: {e}", time=time.time() - t0)
    out = dict(status="OK", time=time.time() - t0, feasible=res.feasible,
               amb=len(getattr(res, "ambiguities", []) or []),
               stepA=f"{res.initial_feasibility.decision}/{res.initial_feasibility.status}")
    v = res.verification
    if res.feasible and v is None:
        out["status"] = "NO_REPORT(extremify failed)"
    if v is not None:
        mpi = v.max_pairwise_intersection
        out.update(s=v.active_count, ranks=v.ranks, physical=bool(v.physical_valid),
                   extreme=bool(v.extreme), DS=v.D_S, DK=v.D_K,
                   reported_valid=bool(v.valid),
                   max_inter=getattr(mpi, "max_dimension", mpi))
    return out


def main():
    rows = []
    for case in build_suite():
        truth = dict(case["truth"])
        if truth.get("needs_exhaustive"):
            truth.update(exhaustive_truth(case["rhos"], case["n"]))
            truth["source"] = "exhaustive SDP (cùng tol)"
        out = run_case(case)

        if out["status"] == "CRASH":
            verdict = "CRASH"
        elif truth.get("below_tol") and out.get("feasible"):
            verdict = "THEO TOL (nhiễu < kernel_tol bị coi là 0)"
        elif truth["feasible"] is not None and out.get("feasible") != truth["feasible"]:
            verdict = f"SAI feasibility (Bước A trả {out.get('stepA')})"
        elif out.get("feasible") and truth.get("s_min") is not None and "s" in out:
            verdict = "TỐI ƯU" if out["s"] == truth["s_min"] else f"DƯ ({out['s']} > {truth['s_min']})"
            if not out["physical"]:
                verdict += " / KHÔNG HỢP LỆ VẬT LÝ"
        else:
            verdict = "OK" if out.get("feasible") is False else out["status"]
        rows.append((case["name"], truth, out, verdict))
        print(f"{case['name']:<34} truth(feas={truth['feasible']}, s_min={truth.get('s_min')}) "
              f"-> {verdict}  | s={out.get('s')} ranks={out.get('ranks')} "
              f"amb={out.get('amb')} t={out['time']:.1f}s {out.get('error','')}")
    return rows


if __name__ == "__main__":
    main()
