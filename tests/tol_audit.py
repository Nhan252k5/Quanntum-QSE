"""
Kiểm toán ngưỡng: với MỖI quyết định dựa trên tol, ghi lại "biên an toàn"
    margin_above = log10(giá trị nhỏ nhất được GIỮ / tol)
    margin_below = log10(tol / giá trị lớn nhất bị COI LÀ 0)
Biên < 2 (tức < 100 lần) nghĩa là quyết định dễ lật khi đổi tol hoặc nhiễu.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from collections import defaultdict
import qse.states.kernels as KER
import qse.linalg.numerical_rank as NR
import qse.extremality.nullspace as NS
import qse.feasibility.sdp as SDP

LOG = defaultdict(list)

def _margins(vals, tol, keep_above=True):
    vals = np.abs(np.asarray(vals))
    above = vals[vals > tol]; below = vals[vals <= tol]
    up = np.log10(above.min() / tol) if len(above) else np.inf
    dn = np.log10(tol / max(below.max(), 1e-300)) if len(below) else np.inf
    return up, dn

def install(config):
    orig_k = KER.kernel_basis
    def kb(rho, cfg):
        lam = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
        tol = max(cfg.kernel_atol, cfg.kernel_rtol * max(1.0, np.abs(lam).max()))
        LOG["kernel"].append(_margins(lam, tol)); return orig_k(rho, cfg)
    KER.kernel_basis = kb

    orig_s = NR.psd_support
    def ps(M, cfg):
        lam = np.linalg.eigvalsh((M + M.conj().T) / 2)
        tol = max(cfg.rank_atol, cfg.rank_rtol * max(1.0, np.abs(lam).max()))
        LOG["rank"].append(_margins(lam, tol)); return orig_s(M, cfg)
    NR.psd_support = ps
    import qse.extremality.supports as SU; SU.psd_support = ps

    orig_n = NS.compute_nullspace
    def cn(Phi, cfg):
        info = orig_n(Phi, cfg)
        if len(info.singular_values):
            LOG["null"].append(_margins(info.singular_values, info.tolerance))
        return info
    NS.compute_nullspace = cn
    import qse.extremality.extremify as EX, qse.invariants.verify as VE
    EX.compute_nullspace = cn; VE.compute_nullspace = cn

    orig_f = SDP.solve_feasible_povm
    def sf(*a, **k):
        r = orig_f(*a, **k)
        if r.completeness_residual is not None:
            LOG["sdp_resid"].append(r.completeness_residual)
            LOG["sdp_mineig"].append(r.min_eigenvalue)
        return r
    SDP.solve_feasible_povm = sf
    import qse.pipeline.reduction as RD, qse.elimination.dbrge as DB
    RD.solve_feasible_povm = sf; DB.solve_feasible_povm = sf
    RD.compute_kernel_bases.__globals__["kernel_basis"] = kb
