"""
Sinh dữ liệu "hỗn loạn" cho SQSE kèm ĐÁP ÁN ĐÚNG (ground truth).

Mỗi generator trả về dict:
    name, n, rhos, truth = {"feasible": bool|None, "s_min": int|None, ...}
"""
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.stats import unitary_group


# ------------------------------------------------------------------ helpers
def haar_unitary(n, rng):
    return unitary_group.rvs(n, random_state=rng)


def state_with_kernel(QK, rng, spectrum="random", rank_eps=None):
    """rho có ker(rho) = span(QK), phổ ngẫu nhiên trên phần bù."""
    n = QK.shape[0]
    P = np.eye(n) - QK @ QK.conj().T
    w, V = np.linalg.eigh(P)
    QS = V[:, w > 0.5]                       # basis của K^perp
    r = QS.shape[1]
    lam = rng.random(r) + 0.05 if spectrum == "random" else np.ones(r)
    if rank_eps is not None:                 # phổ suy biến mờ
        lam[-1] = rank_eps
    lam = lam / lam.sum()
    W = haar_unitary(r, rng) if r > 1 else np.ones((1, 1))
    rho = QS @ W @ np.diag(lam) @ W.conj().T @ QS.conj().T
    return (rho + rho.conj().T) / 2


def min_subset_size(k, is_feasible):
    """s_min chính xác: tập con nhỏ nhất khả thi (vét cạn, k nhỏ)."""
    for t in range(2, k + 1):
        for A in itertools.combinations(range(k), t):
            if is_feasible(A):
                return t, A
    return None, None


# ------------------------------------------- 1. Qubit thuần Haar (LP chính xác)
def bloch(psi):
    rho = np.outer(psi, psi.conj())
    return np.real([2 * rho[0, 1].real, 2 * rho[1, 0].imag, rho[0, 0] - rho[1, 1]])


def zero_in_hull(points):
    """0 ∈ conv(points)?  LP: sum c_i p_i = 0, sum c_i = 1, c >= 0."""
    P = np.asarray(points).T
    A_eq = np.vstack([P, np.ones(P.shape[1])])
    b_eq = np.r_[np.zeros(P.shape[0]), 1.0]
    res = linprog(np.zeros(P.shape[1]), A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * P.shape[1], method="highs")
    return res.status == 0


def gen_random_qubit_pure(k, rng, force_feasible=True):
    """
    Với qubit thuần: K_i = psi_i^perp, Bloch(K_i) = -b_i.
    SQSE khả thi  <=>  0 ∈ conv{b_i}.   s_min = tập con nhỏ nhất có 0 trong bao lồi.
    """
    while True:
        psis = [haar_unitary(2, rng)[:, 0] for _ in range(k)]
        bs = [bloch(p) for p in psis]
        feas = zero_in_hull(bs)
        if feas or not force_feasible:
            break
    s_min, A = (min_subset_size(k, lambda A: zero_in_hull([bs[i] for i in A]))
                if feas else (None, None))
    rhos = [np.outer(p, p.conj()) for p in psis]
    return dict(name=f"qubit_pure_haar_k{k}", n=2, rhos=rhos,
                truth=dict(feasible=feas, s_min=s_min, witness=A))


# ---------------------------------- 2. Set-cover bị xoay bởi unitary Haar
def set_cover_min(sets, universe):
    k = len(sets)
    for t in range(1, k + 1):
        for A in itertools.combinations(range(k), t):
            if set().union(*(sets[i] for i in A)) >= universe:
                return t, A
    return None, None


def gen_rotated_set_cover(sets, n, rng, name="rotated_set_cover"):
    """
    K_i = U · span{e_j : j ∈ T_i}.  Vì các K_i là ảnh xoay của các
    không gian tọa độ, SQSE khả thi <=> {T_i} phủ {0..n-1},
    và s_min = số tập nhỏ nhất phủ được (set cover).
    """
    U = haar_unitary(n, rng)
    rhos = []
    for T in sets:
        QK = U[:, sorted(T)]
        rhos.append(state_with_kernel(QK, rng))
    s_min, A = set_cover_min([set(T) for T in sets], set(range(n)))
    return dict(name=name, n=n, rhos=rhos,
                truth=dict(feasible=s_min is not None, s_min=s_min, witness=A))


def gen_user_example(rng):
    """Ví dụ trong hội thoại: a..k -> 0..8, n = 9, đáp án s_min = 4."""
    idx = {c: i for i, c in enumerate("abcdefghk")}
    sets = [{idx[c] for c in s} for s in ["abc", "acde", "ef", "ghk", "dge"]]
    return gen_rotated_set_cover(sets, 9, rng, name="user_example_n9")


def gen_random_set_cover(n, k, rng, p=0.35):
    while True:
        sets = [set(np.flatnonzero(rng.random(n) < p)) or {int(rng.integers(n))}
                for _ in range(k)]
        if all(len(s) < n for s in sets) and set().union(*sets) == set(range(n)):
            break
    return gen_rotated_set_cover(sets, n, rng, name=f"random_set_cover_n{n}_k{k}")


# ------------------------------- 3. Kernel tổng quát ngẫu nhiên (truth = SDP vét cạn)
def gen_random_kernels(n, dims, rng):
    rhos = []
    for d in dims:
        QK = haar_unitary(n, rng)[:, :d]
        rhos.append(state_with_kernel(QK, rng))
    return dict(name=f"random_kernels_n{n}_d{'-'.join(map(str, dims))}", n=n,
                rhos=rhos, truth=dict(feasible=None, s_min=None, needs_exhaustive=True))


# ------------------------------------- 4. Nhiễu full-rank (phải INFEASIBLE)
def add_depolarizing(case, eps):
    n = case["n"]
    rhos = [(1 - eps) * r + eps * np.eye(n) / n for r in case["rhos"]]
    return dict(name=f"{case['name']}_depol{eps:.0e}", n=n, rhos=rhos,
                truth=dict(feasible=False, s_min=None,
                   # Về toán: rho full rank => K_i = {0} => SQSE rỗng.
                   # Về số: nếu eps/n < kernel_tol thì code (đúng theo thiết kế)
                   # coi phần nhiễu là 0 và trả về nghiệm của bài gốc.
                   below_tol=eps / n < 1e-10))


# -------------------------------- 5. Phổ suy biến mờ (nhạy với kernel_tol)
def gen_blurred_spectrum(sets, n, rng, tiny):
    U = haar_unitary(n, rng)
    rhos = []
    for T in sets:
        rhos.append(state_with_kernel(U[:, sorted(T)], rng, rank_eps=tiny))
    return dict(name=f"blurred_spectrum_{tiny:.0e}", n=n, rhos=rhos,
                # Về toán: ker = span(U e_T) chính xác, các T phủ {0..5} => khả thi, s_min = 3.
        # tiny gần kernel_tol => eigh trộn kernel với hướng tiny (sai số ~ 1e-16/tiny).
        truth=dict(feasible=True, s_min=3))


# ------------------------------- 6. Kernel gần giao nhau (tranh chấp gần suy biến)
def gen_near_intersecting(delta, rng, n=4):
    """
    K1 = span(e0, e1), K2 = span(e2, e3 + ... ) nghiêng góc delta về phía e1.
    delta = 0  -> K1 ∩ K2 ≠ {0}; delta nhỏ -> rất gần suy biến.
    K3 bổ sung để bài toán khả thi.
    """
    U = haar_unitary(n, rng)
    e = np.eye(n)
    K1 = e[:, [0, 1]]
    v = np.cos(delta) * e[:, 1] + np.sin(delta) * e[:, 2]
    K2 = np.linalg.qr(np.column_stack([v, e[:, 3]]))[0]
    K3 = e[:, [2, 3]]
    rhos = [state_with_kernel(U @ K, rng) for K in (K1, K2, K3)]
    return dict(name=f"near_intersect_delta{delta:.0e}", n=n, rhos=rhos,
                truth=dict(feasible=True, s_min=2, witness=(0, 2)))


# ------------------------------------------------- 7. Họ có đáp án giải tích
def gen_trine():
    psis = [np.array([np.cos(np.pi * j / 3), np.sin(np.pi * j / 3)]) for j in range(3)]
    return dict(name="trine", n=2, rhos=[np.outer(p, p) for p in psis],
                truth=dict(feasible=True, s_min=3))


def gen_infeasible_hemisphere(k, rng):
    """Qubit thuần với mọi Bloch vector ở bán cầu z > 0.2 -> INFEASIBLE."""
    rhos = []
    while len(rhos) < k:
        psi = haar_unitary(2, rng)[:, 0]
        if bloch(psi)[2] > 0.2:
            rhos.append(np.outer(psi, psi.conj()))
    return dict(name=f"hemisphere_k{k}", n=2, rhos=rhos,
                truth=dict(feasible=False, s_min=None))


def build_suite(seed=2026):
    rng = np.random.default_rng(seed)
    base = gen_random_set_cover(6, 6, rng)
    suite = [gen_user_example(rng), gen_trine(), gen_infeasible_hemisphere(5, rng)]
    suite += [gen_random_qubit_pure(k, rng) for k in (4, 6, 8, 8, 10)]
    suite += [gen_random_set_cover(n, k, rng) for n, k in ((5, 6), (6, 7), (7, 8), (8, 8))]
    suite.append(base)
    suite += [add_depolarizing(base, e) for e in (1e-3, 1e-8, 1e-13)]
    sets_b = [{0, 1, 2}, {2, 3, 4}, {4, 5, 0}, {1, 3, 5}]
    suite += [gen_blurred_spectrum(sets_b, 6, rng, t) for t in (1e-6, 1e-9, 1e-11)]
    suite += [gen_near_intersecting(d, rng) for d in (1e-1, 1e-4, 1e-7, 1e-10)]
    suite += [gen_random_kernels(4, (2, 2, 2, 3), rng),
              gen_random_kernels(5, (3, 3, 2, 2, 3), rng)]
    return suite


# ------------------------------------------------ xuất ra xlsx theo định dạng load_data.py
def export_xlsx(case, folder="data"):
    """Ghi Input_Rho (ô phức dạng chuỗi 'a+bj') và sheet Truth. Tên file chứa _c{n}_."""
    import os
    import pandas as pd
    n = case["n"]
    rows = [[None] * n]
    for rho in case["rhos"]:
        for r in rho:
            rows.append([f"{complex(z).real:.17g}{complex(z).imag:+.17g}j" for z in r])
        rows += [[None] * n, [None] * n]
    truth = {k: str(v) for k, v in case["truth"].items()}
    fname = f"chaos_{case['name']}_c{n}_.xlsx"
    path = os.path.join(folder, fname)
    with pd.ExcelWriter(path) as w:
        pd.DataFrame(rows).to_excel(w, sheet_name="Input_Rho", header=False, index=False)
        pd.DataFrame(list(truth.items()), columns=["key", "value"]).to_excel(
            w, sheet_name="Truth", index=False)
    return fname
