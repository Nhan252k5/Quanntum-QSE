# QSE — Quantum State Exclusion Toolkit

Thư viện Python phục vụ nghiên cứu **Quantum State Exclusion (QSE/SQSE)** với các mục tiêu chính:

- tìm POVM loại trừ hoàn hảo;
- khai thác kernel của các trạng thái lượng tử;
- giải feasibility bằng SDP;
- đưa nghiệm về điểm cực biên;
- giảm số outcome bằng DBRGE;
- kiểm tra các invariant \(D_S, D_K\);
- hỗ trợ một nhánh riêng cho POVM hiệp biến theo đối xứng nhóm.

---

## 1. Bài toán

Cho các trạng thái lượng tử

\[
\rho_i\succeq0,\qquad \operatorname{Tr}(\rho_i)=1.
\]

Ta cần tìm POVM \(\{M_i\}\) thỏa

\[
M_i\succeq0,\qquad \sum_i M_i=I_n,
\]

và perfect exclusion

\[
\operatorname{Tr}(\rho_iM_i)=0.
\]

Với ma trận PSD:

\[
\operatorname{Tr}(\rho_iM_i)=0
\iff
\operatorname{range}(M_i)\subseteq\ker(\rho_i).
\]

Đặt \(K_i=\ker(\rho_i)\). Nếu \(Q_i^{(K)}\) là basis trực chuẩn của \(K_i\), ta parameterize

\[
M_i=Q_i^{(K)}X_iQ_i^{(K)\dagger},
\qquad X_i\succeq0.
\]

---

## 2. Pipeline chính

```text
rho_i
  │
  ▼
Validate states
  │
  ▼
Compute kernels K_i
  │
  ▼
SDP feasibility
  │
  ▼
Extremify
  │
  ▼
DBRGE
  │
  ▼
Verify invariants
  │
  ▼
Final POVM
```

Tóm tắt:

\[
\boxed{
\rho_i\to K_i\to SDP\to Extremify\to DBRGE\to Verify
}
\]

---

## 3. Cấu trúc project

```text
qse_project/
├── qse/
│   ├── __init__.py
│   ├── config.py
│   ├── types.py
│   ├── states/
│   │   ├── validation.py
│   │   └── kernels.py
│   ├── linalg/
│   │   ├── hermitian.py
│   │   └── numerical_rank.py
│   ├── feasibility/
│   │   └── sdp.py
│   ├── extremality/
│   │   ├── supports.py
│   │   ├── phi.py
│   │   ├── nullspace.py
│   │   ├── boundary.py
│   │   └── extremify.py
│   ├── elimination/
│   │   └── dbrge.py
│   ├── invariants/
│   │   ├── dimensions.py
│   │   ├── intersections.py
│   │   └── verify.py
│   ├── symmetry/
│   │   ├── types.py
│   │   ├── group_action.py
│   │   ├── averaging.py
│   │   ├── seed.py
│   │   ├── covariant_povm.py
│   │   ├── covariant_extremality.py
│   │   ├── commutant.py
│   │   └── verify.py
│   └── pipeline/
│       ├── reduction.py
│       └── symmetry.py
└── main.py
```

---

## 4. Cài đặt

Khuyến nghị Python 3.11+.

```bash
pip install numpy scipy cvxpy pytest
```

Kiểm tra solver CVXPY:

```python
import cvxpy as cp
print(cp.installed_solvers())
```

---

## 5. Numerical configuration

Các tolerance được quản lý tập trung trong `qse/config.py`.

Ví dụ:

```python
@dataclass(frozen=True)
class NumericalConfig:
    kernel_atol: float = 1e-12
    kernel_rtol: float = 1e-10
    rank_atol: float = 1e-12
    rank_rtol: float = 1e-9
    null_atol: float = 1e-12
    null_rtol: float = 1e-9
    psd_tol: float = 1e-9
    completeness_tol: float = 1e-8
    exclusion_tol: float = 1e-9
    support_tol: float = 1e-8
    boundary_atol: float = 1e-12
    boundary_rtol: float = 1e-10
```

Không nên dùng một `TOL` duy nhất cho kernel, rank, nullspace, PSD và residual.

---

## 6. Validation

File:

```text
qse/states/validation.py
```

Hàm chính:

```python
validate_density_matrix(rho)
validate_states(rhos)
```

Kiểm tra:

\[
\rho_i=\rho_i^\dagger,\qquad
\rho_i\succeq0,\qquad
\operatorname{Tr}(\rho_i)=1.
\]

---

## 7. Kernel extraction

File:

```text
qse/states/kernels.py
```

Từ eigendecomposition của \(
ho_i\), lấy các eigenvector ứng với eigenvalue được xem là zero số học để xây

\[
Q_i^{(K)}
\]

là basis trực chuẩn của

\[
K_i=\ker(\rho_i).
\]

Quy ước:

```text
QK_i = kernel basis
QS_i = support basis
```

---

## 8. SDP feasibility

File:

```text
qse/feasibility/sdp.py
```

Giải:

\[
M_i=Q_i^{(K)}X_iQ_i^{(K)\dagger},
\qquad X_i\succeq0,
\]

với constraint

\[
\sum_iM_i=I_n.
\]

Đây là feasibility SDP, thường dùng

```python
cp.Minimize(0)
```

với CLARABEL hoặc SCS.

Solver status luôn phải đi kèm post-verification.

---

## 9. Extremality

Với

\[
S_i=\operatorname{range}(M_i),\qquad
r_i=\operatorname{rank}(M_i),
\]

định nghĩa

\[
\Phi(H_1,\ldots,H_k)
=
\sum_iQ_i^{(S)}H_iQ_i^{(S)\dagger}.
\]

Trong code, `PhiData.matrix` là ma trận thực biểu diễn ánh xạ này.

Shape:

\[
n^2\times\sum_i r_i^2.
\]

Điều kiện cực biên:

\[
\boxed{\ker\Phi=\{0\}}
\]

hay

\[
\operatorname{nullity}(\Phi)=0.
\]

Tại extreme:

\[
\boxed{\sum_i r_i^2=D_S}.
\]

---

## 10. Nullspace bằng SVD

File:

```text
qse/extremality/nullspace.py
```

Dùng:

```python
U, s, Vh = np.linalg.svd(
    Phi,
    full_matrices=True
)
```

với

\[
\Phi=U\Sigma V^\dagger.
\]

Các right singular vectors ứng với singular value zero tạo basis của \(\ker\Phi\).

Nếu `nullity > 0`, chọn

\[
z\neq0,\qquad \Phi z\approx0,
\]

sau đó decode thành \(H_i\), rồi

\[
\Delta_i
=
Q_i^{(S)}H_iQ_i^{(S)\dagger}.
\]

Ta có

\[
\sum_i\Delta_i=0.
\]

---

## 11. Boundary step

File:

```text
qse/extremality/boundary.py
```

Xét

\[
M_i(t)=M_i+t\Delta_i.
\]

Trên support:

\[
B_i
=
Q_i^{(S)\dagger}M_iQ_i^{(S)},
\]

\[
H_i
=
Q_i^{(S)\dagger}\Delta_iQ_i^{(S)}.
\]

Whitening:

\[
C_i=B_i^{-1/2}H_iB_i^{-1/2}.
\]

Nếu

\[
C_iv=\mu v,
\]

thì PSD tương đương với

\[
1+t\mu\ge0.
\]

Boundary:

\[
\boxed{t=-1/\mu}.
\]

Các hàm chính:

```python
block_step_interval(...)
global_step_interval(...)
choose_boundary_step(...)
```

---

## 12. Extremify

File:

```text
qse/extremality/extremify.py
```

Flow:

```text
compute supports
      ↓
build Phi
      ↓
SVD
      ↓
nullity = 0 ? ── yes ──> extreme
      │
      no
      ↓
choose z in ker(Phi)
      ↓
decode Delta_i
      ↓
find PSD boundary
      ↓
M_i <- M_i + t Delta_i
      ↓
rank decreases
      ↓
repeat
```

---

## 13. DBRGE

File:

```text
qse/elimination/dbrge.py
```

DBRGE là:

\[
\boxed{
\text{Dynamic Bidirectional Rank-Driven Greedy Elimination}
}
\]

Mục tiêu là giảm số outcome active.

Với candidate \(j\), thuật toán không đơn giản đặt \(M_j=0\), mà giải lại SDP trên active set

\[
A\setminus\{j\}.
\]

Nếu còn feasible:

\[
SDP\to Extremify\to rerank\to resort.
\]

Thứ tự thử:

```text
rank nhỏ nhất
rank lớn nhất
rank nhỏ thứ hai
rank lớn thứ hai
...
```

Sau mỗi deletion thành công, ranks được tính lại.

**Lưu ý:** DBRGE là greedy heuristic; kết quả inclusion-minimal theo single-outcome deletion không đồng nghĩa global minimum cardinality.

---

## 14. Verification

File:

```text
qse/invariants/verify.py
```

Kiểm tra:

- \(M_i\succeq0\);
- \(\sum_iM_i\approx I\);
- \(\operatorname{Tr}(\rho_iM_i)\approx0\);
- \(M_i\approx P_iM_iP_i\);
- `nullity_phi == 0`;
- \(\sum_i r_i^2=D_S\);
- \(D_S\le D_K\);
- \(D_K\le\min(n^2,\sum_i d_i^2)\);
- \(s\le\min(k,D_K)\);
- \(S_i\cap S_j=\{0\}\) tại extreme.

Trong đó

\[
D_S
=
\dim_{\mathbb R}
\left(
\sum_i\mathcal L(S_i)
\right)
\]

và

\[
D_K
=
\dim_{\mathbb R}
\left(
\sum_i\mathcal L(K_i)
\right).
\]

Tại extreme:

\[
\boxed{
s
\le
\sum_i r_i^2
=
D_S
\le
D_K
\le
\min(n^2,\sum_i d_i^2)
}.
\]

---

## 15. Pairwise intersections

File:

```text
qse/invariants/intersections.py
```

Dùng công thức

\[
\dim(A\cap B)
=
\dim A+\dim B-\dim(A+B).
\]

Nếu `QA`, `QB` là basis matrices:

\[
\dim(A+B)
=
\operatorname{rank}([Q_A\;Q_B]).
\]

Do đó:

\[
\boxed{
\dim(A\cap B)
=
d_A+d_B-
\operatorname{rank}([Q_A\;Q_B])
}.
\]

Pairwise intersection bằng 0 chỉ là điều kiện cần cho extremality, không thay thế kiểm tra \(\ker\Phi=0\).

---

## 16. Nhánh symmetry

Nhánh symmetry được giữ riêng khỏi DBRGE.

```text
rho_i + group G
      ↓
check unitary representation
      ↓
check state covariance
      ↓
H = Stab(base)
      ↓
choose orbit representatives
      ↓
solve seed SDP
      ↓
M_i = U_g N U_g†
      ↓
covariant extremality
      ↓
full-SQSE extremality
      ↓
verify
```

Seed \(N=M_0\) thỏa:

\[
N\succeq0,
\]

\[
\operatorname{range}(N)\subseteq K_0,
\]

\[
U_hNU_h^\dagger=N
\quad\forall h\in H,
\]

và

\[
\sum_{gH}U_gNU_g^\dagger=I.
\]

Sau đó

\[
M_{g\cdot0}=U_gNU_g^\dagger.
\]

---

## 17. Twirling

Một SQSE solution bất kỳ không nhất thiết covariant.

Có thể covariantize bằng group average:

\[
\bar M_i
=
\frac1{|G|}
\sum_{g\in G}
U_gM_{g^{-1}\cdot i}U_g^\dagger.
\]

Code:

```text
qse/symmetry/averaging.py
```

---

## 18. Covariant extremality và full extremality

Phải phân biệt:

```python
extreme_in_covariant_set
```

và

```python
extreme_in_full_sqse
```

Một POVM có thể cực biên trong tập covariant nhưng không cực biên trong toàn SQSE.

Không được dùng commutant dimension như một chặn tổng quát cho

\[
\sum_i r_i^2.
\]

Nên có regression test cho phản ví dụ \(Z_4\).

---

## 19. Ví dụ sử dụng

```python
from qse import (
    NumericalConfig,
    run_reduction_pipeline,
)

config = NumericalConfig()

result = run_reduction_pipeline(
    rhos,
    config
)

if not result.feasible:
    print("SQSE infeasible.")
else:
    report = result.verification

    print("===== FINAL REPORT =====")
    print("valid =", report.valid)
    print("extreme =", report.extreme)
    print("ranks =", report.ranks)
    print("active =", report.active_count)
    print("sum r_i^2 =", report.sum_rank_squared)
    print("D_S =", report.D_S)
    print("D_K =", report.D_K)
    print("nullity Phi =", report.nullity_phi)
    print(
        "completeness residual =",
        report.completeness_residual
    )
    print(
        "exclusion residual =",
        report.exclusion_residual
    )
```

---

## 20. Ví dụ output

```text
===== FINAL REPORT =====
valid = True
extreme = True

ranks = [2, 0, 2]
active = 2

sum r_i^2 = 8
D_S = 8
D_K = 10

nullity Phi = 0

completeness residual = 1.9e-15
exclusion residual = 0.0
```

Ý nghĩa:

\[
r=[2,0,2],
\]

nên outcome thứ hai inactive và

\[
s=2.
\]

Đồng thời

\[
2^2+0^2+2^2=8=D_S
\]

và

\[
\operatorname{nullity}\Phi=0,
\]

nên POVM là extreme.

---

## 21. Testing

Nên test invariant thay vì hard-code active indices:

```python
assert report.extreme
assert report.sum_rank_squared == report.D_S
assert report.D_S <= report.D_K
assert report.active_count <= report.D_K
assert report.nullity_phi == 0
```

Nếu null direction khác nhau dẫn đến các extreme point khác nhau nhưng cùng invariant, test vẫn ổn định.

---

## 22. Numerical sensitivity

Đối với kết quả nghiên cứu, nên chạy lại với nhiều tolerance như

```text
1e-8
1e-9
1e-10
```

và so sánh:

- ranks;
- active set;
- \(D_S\);
- \(D_K\);
- nullity;
- residuals.

Nếu kết luận thay đổi theo tolerance, nên đánh dấu kết quả `AMBIGUOUS`.

---

## 23. Experimental Dykstra

```text
experimental/projection_dykstra.py
```

chỉ dùng cho benchmark hoặc nghiên cứu projection methods.

Pipeline chính vẫn dùng SDP vì projection failure không phải chứng chỉ infeasibility.

---

## 24. Giới hạn hiện tại

1. DBRGE không chứng minh global minimum số outcome.
2. Numerical extremality phụ thuộc threshold SVD/rank.
3. Solver status phải được hậu kiểm bằng residual.
4. Symmetry branch hiện tập trung vào finite transitive group actions.
5. Pairwise support intersection bằng 0 không đủ để kết luận extreme.
6. Commutant dimension không phải chặn tổng quát cho \(\sum_i r_i^2\).
7. Không nên PSD-clip độc lập từng \(M_i\) sau boundary update vì có thể phá completeness.

---

## 25. Quy ước ký hiệu

```text
rho_i   : quantum state
K_i     : ker(rho_i)
QK_i    : orthonormal basis of K_i
M_i     : POVM effect
S_i     : range(M_i)
QS_i    : orthonormal basis of S_i
r_i     : rank(M_i)
d_i     : dim(K_i)
Phi     : extremality map
D_S     : dim sum_i L(S_i)
D_K     : dim sum_i L(K_i)
s       : number of active outcomes
```

---

## 26. Công thức cốt lõi

\[
\boxed{
\operatorname{Tr}(\rho_iM_i)=0
\iff
\operatorname{range}(M_i)\subseteq\ker\rho_i
}
\]

\[
\boxed{
M\text{ extreme}
\iff
\ker\Phi=\{0\}
}
\]

\[
\boxed{
\dim\ker\Phi
=
\sum_i r_i^2-D_S
}
\]

\[
\boxed{
s
\le
\sum_i r_i^2
=
D_S
\le
D_K
\le
\min(n^2,\sum_i d_i^2)
}
\]

\[
\boxed{
t=-1/\mu
}
\]

---

## 27. Phát triển tiếp

- chi tiết hóa `VerificationReport` thành từng boolean check;
- lưu DBRGE history;
- exhaustive subset search cho \(k\) nhỏ;
- tolerance sensitivity sweep;
- export POVM sang JSON/NPY/XLSX;
- solver benchmark;
- covariant extremification;
- regression test cho symmetry counterexamples;
- high-precision verification cho trường hợp gần threshold.
