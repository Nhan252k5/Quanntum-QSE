from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class NumericalConfig:
    """
    Ngưỡng số học theo CHUỖI PHÂN CẤP, suy ra từ 2 đại lượng gốc:

        input_precision : sai số của dữ liệu đầu vào rho
                          (1e-12 cho dữ liệu sinh bằng máy, 1e-6 nếu nhập Excel 6 chữ số)
        sdp_accuracy    : độ chính xác yêu cầu solver đạt được

        noise   = max(n * input_precision, sdp_accuracy)       mức "rác" số học
        psd / completeness / exclusion tol = 10 * noise        kiểm tra vật lý
        support_tol = sqrt(exclusion_tol)                      (sai lệch bậc 1)
        rank_tol   = 1e3 * noise                               (>= 100 x psd_tol)
        kernel_tol = 10 * n * input_precision
        null_tol   = max(null_rtol * sigma_max,
                         10 * noise / lambda_min_kept)         (Davis–Kahan)

    Mọi quyết định có giá trị nằm trong vòng `margin_decades` quanh ngưỡng
    sẽ được ghi vào danh sách cảnh báo AMBIGUOUS.
    """

    input_precision: float = 1e-12
    sdp_accuracy: float = 1e-10
    margin_decades: float = 1.0

    null_rtol: float = 1e-9

    boundary_rtol: float = 1e-10
    boundary_step_tol: float = 1e-14

    max_extremify_iter: int = 500

    solver_order: tuple[str, ...] = ("CLARABEL", "SCS")
    solver_verbose: bool = False

    # ---------- ngưỡng cũ: CHỈ còn dùng cho nhánh symmetry ----------
    hermitian_tol: float = 1e-10
    trace_tol: float = 1e-10
    kernel_atol: float = 1e-12
    kernel_rtol: float = 1e-10
    rank_atol: float = 1e-12
    rank_rtol: float = 1e-9
    null_atol: float = 1e-12
    psd_tol: float = 1e-9
    completeness_tol: float = 1e-8
    exclusion_tol: float = 1e-9
    support_tol: float = 1e-8
    boundary_atol: float = 1e-12

    # ---------- các ngưỡng suy ra (nhánh reduction) ----------
    def noise(self, n: int) -> float:
        return max(n * self.input_precision, self.sdp_accuracy)

    def tol_hermitian(self, n): return 10 * n * self.input_precision + 1e-14
    def tol_trace(self, n):     return 10 * n * self.input_precision + 1e-14
    def tol_input_psd(self, n): return 10 * n * self.input_precision + 1e-14
    def tol_kernel(self, n):    return 10 * n * self.input_precision + 1e-14

    def tol_psd(self, n):          return 10 * self.noise(n)
    def tol_completeness(self, n): return 10 * self.noise(n)
    def tol_exclusion(self, n):    return 10 * self.noise(n)
    # ||M - PMP|| là sai lệch BẬC 1 (~ sqrt(leak)), còn Tr(rho M) là bậc 2,
    # nên ngưỡng support phải là căn của ngưỡng exclusion.
    def tol_support(self, n):      return float(np.sqrt(self.tol_exclusion(n)))
    def tol_rank(self, n):         return 1e3 * self.noise(n)

    def tol_null(self, n, sigma_max, lam_min_kept):
        dk = 10 * self.noise(n) / max(lam_min_kept, 1e-300)
        return max(self.null_rtol * sigma_max, dk, 1e-14)


DEFAULT_CONFIG = NumericalConfig()
