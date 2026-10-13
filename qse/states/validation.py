import numpy as np

from qse.config import NumericalConfig, DEFAULT_CONFIG


def validate_density_matrix(rho, config: NumericalConfig = DEFAULT_CONFIG) -> None:
    """
    rho = rho^dagger, rho >= 0, Tr rho = 1  — trong phạm vi sai số ĐẦU VÀO
    (config.input_precision), không phải sai số máy.
    """
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho phải là ma trận vuông.")
    n = rho.shape[0]

    herm_err = np.linalg.norm(rho - rho.conj().T, ord="fro")
    if herm_err > config.tol_hermitian(n):
        raise ValueError(f"rho không Hermitian. error={herm_err:.3e} > {config.tol_hermitian(n):.1e}")

    rho_h = (rho + rho.conj().T) / 2
    lam_min = np.linalg.eigvalsh(rho_h).min()
    if lam_min < -config.tol_input_psd(n):
        raise ValueError(
            f"rho không PSD. lambda_min={lam_min:.3e} < -{config.tol_input_psd(n):.1e}. "
            "Nếu dữ liệu bị làm tròn, hãy đặt input_precision phù hợp.")

    tr_err = abs(np.trace(rho_h).real - 1.0)
    if tr_err > config.tol_trace(n):
        raise ValueError(f"Tr(rho) != 1. error={tr_err:.3e} > {config.tol_trace(n):.1e}")


def validate_states(rhos, config: NumericalConfig = DEFAULT_CONFIG) -> None:
    if len(rhos) < 2:
        raise ValueError("Cần ít nhất 2 trạng thái.")
    n = rhos[0].shape[0]
    for i, rho in enumerate(rhos):
        if rho.shape != (n, n):
            raise ValueError(f"rho[{i}] có dimension khác.")
        validate_density_matrix(rho, config)
