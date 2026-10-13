import numpy as np

from qse.config import NumericalConfig
from qse.linalg import margins


def kernel_basis(rho, config: NumericalConfig, index=None):
    """
    K = ker(rho): eigenvalue <= tol_kernel(n) = 10 n input_precision.

    Lưu ý: eigenvector của cụm kernel bị trộn với các hướng có eigenvalue
    nhỏ cỡ lambda một lượng ~ eps_mach / lambda. Vì vậy mọi eigenvalue nằm
    gần ngưỡng đều được ghi AMBIGUOUS.
    """
    rho = (rho + rho.conj().T) / 2
    n = rho.shape[0]
    eigvals, eigvecs = np.linalg.eigh(rho)
    tol = config.tol_kernel(n)
    margins.check("kernel", eigvals, tol, config, state=index)
    return eigvecs[:, eigvals <= tol]


def compute_kernel_bases(rhos, config: NumericalConfig):
    return [kernel_basis(rho, config, index=i) for i, rho in enumerate(rhos)]


def kernel_projector(QK):
    return QK @ QK.conj().T
