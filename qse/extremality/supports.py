from qse.config import NumericalConfig
from qse.linalg.numerical_rank import psd_support


def compute_supports(Ms, config: NumericalConfig):
    bases, ranks = [], []
    for M in Ms:
        Q, r, _ = psd_support(M, config)
        bases.append(Q)
        ranks.append(r)
    return bases, ranks
