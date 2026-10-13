from dataclasses import dataclass
from typing import (
    Callable,
    Hashable,
    Optional,
)

import numpy as np


GroupElement = Hashable


@dataclass(frozen=True)
class SymmetrySpec:
    """
    Mô tả đối xứng hữu hạn.

    elements:
        Danh sách phần tử g ∈ G.

    unitaries[g]:
        Ma trận U_g.

    action(g, i):
        Trả về outcome g·i.

        Quy ước index Python:
            i = 0, ..., k-1.

    base_index:
        Outcome dùng làm seed.
        Mặc định chọn outcome 0.
    """

    elements: tuple[GroupElement, ...]

    unitaries: dict[
        GroupElement,
        np.ndarray
    ]

    action: Callable[
        [GroupElement, int],
        int
    ]

    base_index: int = 0


@dataclass
class SymmetryStructure:
    """
    Cấu trúc được suy ra từ SymmetrySpec.
    """

    stabilizer: list[GroupElement]

    # representatives[i] = g_i
    # sao cho g_i · base = i
    representatives: dict[
        int,
        GroupElement
    ]


@dataclass
class SeedResult:
    decision: str
    status: str

    seed: Optional[np.ndarray] = None

    solver: Optional[str] = None

    psd_residual: Optional[float] = None
    stabilizer_residual: Optional[float] = None
    completeness_residual: Optional[float] = None

    @property
    def feasible(self):
        return self.decision == "FEASIBLE"


@dataclass
class CovariantExtremalityReport:
    extreme_in_covariant_set: bool

    domain_dimension: int
    map_rank: int
    nullity: int

    invariant_support_dimension: int


@dataclass
class SymmetryVerificationReport:
    valid: bool

    transitive: bool
    state_covariant: bool
    povm_covariant: bool

    psd_valid: bool
    completeness_valid: bool
    exclusion_valid: bool

    state_covariance_residual: float
    povm_covariance_residual: float
    completeness_residual: float
    exclusion_residual: float
    min_eigenvalue: float

    ranks: list[int]

    extreme_in_covariant_set: bool
    extreme_in_full_sqse: bool

    D_K: int
    commutant_dimension: int


@dataclass
class SymmetryPipelineResult:
    feasible: bool

    seed: Optional[np.ndarray] = None
    effects: Optional[list[np.ndarray]] = None

    seed_result: Optional[SeedResult] = None

    covariant_extremality: Optional[
        CovariantExtremalityReport
    ] = None

    verification: Optional[
        SymmetryVerificationReport
    ] = None