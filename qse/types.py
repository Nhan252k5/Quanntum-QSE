from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FeasibilityResult:
    """
    Kết quả Bước A hoặc bài feasibility bên trong DBRGE. Để định nghĩa các kiểu dữ liệu kết quả trả về

    decision:
        FEASIBLE
        INFEASIBLE
        UNKNOWN
    """

    decision: str
    status: str

    effects: Optional[list[np.ndarray]] = None

    solver: Optional[str] = None

    completeness_residual: Optional[float] = None
    min_eigenvalue: Optional[float] = None

    @property
    def feasible(self) -> bool:
        return self.decision == "FEASIBLE"


@dataclass
class ExtremalityResult:
    effects: list[np.ndarray]

    extreme: bool
    ranks: list[int]

    nullity: int
    iterations: int

    history: list[dict] = field(
        default_factory=list
    )


@dataclass
class VerificationReport:
    valid: bool
    physical_valid: bool
    extreme: bool

    completeness_residual: float
    exclusion_residual: float
    support_residual: float
    min_eigenvalue: float

    ranks: list[int]
    active_count: int

    sum_rank_squared: int
    D_S: int
    D_K: int

    nullity_phi: int
    max_pairwise_intersection: int


@dataclass
class PipelineResult:
    feasible: bool
    status: str = ""
    ambiguities: list = field(default_factory=list)

    effects: Optional[list[np.ndarray]] = None

    verification: Optional[
        VerificationReport
    ] = None

    initial_feasibility: Optional[
        FeasibilityResult
    ] = None

    initial_extreme: Optional[
        ExtremalityResult
    ] = None