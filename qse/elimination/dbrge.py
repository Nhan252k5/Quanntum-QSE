from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from qse.config import NumericalConfig

from qse.feasibility.sdp import (
    solve_feasible_povm,
)
from qse.extremality.supports import (
    compute_supports,
)
from qse.extremality.extremify import (
    extremify,
)

@dataclass
class EliminationRun:
    effects: list[np.ndarray]
    active: list[int]          # outcome còn lại (s = len(active))
    confirmed: list[int]       # xóa -> INFEASIBLE: bắt buộc giữ (chắc chắn, theo đơn điệu)
    undecided: list[int]       # xóa -> UNKNOWN / Extremify lỗi: giữ lại nhưng KHÔNG chắc
    sdp_calls: int
    order: list[int]           # thứ tự thực sự đã thử
    seed: Optional[int] = None

@dataclass
class EliminationResult:
    best: EliminationRun
    runs: list[EliminationRun] = field(default_factory=list)
    lower_bound: int = 1       # cận dưới s_min 
    @property
    def effects(self):
        return self.best.effects

    @property
    def one_minimal(self) -> bool:
        """True nếu mọi outcome còn lại đều được CHỨNG MINH là không xóa được."""
        return not self.best.undecided

    @property
    def certified_optimal(self) -> bool:
        return self.one_minimal and len(self.best.active) == self.lower_bound


def _bidirectional_order(
    ordered_indices,
):
    """
    Tạo thứ tự:

        smallest,
        largest,
        second smallest,
        second largest,
        ...

    Đây tương đương 2 con trỏ
    lo / hi trong sơ đồ DBRGE.
    """

    out = []

    lo = 0
    hi = len(ordered_indices) - 1

    take_low = True

    while lo <= hi:

        if take_low:
            out.append(
                ordered_indices[lo]
            )
            lo += 1

        else:
            out.append(
                ordered_indices[hi]
            )
            hi -= 1

        take_low = not take_low

    return out


def dbrge(
    kernel_bases,
    Ms,
    config: NumericalConfig,
):
    """
    Dynamic Bidirectional Rank-Driven
    Greedy Elimination.

    Sau mỗi deletion thành công:

        solve SDP
        -> Extremify
        -> recompute ranks
        -> resort
        -> restart.
    """

    n = Ms[0].shape[0]

    Ms_current = Ms

    while True:

        _, ranks = compute_supports(
            Ms_current,
            config
        )

        active = [
            i
            for i, r in enumerate(ranks)
            if r > 0
        ]

        # rank nhỏ -> lớn
        ordered = sorted(
            active,
            key=lambda i: (
                ranks[i],
                i
            )
        )

        candidates = (
            _bidirectional_order(
                ordered
            )
        )

        deleted_something = False

        for candidate in candidates:

            trial_active = [
                i
                for i in active
                if i != candidate
            ]

            if len(trial_active) == 0:
                continue

            feasibility = (
                solve_feasible_povm(
                    kernel_bases,
                    n,
                    config,
                    active_indices=(
                        trial_active
                    ),
                )
            )

            # UNKNOWN không được xem
            # như INFEASIBLE
            if not feasibility.feasible:
                #candidate không thể bị xóa
                continue
            elif feasibility.decision == "UNKNOWN":
                # numerical ambiguity:
                # không chấp nhận deletion
                continue    
            extreme = extremify(
                feasibility.effects,
                config
            )

            if not extreme.extreme:
                # numerical ambiguity:
                # không chấp nhận deletion
                continue

            Ms_current = (
                extreme.effects
            )

            deleted_something = True

            # Dynamic:
            # rank đã thay đổi,
            # thoát để sort lại từ đầu
            break

        if not deleted_something:
            break

    return Ms_current