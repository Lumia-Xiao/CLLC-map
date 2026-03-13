from __future__ import annotations

from .solver_base import ModeSolverBase
from .types import OperatingPoint, SolveResult


class _PlaceholderSolver(ModeSolverBase):
    def solve(self, op: OperatingPoint) -> SolveResult:
        return SolveResult(
            mode=self.mode_name,
            success=False,
            operating_point=op,
            message=f"{self.mode_name} solver placeholder only. Not implemented yet.",
        )


class ACSolver(_PlaceholderSolver):
    mode_name = "AC"


class BCBSolver(_PlaceholderSolver):
    mode_name = "BCB"


class CBASolver(_PlaceholderSolver):
    mode_name = "CBA"


class CBSolver(_PlaceholderSolver):
    mode_name = "CB"