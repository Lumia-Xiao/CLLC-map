from __future__ import annotations

import numpy as np

from .mode_ac import ACSolver
from .mode_bcb import BCBSolver
from .mode_ca import CASolver
from .mode_cba import CBASolver
from .types import OperatingPoint, SolveResult


class OperatingPointSolver:
    """Reusable operating-point solver with persistent mode solver instances."""

    def __init__(self) -> None:
        self.candidates = [ACSolver(), BCBSolver(), CBASolver(), CASolver()]

    @staticmethod
    def _residual_key(result: SolveResult) -> float:
        return result.max_residual if result.max_residual is not None else float("inf")

    @staticmethod
    def _maybe_update_initial_guess(solver, result: SolveResult) -> None:
        if result.success and result.params:
            solver.initial_guess = np.array(list(result.params.values()), dtype=float)

    def solve_point(self, op: OperatingPoint) -> SolveResult:
        results: list[SolveResult] = []
        for solver in self.candidates:
            result = solver.solve(op)
            self._maybe_update_initial_guess(solver, result)
            results.append(result)

        feasible = [r for r in results if r.success]
        if feasible:
            feasible.sort(key=self._residual_key)
            return feasible[0]

        results.sort(key=lambda r: (0 if r.success else 1, self._residual_key(r)))
        return results[0]

    def solve(self, F: float, k: float, P: float) -> SolveResult:
        op = OperatingPoint(F=F, k=k, P=P)
        return self.solve_point(op)


def solve_operating_point(F: float, k: float, P: float) -> SolveResult:
    return OperatingPointSolver().solve(F=F, k=k, P=P)
