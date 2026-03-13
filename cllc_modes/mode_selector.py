from __future__ import annotations

import numpy as np

from .mode_ac import ACSolver
from .mode_bcb import BCBSolver
from .mode_ca import CASolver
from .mode_cb import CBSolver
from .mode_cba import CBASolver
from .types import OperatingPoint, SolveResult


class OperatingPointSolver:
    """Reusable operating-point solver with persistent mode solver instances."""

    def __init__(self) -> None:
        self.candidates = [ACSolver(), BCBSolver(), CBSolver(), CBASolver(), CASolver()]
        self._solver_by_mode = {solver.mode_name: solver for solver in self.candidates}
        self._last_success_mode: str | None = None

    @staticmethod
    def _residual_key(result: SolveResult) -> float:
        return result.max_residual if result.max_residual is not None else float("inf")

    @staticmethod
    def _maybe_update_initial_guess(solver, result: SolveResult) -> None:
        if result.success and result.params:
            solver.initial_guess = np.array(list(result.params.values()), dtype=float)

    def _build_ordered_candidates(self):
        """
        Try the mode from the last successful point first.
        This follows the assumption that small operating-point changes tend to
        remain in the same mode and near the previous solution.
        """
        if self._last_success_mode is None:
            return self.candidates

        preferred = self._solver_by_mode.get(self._last_success_mode)
        if preferred is None:
            return self.candidates

        return [preferred] + [solver for solver in self.candidates if solver is not preferred]

    def solve_point(self, op: OperatingPoint) -> SolveResult:
        results: list[SolveResult] = []

        for solver in self._build_ordered_candidates():
            result = solver.solve(op)
            self._maybe_update_initial_guess(solver, result)
            results.append(result)

        feasible = [r for r in results if r.success]
        if feasible:
            feasible.sort(key=self._residual_key)
            best = feasible[0]
            self._last_success_mode = best.mode
            return best

        results.sort(key=lambda r: (0 if r.success else 1, self._residual_key(r)))
        return results[0]

    def solve(self, F: float, k: float, P: float) -> SolveResult:
        op = OperatingPoint(F=F, k=k, P=P)
        return self.solve_point(op)

_DEFAULT_SOLVER = OperatingPointSolver()

def solve_operating_point(F: float, k: float, P: float) -> SolveResult:
    """
    Module-level convenience API that keeps warm-start state across calls.
    """
    return _DEFAULT_SOLVER.solve(F=F, k=k, P=P)

