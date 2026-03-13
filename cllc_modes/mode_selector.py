from __future__ import annotations

from .mode_ac import ACSolver
from .mode_bcb import BCBSolver
from .mode_ca import CASolver
from .mode_cba import CBASolver
from .types import OperatingPoint, SolveResult


def solve_operating_point(F: float, k: float, P: float) -> SolveResult:
    op = OperatingPoint(F=F, k=k, P=P)
    candidates = [ACSolver(), BCBSolver(), CBASolver(), CASolver()]
    results: list[SolveResult] = [solver.solve(op) for solver in candidates]
    feasible = [r for r in results if r.success]
    if feasible:
        feasible.sort(key=lambda r: (r.max_residual if r.max_residual is not None else float("inf")))
        return feasible[0]
    results.sort(key=lambda r: (0 if r.success else 1, r.max_residual if r.max_residual is not None else float("inf")))
    return results[0]
