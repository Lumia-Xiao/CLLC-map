from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from scipy.optimize import OptimizeResult, root

from .types import OperatingPoint, SolveResult


class ModeSolverBase(ABC):
    mode_name: str = "UNDEFINED"
    _warm_starts: dict[str, tuple[OperatingPoint, np.ndarray]] = {}

    @abstractmethod
    def solve(self, op: OperatingPoint) -> SolveResult:
        raise NotImplementedError

    def _build_guess_pool(self, op: OperatingPoint, initial_guess: np.ndarray) -> list[np.ndarray]:
        guesses: list[np.ndarray] = []

        warm = self._warm_starts.get(self.mode_name)
        if warm is not None:
            _, warm_x = warm
            if warm_x.shape == initial_guess.shape:
                guesses.append(np.array(warm_x, dtype=float))

        base = np.array(initial_guess, dtype=float)
        guesses.append(base)
        guesses.append(base * 0.9)
        guesses.append(base * 1.1)

        mag = np.maximum(np.abs(base), 1.0)
        guesses.append(base + 0.05 * mag)
        guesses.append(base - 0.05 * mag)

        unique: list[np.ndarray] = []
        for g in guesses:
            if not any(np.allclose(g, u, atol=1e-12, rtol=1e-9) for u in unique):
                unique.append(g)
        return unique

    def _solve_with_restarts(
        self,
        op: OperatingPoint,
        equations: Callable[[np.ndarray], np.ndarray],
        initial_guess: np.ndarray,
        method: str,
        maxfev: int,
    ) -> tuple[OptimizeResult, np.ndarray, float]:
        best_sol: OptimizeResult | None = None
        best_residual: np.ndarray | None = None
        best_max_residual = float("inf")

        for guess in self._build_guess_pool(op, initial_guess):
            sol = root(fun=equations, x0=guess, method=method, options={"maxfev": maxfev})
            residual = np.asarray(equations(sol.x), dtype=float)
            max_residual = float(np.max(np.abs(residual)))

            better = False
            if best_sol is None:
                better = True
            else:
                if bool(sol.success) and not bool(best_sol.success):
                    better = True
                elif bool(sol.success) == bool(best_sol.success) and max_residual < best_max_residual:
                    better = True

            if better:
                best_sol = sol
                best_residual = residual
                best_max_residual = max_residual

        assert best_sol is not None and best_residual is not None
        if bool(best_sol.success):
            self._warm_starts[self.mode_name] = (op, np.array(best_sol.x, dtype=float))

        return best_sol, best_residual, best_max_residual