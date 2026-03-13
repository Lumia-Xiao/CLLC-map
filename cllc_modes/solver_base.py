from __future__ import annotations

from abc import ABC, abstractmethod

from .types import OperatingPoint, SolveResult


class ModeSolverBase(ABC):
    mode_name: str = "UNDEFINED"

    @abstractmethod
    def solve(self, op: OperatingPoint) -> SolveResult:
        raise NotImplementedError
