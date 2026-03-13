from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class OperatingPoint:
    F: float
    k: float
    P: float


@dataclass(slots=True)
class SolveResult:
    mode: str
    success: bool
    operating_point: OperatingPoint
    params: dict[str, float] = field(default_factory=dict)
    max_residual: float | None = None
    residual_vector: list[float] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    waveforms: dict[str, Any] = field(default_factory=dict)
