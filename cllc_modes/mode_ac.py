from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root

from . import config
from .checks import check_positive_over_interval
from .solver_base import ModeSolverBase
from .stages import make_stage4
from .types import OperatingPoint, SolveResult


@dataclass(slots=True)
class ACParams:
    M: float
    theta_a: float
    theta_c: float
    C1C: float
    C2C: float
    C3C: float
    C4C: float
    C1A: float
    C2A: float
    C3A: float
    C4A: float

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class ACSolver(ModeSolverBase):
    mode_name = "AC"

    def __init__(self) -> None:
        self.initial_guess = np.array([0.8, 0.38, 2.24, -0.08, 0.02, -1.22, 0.59, -0.39, -0.24, -1.16, -1.00], dtype=float)

    @staticmethod
    def _unpack(x: np.ndarray) -> ACParams:
        return ACParams(*map(float, x.tolist()))

    @staticmethod
    def _build_stages(p: ACParams, k: float) -> tuple[dict[str, object], dict[str, object]]:
        stage_c = make_stage4(p.C1C, p.C2C, p.C3C, p.C4C, 2.0, -2.0 * p.M, 1.0, p.M, k)
        stage_a = make_stage4(p.C1A, p.C2A, p.C3A, p.C4A, 2.0, 2.0 * p.M, 1.0, -p.M, k)
        return stage_c, stage_a

    def _equations(self, x: np.ndarray, op: OperatingPoint) -> np.ndarray:
        p = self._unpack(x)
        stage_c, stage_a = self._build_stages(p, op.k)
        I1C, I2C, V1C, V2C = stage_c["I1"], stage_c["I2"], stage_c["V1"], stage_c["V2"]
        I1A, I2A, V1A, V2A, ImaA = stage_a["I1"], stage_a["I2"], stage_a["V1"], stage_a["V2"], stage_a["Ima"]

        power_c = quad(lambda th: float(I1C(th)), 0.0, p.theta_c, limit=200)[0]
        power_a = quad(lambda th: float(I1A(th)), 0.0, p.theta_a, limit=200)[0]
        half_period = math.pi / op.F

        return np.array([
            I1A(p.theta_a) - I1C(0.0),
            I2A(p.theta_a) - I2C(0.0),
            V1A(p.theta_a) - V1C(0.0),
            V2A(p.theta_a) - V2C(0.0),
            I1C(p.theta_c) + I1A(0.0),
            I2C(p.theta_c) + I2A(0.0),
            V1C(p.theta_c) + V1A(0.0),
            V2C(p.theta_c) + V2A(0.0),
            I1A(p.theta_a) - ImaA(p.theta_a),
            p.theta_a + p.theta_c - half_period,
            (op.F / math.pi) * (power_c + power_a) - op.P,
        ], dtype=float)

    @staticmethod
    def _make_join(theta_a: float, half_period: float, fa, fc):
        def joined(theta: np.ndarray | float):
            t = np.asarray(theta, dtype=float)
            out = np.empty_like(t)
            m1 = (t >= -half_period) & (t <= theta_a - half_period)
            m2 = (t > theta_a - half_period) & (t <= 0.0)
            m3 = (t > 0.0) & (t <= theta_a)
            m4 = (t > theta_a) & (t <= half_period)
            out[m1] = -np.asarray(fa(t[m1] + half_period), dtype=float)
            out[m2] = -np.asarray(fc(t[m2] - theta_a + half_period), dtype=float)
            out[m3] = np.asarray(fa(t[m3]), dtype=float)
            out[m4] = np.asarray(fc(t[m4] - theta_a), dtype=float)
            if np.isscalar(theta):
                return float(out.reshape(-1)[0])
            return out

        return joined

    def _build_waveforms(self, p: ACParams, op: OperatingPoint) -> tuple[dict[str, object], float]:
        stage_c, stage_a = self._build_stages(p, op.k)
        half_period = math.pi / op.F
        join = lambda fa, fc: self._make_join(p.theta_a, half_period, fa, fc)
        return {
            "Vin": join(stage_a["Vin"], stage_c["Vin"]),
            "Vo": join(stage_a["Vo"], stage_c["Vo"]),
            "Ir1": join(stage_a["I1"], stage_c["I1"]),
            "Vr1": join(stage_a["V1"], stage_c["V1"]),
            "Ir2": join(stage_a["I2"], stage_c["I2"]),
            "Vr2": join(stage_a["V2"], stage_c["V2"]),
            "Ima": join(stage_a["Ima"], stage_c["Ima"]),
            "stage_a": stage_a,
            "stage_c": stage_c,
        }, half_period

    def solve(self, op: OperatingPoint) -> SolveResult:
        sol = root(fun=lambda x: self._equations(x, op), x0=self.initial_guess, method=config.ROOT_METHOD,
                   options={"maxfev": config.MAX_FEVAL, "xtol": config.SOLVER_TOL})
        p = self._unpack(sol.x)
        residual = self._equations(sol.x, op)
        max_residual = float(np.max(np.abs(residual)))
        waveforms, half_period = self._build_waveforms(p, op)
        stage_c, stage_a = waveforms["stage_c"], waveforms["stage_a"]
        checks = {
            "Stage C: Ir1 > Ima": check_positive_over_interval(lambda th: np.asarray(stage_c["I1"](th), dtype=float) - np.asarray(stage_c["Ima"](th), dtype=float), 0.0, p.theta_c, tol=config.CHECK_TOL, n=config.DENSE_CHECK_POINTS),
            "Stage A: Ir1 < Ima": check_positive_over_interval(lambda th: np.asarray(stage_a["Ima"](th), dtype=float) - np.asarray(stage_a["I1"](th), dtype=float), 0.0, p.theta_a, tol=config.CHECK_TOL, n=config.DENSE_CHECK_POINTS),
            "theta_a_positive": {"passed": p.theta_a > 0.0, "value": p.theta_a},
            "theta_c_positive": {"passed": p.theta_c > 0.0, "value": p.theta_c},
            "M_positive": {"passed": p.M > 0.0, "value": p.M},
        }
        all_checks_pass = all(bool(item.get("passed", False)) for item in checks.values())
        success = bool(sol.success) and (max_residual < config.SOLVER_TOL) and all_checks_pass
        return SolveResult(mode=self.mode_name, success=success, operating_point=op, params=p.as_dict(), max_residual=max_residual, residual_vector=[float(v) for v in residual.tolist()], checks=checks, message=sol.message, waveforms={"half_period": half_period, "Vin": waveforms["Vin"], "Vo": waveforms["Vo"], "Ir1": waveforms["Ir1"], "Vr1": waveforms["Vr1"], "Ir2": waveforms["Ir2"], "Vr2": waveforms["Vr2"], "Ima": waveforms["Ima"]})
