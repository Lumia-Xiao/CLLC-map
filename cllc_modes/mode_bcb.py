from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root

from . import config
from .checks import check_bounded_over_interval, check_positive_over_interval
from .solver_base import ModeSolverBase
from .stages import make_stage2, make_stage4
from .types import OperatingPoint, SolveResult


@dataclass(slots=True)
class BCBParams:
    M: float
    theta_b1: float
    theta_c: float
    theta_b2: float
    C1B1: float
    C2B1: float
    V2B01: float
    C1C: float
    C2C: float
    C3C: float
    C4C: float
    C1B2: float
    C2B2: float
    V2B02: float

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class BCBSolver(ModeSolverBase):
    mode_name = "BCB"

    def __init__(self) -> None:
        self.initial_guess = np.array([1.02, 0.05, 3.10, 0.05, -0.44, -1.15, -0.14, -0.13, 0.01, -1.23, 0.99, 0.41, -1.06, 0.16], dtype=float)

    @staticmethod
    def _unpack(x: np.ndarray) -> BCBParams:
        return BCBParams(*map(float, x.tolist()))

    @staticmethod
    def _build_stages(p: BCBParams, k: float) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        stage_b1 = make_stage2(p.C1B1, p.C2B1, 1.0, p.V2B01, 1.0, 1.0, k)
        stage_c = make_stage4(p.C1C, p.C2C, p.C3C, p.C4C, 2.0, -2.0 * p.M, 1.0, p.M, k)
        stage_b2 = make_stage2(p.C1B2, p.C2B2, 1.0, p.V2B02, 1.0, 1.0, k)
        return stage_b1, stage_c, stage_b2

    def _equations(self, x: np.ndarray, op: OperatingPoint) -> np.ndarray:
        p = self._unpack(x)
        stage_b1, stage_c, stage_b2 = self._build_stages(p, op.k)
        I1B1, I2B1, V1B1, V2B1, VoB1 = stage_b1["I1"], stage_b1["I2"], stage_b1["V1"], stage_b1["V2"], stage_b1["Vo"]
        I1C, I2C, V1C, V2C = stage_c["I1"], stage_c["I2"], stage_c["V1"], stage_c["V2"]
        I1B2, I2B2, V1B2, V2B2 = stage_b2["I1"], stage_b2["I2"], stage_b2["V1"], stage_b2["V2"]

        power_b1 = quad(lambda th: float(I1B1(th)), 0.0, p.theta_b1, limit=200)[0]
        power_c = quad(lambda th: float(I1C(th)), 0.0, p.theta_c, limit=200)[0]
        power_b2 = quad(lambda th: float(I1B2(th)), 0.0, p.theta_b2, limit=200)[0]
        half_period = math.pi / op.F
        return np.array([
            I1B1(p.theta_b1) - I1C(0.0),
            I2B1(p.theta_b1) - I2C(0.0),
            V1B1(p.theta_b1) - V1C(0.0),
            V2B1(p.theta_b1) - V2C(0.0),
            I1C(p.theta_c) - I1B2(0.0),
            I2C(p.theta_c) - I2B2(0.0),
            V1C(p.theta_c) - V1B2(0.0),
            V2C(p.theta_c) - V2B2(0.0),
            I1B2(p.theta_b2) + I1B1(0.0),
            V1B2(p.theta_b2) + V1B1(0.0),
            V2B2(p.theta_b2) + V2B1(0.0),
            p.theta_b1 + p.theta_c + p.theta_b2 - half_period,
            VoB1(p.theta_b1) - p.M,
            (op.F / math.pi) * (power_b1 + power_c + power_b2) - op.P,
        ], dtype=float)

    @staticmethod
    def _make_join(theta_b1: float, theta_c: float, half_period: float, fb1, fc, fb2):
        def joined(theta: np.ndarray | float):
            t = np.asarray(theta, dtype=float)
            out = np.empty_like(t)
            m1 = (t >= -half_period) & (t <= theta_b1 - half_period)
            m2 = (t > theta_b1 - half_period) & (t <= theta_b1 + theta_c - half_period)
            m3 = (t > theta_b1 + theta_c - half_period) & (t <= 0.0)
            m4 = (t > 0.0) & (t <= theta_b1)
            m5 = (t > theta_b1) & (t <= theta_b1 + theta_c)
            m6 = (t > theta_b1 + theta_c) & (t <= half_period)
            out[m1] = -np.asarray(fb1(t[m1] + half_period), dtype=float)
            out[m2] = -np.asarray(fc(t[m2] - theta_b1 + half_period), dtype=float)
            out[m3] = -np.asarray(fb2(t[m3] - (theta_b1 + theta_c) + half_period), dtype=float)
            out[m4] = np.asarray(fb1(t[m4]), dtype=float)
            out[m5] = np.asarray(fc(t[m5] - theta_b1), dtype=float)
            out[m6] = np.asarray(fb2(t[m6] - (theta_b1 + theta_c)), dtype=float)
            if np.isscalar(theta):
                return float(out.reshape(-1)[0])
            return out

        return joined

    def _build_waveforms(self, p: BCBParams, op: OperatingPoint) -> tuple[dict[str, object], float]:
        stage_b1, stage_c, stage_b2 = self._build_stages(p, op.k)
        half_period = math.pi / op.F
        join = lambda fb1, fc, fb2: self._make_join(p.theta_b1, p.theta_c, half_period, fb1, fc, fb2)
        return {
            "Vin": join(stage_b1["Vin"], stage_c["Vin"], stage_b2["Vin"]),
            "Vo": join(stage_b1["Vo"], stage_c["Vo"], stage_b2["Vo"]),
            "Ir1": join(stage_b1["I1"], stage_c["I1"], stage_b2["I1"]),
            "Vr1": join(stage_b1["V1"], stage_c["V1"], stage_b2["V1"]),
            "Ir2": join(stage_b1["I2"], stage_c["I2"], stage_b2["I2"]),
            "Vr2": join(stage_b1["V2"], stage_c["V2"], stage_b2["V2"]),
            "Ima": join(stage_b1["Ima"], stage_c["Ima"], stage_b2["Ima"]),
            "stage_b1": stage_b1,
            "stage_c": stage_c,
            "stage_b2": stage_b2,
        }, half_period

    def solve(self, op: OperatingPoint) -> SolveResult:
        sol = root(fun=lambda x: self._equations(x, op), x0=self.initial_guess, method=config.ROOT_METHOD, options={"maxfev": config.MAX_FEVAL, "xtol": config.SOLVER_TOL})
        p = self._unpack(sol.x)
        residual = self._equations(sol.x, op)
        max_residual = float(np.max(np.abs(residual)))
        waveforms, half_period = self._build_waveforms(p, op)
        stage_b1, stage_c, stage_b2 = waveforms["stage_b1"], waveforms["stage_c"], waveforms["stage_b2"]
        checks = {
            "Stage B1": check_bounded_over_interval(stage_b1["Vo"], -p.M, p.M, 0.0, p.theta_b1, tol=config.CHECK_TOL, n=config.DENSE_CHECK_POINTS),
            "Stage C": check_positive_over_interval(lambda th: np.asarray(stage_c["I1"](th), dtype=float) - np.asarray(stage_c["Ima"](th), dtype=float), 0.0, p.theta_c, tol=config.CHECK_TOL, n=config.DENSE_CHECK_POINTS),
            "Stage B2": check_bounded_over_interval(stage_b2["Vo"], -p.M, p.M, 0.0, p.theta_b2, tol=config.CHECK_TOL, n=config.DENSE_CHECK_POINTS),
            "theta_b1_positive": {"passed": p.theta_b1 > 0.0, "value": p.theta_b1},
            "theta_c_positive": {"passed": p.theta_c > 0.0, "value": p.theta_c},
            "theta_b2_positive": {"passed": p.theta_b2 > 0.0, "value": p.theta_b2},
            "M_positive": {"passed": p.M > 0.0, "value": p.M},
        }
        all_checks_pass = all(bool(item.get("passed", False)) for item in checks.values())
        success = bool(sol.success) and (max_residual < config.SOLVER_TOL) and all_checks_pass
        return SolveResult(mode=self.mode_name, success=success, operating_point=op, params=p.as_dict(), max_residual=max_residual, residual_vector=[float(v) for v in residual.tolist()], checks=checks, message=sol.message, waveforms={"half_period": half_period, "Vin": waveforms["Vin"], "Vo": waveforms["Vo"], "Ir1": waveforms["Ir1"], "Vr1": waveforms["Vr1"], "Ir2": waveforms["Ir2"], "Vr2": waveforms["Vr2"], "Ima": waveforms["Ima"]})
