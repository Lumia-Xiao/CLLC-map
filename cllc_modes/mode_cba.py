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
class CBAParams:
    M: float
    theta_c: float
    theta_b: float
    theta_a: float
    C1C: float
    C2C: float
    C3C: float
    C4C: float
    C1B: float
    C2B: float
    V2B0: float
    C1A: float
    C2A: float
    C3A: float
    C4A: float

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class CBASolver(ModeSolverBase):
    mode_name = "CBA"

    def __init__(self) -> None:
        self.initial_guess = np.array([1.16, 3.20, 1.98, 0.11, -0.15, 0.01, -2.18, 1.50, 0.15, -0.15, 1.01, 0.15, 0.01, 0.16, -0.15], dtype=float)

    @staticmethod
    def _unpack(x: np.ndarray) -> CBAParams:
        return CBAParams(*map(float, x.tolist()))

    @staticmethod
    def _build_stages(p: CBAParams, k: float) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        stage_c = make_stage4(p.C1C, p.C2C, p.C3C, p.C4C, 2.0, -2.0 * p.M, 1.0, p.M, k)
        stage_b = make_stage2(p.C1B, p.C2B, 1.0, p.V2B0, 1.0, 1.0, k)
        stage_a = make_stage4(p.C1A, p.C2A, p.C3A, p.C4A, 2.0, 2.0 * p.M, 1.0, -p.M, k)
        return stage_c, stage_b, stage_a

    def _equations(self, x: np.ndarray, op: OperatingPoint) -> np.ndarray:
        p = self._unpack(x)
        stage_c, stage_b, stage_a = self._build_stages(p, op.k)
        I1C, I2C, V1C, V2C = stage_c["I1"], stage_c["I2"], stage_c["V1"], stage_c["V2"]
        I1B, I2B, V1B, V2B, VoB = stage_b["I1"], stage_b["I2"], stage_b["V1"], stage_b["V2"], stage_b["Vo"]
        I1A, I2A, V1A, V2A = stage_a["I1"], stage_a["I2"], stage_a["V1"], stage_a["V2"]

        power_c = quad(lambda th: float(I1C(th)), 0.0, p.theta_c, limit=200)[0]
        power_b = quad(lambda th: float(I1B(th)), 0.0, p.theta_b, limit=200)[0]
        power_a = quad(lambda th: float(I1A(th)), 0.0, p.theta_a, limit=200)[0]
        half_period = math.pi / op.F
        return np.array([
            I1C(p.theta_c) - I1B(0.0),
            I2C(p.theta_c) - I2B(0.0),
            V1C(p.theta_c) - V1B(0.0),
            V2C(p.theta_c) - V2B(0.0),
            I1B(p.theta_b) - I1A(0.0),
            I2B(p.theta_b) - I2A(0.0),
            V1B(p.theta_b) - V1A(0.0),
            V2B(p.theta_b) - V2A(0.0),
            I1A(p.theta_a) + I1C(0.0),
            I2A(p.theta_a) + I2C(0.0),
            V1A(p.theta_a) + V1C(0.0),
            V2A(p.theta_a) + V2C(0.0),
            p.theta_c + p.theta_b + p.theta_a - half_period,
            VoB(p.theta_b) + p.M,
            (op.F / math.pi) * (power_c + power_b + power_a) - op.P,
        ], dtype=float)

    @staticmethod
    def _make_join(theta_c: float, theta_b: float, half_period: float, fc, fb, fa):
        def joined(theta: np.ndarray | float):
            t = np.asarray(theta, dtype=float)
            out = np.empty_like(t)
            m1 = (t >= -half_period) & (t <= theta_c - half_period)
            m2 = (t > theta_c - half_period) & (t <= theta_c + theta_b - half_period)
            m3 = (t > theta_c + theta_b - half_period) & (t <= 0.0)
            m4 = (t > 0.0) & (t <= theta_c)
            m5 = (t > theta_c) & (t <= theta_c + theta_b)
            m6 = (t > theta_c + theta_b) & (t <= half_period)
            out[m1] = -np.asarray(fc(t[m1] + half_period), dtype=float)
            out[m2] = -np.asarray(fb(t[m2] - theta_c + half_period), dtype=float)
            out[m3] = -np.asarray(fa(t[m3] - (theta_c + theta_b) + half_period), dtype=float)
            out[m4] = np.asarray(fc(t[m4]), dtype=float)
            out[m5] = np.asarray(fb(t[m5] - theta_c), dtype=float)
            out[m6] = np.asarray(fa(t[m6] - (theta_c + theta_b)), dtype=float)
            if np.isscalar(theta):
                return float(out.reshape(-1)[0])
            return out

        return joined

    def _build_waveforms(self, p: CBAParams, op: OperatingPoint) -> tuple[dict[str, object], float]:
        stage_c, stage_b, stage_a = self._build_stages(p, op.k)
        half_period = math.pi / op.F
        join = lambda fc, fb, fa: self._make_join(p.theta_c, p.theta_b, half_period, fc, fb, fa)
        return {
            "Vin": join(stage_c["Vin"], stage_b["Vin"], stage_a["Vin"]),
            "Vo": join(stage_c["Vo"], stage_b["Vo"], stage_a["Vo"]),
            "Ir1": join(stage_c["I1"], stage_b["I1"], stage_a["I1"]),
            "Vr1": join(stage_c["V1"], stage_b["V1"], stage_a["V1"]),
            "Ir2": join(stage_c["I2"], stage_b["I2"], stage_a["I2"]),
            "Vr2": join(stage_c["V2"], stage_b["V2"], stage_a["V2"]),
            "Ima": join(stage_c["Ima"], stage_b["Ima"], stage_a["Ima"]),
            "stage_c": stage_c,
            "stage_b": stage_b,
            "stage_a": stage_a,
        }, half_period

    def solve(self, op: OperatingPoint) -> SolveResult:
        sol = root(fun=lambda x: self._equations(x, op), x0=self.initial_guess, method=config.ROOT_METHOD, options={"maxfev": config.MAX_FEVAL, "xtol": config.SOLVER_TOL})
        p = self._unpack(sol.x)
        residual = self._equations(sol.x, op)
        max_residual = float(np.max(np.abs(residual)))
        waveforms, half_period = self._build_waveforms(p, op)
        stage_c, stage_b, stage_a = waveforms["stage_c"], waveforms["stage_b"], waveforms["stage_a"]
        checks = {
            "Stage C": check_positive_over_interval(lambda th: np.asarray(stage_c["I1"](th), dtype=float) - np.asarray(stage_c["Ima"](th), dtype=float), 0.0, p.theta_c, tol=config.CHECK_TOL, n=config.DENSE_CHECK_POINTS),
            "Stage B": check_bounded_over_interval(stage_b["Vo"], -p.M, p.M, 0.0, p.theta_b, tol=config.CHECK_TOL, n=config.DENSE_CHECK_POINTS),
            "Stage A": check_positive_over_interval(lambda th: np.asarray(stage_a["Ima"](th), dtype=float) - np.asarray(stage_a["I1"](th), dtype=float), 0.0, p.theta_a, tol=config.CHECK_TOL, n=config.DENSE_CHECK_POINTS),
            "theta_c_positive": {"passed": p.theta_c > 0.0, "value": p.theta_c},
            "theta_b_positive": {"passed": p.theta_b > 0.0, "value": p.theta_b},
            "theta_a_positive": {"passed": p.theta_a > 0.0, "value": p.theta_a},
            "M_positive": {"passed": p.M > 0.0, "value": p.M},
        }
        all_checks_pass = all(bool(item.get("passed", False)) for item in checks.values())
        success = bool(sol.success) and (max_residual < config.SOLVER_TOL) and all_checks_pass
        return SolveResult(mode=self.mode_name, success=success, operating_point=op, params=p.as_dict(), max_residual=max_residual, residual_vector=[float(v) for v in residual.tolist()], checks=checks, message=sol.message, waveforms={"half_period": half_period, "Vin": waveforms["Vin"], "Vo": waveforms["Vo"], "Ir1": waveforms["Ir1"], "Vr1": waveforms["Vr1"], "Ir2": waveforms["Ir2"], "Vr2": waveforms["Vr2"], "Ima": waveforms["Ima"]})
