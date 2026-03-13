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
class CAParams:
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
        return {
            "M": self.M,
            "theta_a": self.theta_a,
            "theta_c": self.theta_c,
            "C1C": self.C1C,
            "C2C": self.C2C,
            "C3C": self.C3C,
            "C4C": self.C4C,
            "C1A": self.C1A,
            "C2A": self.C2A,
            "C3A": self.C3A,
            "C4A": self.C4A,
        }


class CASolver(ModeSolverBase):
    mode_name = "CA"

    def __init__(self) -> None:
        self.initial_guess = np.array([0.8, 1.00, 2.51, 0.04, 0.34, -2.37, -0.23, 0.31, -0.01, 0.22, 0.39], dtype=float)

    @staticmethod
    def _unpack(x: np.ndarray) -> CAParams:
        return CAParams(*map(float, x.tolist()))

    @staticmethod
    def _build_stages(p: CAParams, k: float) -> tuple[dict[str, object], dict[str, object]]:
        stage_c = make_stage4(p.C1C, p.C2C, p.C3C, p.C4C, 2.0, -2.0 * p.M, 1.0, p.M, k)
        stage_a = make_stage4(p.C1A, p.C2A, p.C3A, p.C4A, 2.0, 2.0 * p.M, 1.0, -p.M, k)
        return stage_c, stage_a

    def _equations(self, x: np.ndarray, op: OperatingPoint) -> np.ndarray:
        p = self._unpack(x)
        stage_c, stage_a = self._build_stages(p, op.k)
        I1C, I2C, V1C, V2C, ImaC = stage_c["I1"], stage_c["I2"], stage_c["V1"], stage_c["V2"], stage_c["Ima"]
        I1A, I2A, V1A, V2A = stage_a["I1"], stage_a["I2"], stage_a["V1"], stage_a["V2"]

        power_c = quad(lambda th: float(I1C(th)), 0.0, p.theta_c, limit=200)[0]
        power_a = quad(lambda th: float(I1A(th)), 0.0, p.theta_a, limit=200)[0]

        eqs = np.array([
            I1C(p.theta_c) - I1A(0.0),
            I2C(p.theta_c) - I2A(0.0),
            V1C(p.theta_c) - V1A(0.0),
            V2C(p.theta_c) - V2A(0.0),
            I1A(p.theta_a) + I1C(0.0),
            I2A(p.theta_a) + I2C(0.0),
            V1A(p.theta_a) + V1C(0.0),
            V2A(p.theta_a) + V2C(0.0),
            I1C(p.theta_c) - ImaC(p.theta_c),
            p.theta_c + p.theta_a - math.pi / op.F,
            (op.F / math.pi) * (power_c + power_a) - op.P,
        ], dtype=float)
        return eqs

    @staticmethod
    def _make_join_ca(theta_c: float, half_period: float, fc, fa):
        def joined(theta: np.ndarray | float):
            t = np.asarray(theta, dtype=float)
            out = np.empty_like(t)
            m1 = (t >= -half_period) & (t <= theta_c - half_period)
            m2 = (t > theta_c - half_period) & (t <= 0.0)
            m3 = (t > 0.0) & (t <= theta_c)
            m4 = (t > theta_c) & (t <= half_period)
            out[m1] = -np.asarray(fc(t[m1] + half_period), dtype=float)
            out[m2] = -np.asarray(fa(t[m2] - theta_c + half_period), dtype=float)
            out[m3] = np.asarray(fc(t[m3]), dtype=float)
            out[m4] = np.asarray(fa(t[m4] - theta_c), dtype=float)
            if np.isscalar(theta):
                return float(out.reshape(-1)[0])
            return out

        return joined

    def _build_waveforms(self, p: CAParams, op: OperatingPoint) -> tuple[dict[str, object], float]:
        stage_c, stage_a = self._build_stages(p, op.k)
        half_period = math.pi / op.F
        join = lambda fc, fa: self._make_join_ca(p.theta_c, half_period, fc, fa)
        waveforms = {
            "Vin": join(stage_c["Vin"], stage_a["Vin"]),
            "Vo": join(stage_c["Vo"], stage_a["Vo"]),
            "Ir1": join(stage_c["I1"], stage_a["I1"]),
            "Vr1": join(stage_c["V1"], stage_a["V1"]),
            "Ir2": join(stage_c["I2"], stage_a["I2"]),
            "Vr2": join(stage_c["V2"], stage_a["V2"]),
            "Ima": join(stage_c["Ima"], stage_a["Ima"]),
            "stage_c": stage_c,
            "stage_a": stage_a,
        }
        return waveforms, half_period

    def solve(self, op: OperatingPoint) -> SolveResult:
        sol = root(
            fun=lambda x: self._equations(x, op),
            x0=self.initial_guess,
            method=config.ROOT_METHOD,
            options={"maxfev": config.MAX_FEVAL, "xtol": config.SOLVER_TOL},
        )

        p = self._unpack(sol.x)
        residual = self._equations(sol.x, op)
        max_residual = float(np.max(np.abs(residual)))
        waveforms, half_period = self._build_waveforms(p, op)
        stage_c = waveforms["stage_c"]
        stage_a = waveforms["stage_a"]

        checks = {
            "Stage C: Ir1 > Ima": check_positive_over_interval(
                lambda th: np.asarray(stage_c["I1"](th), dtype=float) - np.asarray(stage_c["Ima"](th), dtype=float),
                0.0,
                p.theta_c,
                tol=config.CHECK_TOL,
                n=config.DENSE_CHECK_POINTS,
            ),
            "Stage A: Ir1 < Ima": check_positive_over_interval(
                lambda th: np.asarray(stage_a["Ima"](th), dtype=float) - np.asarray(stage_a["I1"](th), dtype=float),
                0.0,
                p.theta_a,
                tol=config.CHECK_TOL,
                n=config.DENSE_CHECK_POINTS,
            ),
            "theta_a_positive": {"passed": p.theta_a > 0.0, "value": p.theta_a},
            "theta_c_positive": {"passed": p.theta_c > 0.0, "value": p.theta_c},
            "M_positive": {"passed": p.M > 0.0, "value": p.M},
        }

        all_checks_pass = all(bool(item.get("passed", False)) for item in checks.values())
        success = bool(sol.success) and (max_residual < config.SOLVER_TOL) and all_checks_pass

        return SolveResult(
            mode=self.mode_name,
            success=success,
            operating_point=op,
            params=p.as_dict(),
            max_residual=max_residual,
            residual_vector=[float(v) for v in residual.tolist()],
            checks=checks,
            message=sol.message,
            waveforms={
                "half_period": half_period,
                "Vin": waveforms["Vin"],
                "Vo": waveforms["Vo"],
                "Ir1": waveforms["Ir1"],
                "Vr1": waveforms["Vr1"],
                "Ir2": waveforms["Ir2"],
                "Vr2": waveforms["Vr2"],
                "Ima": waveforms["Ima"],
            },
        )
