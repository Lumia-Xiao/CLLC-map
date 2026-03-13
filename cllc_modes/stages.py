from __future__ import annotations

import math
from typing import Callable

import numpy as np

WaveFn = Callable[[np.ndarray | float], np.ndarray | float]


def _as_output(theta: np.ndarray | float, value: np.ndarray) -> np.ndarray | float:
    if np.isscalar(theta):
        return float(np.asarray(value).reshape(-1)[0])
    return value


def make_stage4(c1: float, c2: float, c3: float, c4: float, v10: float, v20: float, vin0: float, vo0: float, k: float) -> dict[str, WaveFn]:
    k1 = 1.0 / math.sqrt(1.0 + 2.0 * k)
    ap = c1 + c2
    am = c1 - c2
    bp = c3 + c4
    bm = c4 - c3

    def i1(theta: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(theta, dtype=float)
        out = 0.5 * (ap * np.cos(t) + am * np.cos(k1 * t) - bp * np.sin(t) + k1 * bm * np.sin(k1 * t))
        return _as_output(theta, out)

    def i2(theta: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(theta, dtype=float)
        out = 0.5 * (ap * np.cos(t) - am * np.cos(k1 * t) - bp * np.sin(t) - k1 * bm * np.sin(k1 * t))
        return _as_output(theta, out)

    def v1(theta: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(theta, dtype=float)
        out = 0.5 * (v10 + bp * np.cos(t) - bm * np.cos(k1 * t) + ap * np.sin(t) + am * np.sin(k1 * t) / k1)
        return _as_output(theta, out)

    def v2(theta: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(theta, dtype=float)
        out = 0.5 * (v20 + bp * np.cos(t) + bm * np.cos(k1 * t) + ap * np.sin(t) - am * np.sin(k1 * t) / k1)
        return _as_output(theta, out)

    def ima(theta: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(theta, dtype=float)
        out = am * np.cos(k1 * t) + k1 * bm * np.sin(k1 * t)
        return _as_output(theta, out)

    def vin(theta: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(theta, dtype=float)
        out = np.full_like(t, vin0, dtype=float)
        return _as_output(theta, out)

    def vo(theta: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(theta, dtype=float)
        out = np.full_like(t, vo0, dtype=float)
        return _as_output(theta, out)

    return {"I1": i1, "I2": i2, "V1": v1, "V2": v2, "Ima": ima, "Vin": vin, "Vo": vo}


def make_stage2(c1: float, c2: float, v1_offset: float, v2_const: float, vin0: float, base_shift: float, k: float) -> dict[str, WaveFn]:
    k2 = 1.0 / math.sqrt(1.0 + k)

    def i1(theta: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(theta, dtype=float)
        out = c1 * np.cos(k2 * t) - c2 * k2 * np.sin(k2 * t)
        return _as_output(theta, out)

    def i2(theta: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(theta, dtype=float)
        out = np.zeros_like(t)
        return _as_output(theta, out)

    def v1(theta: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(theta, dtype=float)
        out = v1_offset + c2 * np.cos(k2 * t) + c1 * np.sin(k2 * t) / k2
        return _as_output(theta, out)

    def v2(theta: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(theta, dtype=float)
        out = np.full_like(t, v2_const, dtype=float)
        return _as_output(theta, out)

    def ima(theta: np.ndarray | float) -> np.ndarray | float:
        return i1(theta)

    def vin(theta: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(theta, dtype=float)
        out = np.full_like(t, vin0, dtype=float)
        return _as_output(theta, out)

    def vo(theta: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(theta, dtype=float)
        v1_now = v1(t)
        out = ((-1.0 + k2 ** 2) * (np.asarray(v1_now, dtype=float) - base_shift)) - v2_const
        return _as_output(theta, out)

    return {"I1": i1, "I2": i2, "V1": v1, "V2": v2, "Ima": ima, "Vin": vin, "Vo": vo}
