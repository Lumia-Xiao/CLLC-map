from __future__ import annotations

import numpy as np


def check_positive_over_interval(func, left: float, right: float, tol: float = 1e-8, n: int = 2000) -> dict[str, float | bool]:
    xs = np.linspace(left, right, n)
    vals = np.asarray(func(xs), dtype=float)
    idx = int(np.argmin(vals))
    min_val = float(vals[idx])
    return {
        "type": "positive",
        "passed": bool(min_val >= -tol),
        "min_value": min_val,
        "argmin": float(xs[idx]),
        "tolerance": tol,
        "effective_margin": min_val + tol,
    }


def check_bounded_over_interval(func, lower: float, upper: float, left: float, right: float, tol: float = 1e-8, n: int = 2000) -> dict[str, float | bool]:
    xs = np.linspace(left, right, n)
    vals = np.asarray(func(xs), dtype=float)
    lower_margin = vals - lower
    upper_margin = upper - vals
    i_low = int(np.argmin(lower_margin))
    i_up = int(np.argmin(upper_margin))
    min_lower = float(lower_margin[i_low])
    min_upper = float(upper_margin[i_up])
    passed = (min_lower >= -tol) and (min_upper >= -tol)
    return {
        "type": "bounded",
        "passed": bool(passed),
        "min_lower_margin": min_lower,
        "argmin_lower": float(xs[i_low]),
        "min_upper_margin": min_upper,
        "argmin_upper": float(xs[i_up]),
        "tolerance": tol,
    }
