from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .mode_selector import OperatingPointSolver


def run_fp_sweep(
    k: float,
    f_min: float,
    f_max: float,
    f_num: int,
    p_min: float,
    p_max: float,
    p_num: int,
    residual_tol: float = 1e-6,
    iterative: bool = True,
):
    """
    Sweep F and P while keeping k fixed.
    For each (F, P), solve the operating point and record M if successful.

    If iterative=True, solver initial guesses are warm-started from previous
    converged points to improve robustness and speed on dense sweeps.
    """

    f_values = np.linspace(f_min, f_max, f_num)
    p_values = np.linspace(p_min, p_max, p_num)

    records: List[dict] = []

    total = len(f_values) * len(p_values)
    count = 0

    solver = OperatingPointSolver() if iterative else None

    for f_idx, F in enumerate(f_values):
        p_iter = p_values if (f_idx % 2 == 0) else p_values[::-1]

        for P in p_iter:
            count += 1
            try:
                if solver is not None:
                    result = solver.solve(F=F, k=k, P=P)
                else:
                    result = OperatingPointSolver().solve(F=F, k=k, P=P)

                if result.success and result.max_residual is not None and result.max_residual < residual_tol:
                    M = result.params.get("M", np.nan)
                    records.append(
                        {
                            "F": float(F),
                            "P": float(P),
                            "M": float(M),
                            "mode": result.mode,
                            "success": True,
                            "max_residual": float(result.max_residual),
                        }
                    )
                else:
                    records.append(
                        {
                            "F": float(F),
                            "P": float(P),
                            "M": np.nan,
                            "mode": result.mode if result.mode is not None else "None",
                            "success": False,
                            "max_residual": float(result.max_residual) if result.max_residual is not None else np.nan,
                        }
                    )
            except Exception:
                records.append(
                    {
                        "F": float(F),
                        "P": float(P),
                        "M": np.nan,
                        "mode": "Error",
                        "success": False,
                        "max_residual": np.nan,
                    }
                )

            if count % 50 == 0 or count == total:
                print(f"Progress: {count}/{total}")

    records.sort(key=lambda r: (r["F"], r["P"]))
    return records


def plot_fp_m_surface(records, title: Optional[str] = None, out_path: Optional[str] = None):
    """
    Plot successful (F, P, M) points as a 3D scatter.
    """
    valid = [r for r in records if r["success"] and np.isfinite(r["M"])]

    if len(valid) == 0:
        raise ValueError("No valid sweep points to plot.")

    F = np.array([r["F"] for r in valid], dtype=float)
    P = np.array([r["P"] for r in valid], dtype=float)
    M = np.array([r["M"] for r in valid], dtype=float)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(F, P, M, c=M, s=18)

    ax.set_xlabel("F")
    ax.set_ylabel("P")
    ax.set_zlabel("M")

    if title is None:
        title = "Distribution of (F, P, M)"
    ax.set_title(title)

    fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.08, label="M")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {out_path}")

    plt.show()


def plot_fp_m_by_mode(records, title: Optional[str] = None, out_path: Optional[str] = None):
    """
    Plot successful (F, P, M) points in 3D scatter, colored by mode.
    """
    valid = [r for r in records if r["success"] and np.isfinite(r["M"])]

    if len(valid) == 0:
        raise ValueError("No valid sweep points to plot.")

    mode_order = ["AC", "BCB", "CBA", "CA"]
    markers = {
        "AC": "o",
        "BCB": "^",
        "CBA": "s",
        "CA": "d",
    }

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    for mode in mode_order:
        pts = [r for r in valid if r["mode"] == mode]
        if not pts:
            continue

        F = np.array([r["F"] for r in pts], dtype=float)
        P = np.array([r["P"] for r in pts], dtype=float)
        M = np.array([r["M"] for r in pts], dtype=float)

        ax.scatter(F, P, M, s=20, marker=markers.get(mode, "o"), label=mode)

    ax.set_xlabel("F")
    ax.set_ylabel("P")
    ax.set_zlabel("M")

    if title is None:
        title = "Distribution of (F, P, M) by Mode"
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {out_path}")

    plt.show()

def plot_fp_m_density(records, title: Optional[str] = None, out_path: Optional[str] = None):
    """
    Plot a 2D density-style map on the (F, P) plane with M encoded by color.
    """
    valid = [r for r in records if r["success"] and np.isfinite(r["M"])]

    if len(valid) == 0:
        raise ValueError("No valid sweep points to plot.")

    F = np.array([r["F"] for r in valid], dtype=float)
    P = np.array([r["P"] for r in valid], dtype=float)
    M = np.array([r["M"] for r in valid], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    if len(valid) >= 3:
        triangulation = ax.tricontourf(F, P, M, levels=24, cmap="viridis")
        color_source = triangulation
    else:
        # Fallback when too few points are available for triangulation.
        color_source = ax.scatter(F, P, c=M, cmap="viridis", s=30, edgecolors="none", alpha=0.9)

    ax.scatter(F, P, c=M, cmap="viridis", s=12, edgecolors="none", alpha=0.85)

    ax.set_xlabel("F")
    ax.set_ylabel("P")
    if title is None:
        title = "2D density map of M over (F, P)"
    ax.set_title(title)

    cbar = fig.colorbar(color_source, ax=ax)
    cbar.set_label("M")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {out_path}")

    plt.show()