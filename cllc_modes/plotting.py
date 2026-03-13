from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import config
from .types import SolveResult


DEFAULT_ORDER = ["Ir1", "Vr1", "Ir2", "Vr2", "Ima", "Vin", "Vo"]


def plot_mode_result(result: SolveResult, save_path: str | Path | None = None, names: list[str] | None = None) -> Path | None:
    if not result.waveforms:
        raise ValueError("No waveform data available in SolveResult.")

    half_period = float(result.waveforms["half_period"])
    theta = np.linspace(-half_period, half_period, config.PLOT_POINTS)
    names = names or DEFAULT_ORDER

    fig, ax = plt.subplots(figsize=(12, 6))
    for name in names:
        fn = result.waveforms[name]
        ax.plot(theta, fn(theta), label=name, linewidth=2.0)

    ax.set_title(f"Mode = {result.mode} | F={result.operating_point.F}, k={result.operating_point.k}, P={result.operating_point.P}")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Normalized waveform")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(ncol=4)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return None
