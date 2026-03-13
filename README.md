# CLLC Time-Domain Mode Solver

A Python implementation of the CLLC time-domain equation solver for **automatic operating-mode identification**, **numerical solution**, and **waveform generation** from a given operating point.

This project is migrated and refactored from the original Mathematica notebook workflow into a modular Python package, enabling reusable stage models, mode solvers, legality checking, and automated plotting.

---

## Overview

Given an operating point specified by:

- normalized switching frequency `F`
- tank parameter `k`
- normalized power `P`

the solver automatically:

- constructs the corresponding time-domain stage equations
- solves candidate operating modes numerically
- evaluates mode legality conditions over the full stage intervals
- identifies the physically valid operating mode
- stitches the stage waveforms into one complete half-cycle / full-cycle representation
- exports the waveform plot

This provides a unified workflow for **CLLC mode discrimination and waveform visualization** without manually testing individual modes one by one.

---

## Features

- Automatic mode identification from input `(F, k, P)`
- Numerical solution of mode-dependent nonlinear equation systems
- Stage-wise legality checking over continuous time intervals
- Waveform stitching for multi-stage operating modes
- Unified waveform plotting and export
- Iterative warm-start sweep for robust `(F, P) -> M` distribution generation
- CSV export of `(F, P, M)` sweep data

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Single operating point:

```bash
python main.py --F 1.0 --k 1.0 --P 0.2 --out waveform.png
```

Sweep a grid and generate `(F, P, M)` distribution:

```bash
python main.py --sweep --k 4.0 --f_min 0.5 --f_max 1.5 --f_num 10 --p_min 0.1 --p_max 0.8 --p_num 10
```

This produces:

- `fp_m_distribution.png` (3D `(F, P, M)` distribution)
- `fp_m_distribution_by_mode.png` (same distribution separated by mode)
- `fp_m_distribution.csv` (tabular data)

To disable iterative warm-start in sweep mode:

```bash
python main.py --sweep --k 1.0 --no_iterative
```

## Files

- `cllc_modes/stages.py` Stage equations converted from the original Mathematica derivation.
- `cllc_modes/mode_*.py` Individual operating-mode solvers, including equation construction and legality checks.
- `cllc_modes/mode_selector.py` Candidate-mode evaluation and reusable operating-point solver.
- `cllc_modes/sweep.py` `(F, P)` sweep logic and 3D distribution plotting.
- `cllc_modes/plotting.py` Waveform stitching and plotting utilities.
- `main.py` Command-line entry point for solving, mode identification, and sweep export.