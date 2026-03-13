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
- Modular code structure for future extension to additional modes

---

## Current Implementation

The current solver framework supports:

- automated candidate mode evaluation
- legality verification for each solved mode
- automatic mode selection
- waveform generation and plotting

The codebase is organized so that each operating mode can be implemented as an independent solver module while sharing common stage equations, validation rules, and plotting utilities.

---

## Installation

```bash
pip install -r requirements.txt
```

## Files

- cllc_modes/stages.py Stage equations converted from the original Mathematica derivation.

- cllc_modes/mode_*.py Individual operating-mode solvers, including equation construction and legality checks.

- cllc_modes/mode_selector.py Candidate-mode evaluation and final automatic mode selection.

- cllc_modes/plotting.py Waveform stitching and plotting utilities.

- main.py Command-line entry point for solving, mode identification, and waveform export.

