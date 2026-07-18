"""Optimisation utilities for the Phase-2 Constrained-MDP core.

Currently exposes the PID-Lagrangian dual update used by ``train_cmdp.py``.
"""

from .pid_lagrangian import PIDLagrangian

__all__ = ["PIDLagrangian"]
