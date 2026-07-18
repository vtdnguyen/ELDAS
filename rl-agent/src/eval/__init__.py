"""G2.2/G2.3 — Static Pareto evaluation package.

Modules:
  - ``qos``          : QoS → penalty weight κ / deadline-slack factor (mirrors Java).
  - ``trace_loader`` : Alibaba trace CSV → task records + scenario filter.
  - ``topology``     : host specs from ``config/topology-hetero.json`` (or homogeneous).
  - ``static_model`` : deterministic static energy/SLA surrogate for a placement.
  - ``nsga2_baseline``: pymoo NSGA-II reference Pareto front (G2.3).

**Scientific caveat (CLAUDE.md Lưu ý #9).** The NSGA-II front produced here is
a *static, offline idealisation*: it optimises task→host assignment with full
foreknowledge of the whole trace and NO temporal dynamics. It is an
upper-bound / reference front, **not** a head-to-head competitor to the online
RL scheduler. The static model reuses the same units as the Java DES —
energy in kWh (lower better) and ``C_SLA = Σ κ·max(0, completion − deadline)``
(lower better) — so its points share the fixed reference point used for
hypervolume (Lưu ý #8/#15), but its absolute numbers are a surrogate, not the
DES output.
"""
