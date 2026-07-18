"""
G1.6 — Prometheus exporter for the CMDP training loop.

Emits the dual-variable + constraint telemetry the thesis needs to *see* the
PID-Lagrangian converge, on the ephemeral Python exporter port (8000 by default;
the long-running Java exporter on 9091 stays the source of truth for energy /
host state — CLAUDE.md monitoring decisions).

Metrics (all Gauges):
  * ``eldas_lambda``                  — current Lagrange multiplier λ
  * ``eldas_episode_cost_sla``        — episodic constraint statistic J (= E[C_SLA], normalised)
  * ``eldas_episode_cost_sla_raw``    — same in physical weighted-tardiness units
  * ``eldas_episode_reward_energy``   — episodic energy reward R_energy
  * ``eldas_budget_d``                — the SLA budget d (constant per run)
  * ``eldas_pid_integral``            — PID integral term (the vanilla-Lagrangian component)

Degrades to a no-op when ``prometheus_client`` is missing or monitoring is
disabled, exactly like ``tracker.ExperimentTracker`` does for WandB — the RL
loop must never depend on the monitoring layer (CLAUDE.md Lưu ý #16).
"""

from __future__ import annotations

import os


def _prometheus_available() -> bool:
    try:
        import prometheus_client  # noqa: F401
        return True
    except ImportError:
        return False


def _monitoring_requested(explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    return os.environ.get("MONITORING_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on")


class CMDPMetricsExporter:
    """Optional Prometheus gauge exporter for CMDP training (G1.6).

    Parameters
    ----------
    port : int
        HTTP port for the ``/metrics`` endpoint (default 8000).
    enabled : bool or None
        Force on/off.  ``None`` (default) auto-detects from ``MONITORING_ENABLED``
        and ``prometheus_client`` availability.
    """

    def __init__(self, port: int = 8000, enabled: bool | None = None) -> None:
        self._enabled = _monitoring_requested(enabled) and _prometheus_available()
        self._gauges: dict[str, object] = {}

        if not self._enabled:
            print("[metrics_exporter] disabled (monitoring off or "
                  "prometheus_client unavailable) — no-op")
            return

        from prometheus_client import Gauge, start_http_server

        specs = {
            "eldas_lambda": "Lagrange multiplier λ for the SLA constraint",
            "eldas_episode_cost_sla": "Episodic constraint statistic J = E[C_SLA] (normalised)",
            "eldas_episode_cost_sla_raw": "Episodic SLA cost in physical weighted-tardiness units",
            "eldas_episode_reward_energy": "Episodic energy reward R_energy",
            "eldas_budget_d": "SLA budget d (constraint limit)",
            "eldas_pid_integral": "PID integral term (vanilla-Lagrangian component)",
        }
        for name, doc in specs.items():
            self._gauges[name] = Gauge(name, doc)

        start_http_server(port)
        print(f"[metrics_exporter] Prometheus exporter on 0.0.0.0:{port}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def update(
        self,
        *,
        lambda_: float,
        episode_cost_sla: float,
        episode_reward_energy: float | None = None,
        episode_cost_sla_raw: float | None = None,
        budget_d: float | None = None,
        pid_integral: float | None = None,
    ) -> None:
        """Set the current gauge values (no-op when disabled)."""
        if not self._enabled:
            return
        self._set("eldas_lambda", lambda_)
        self._set("eldas_episode_cost_sla", episode_cost_sla)
        self._set("eldas_episode_reward_energy", episode_reward_energy)
        self._set("eldas_episode_cost_sla_raw", episode_cost_sla_raw)
        self._set("eldas_budget_d", budget_d)
        self._set("eldas_pid_integral", pid_integral)

    def _set(self, name: str, value: float | None) -> None:
        if value is None:
            return
        g = self._gauges.get(name)
        if g is not None:
            g.set(float(value))
