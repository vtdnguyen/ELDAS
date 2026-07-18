"""QoS coefficient maps — an exact mirror of the Java
``SimulationConfig.qosToWeight`` (κ, G1.0) and ``qosToSlackFactor``.

Keeping these in one small module (rather than duplicating literals across the
eval code) makes the "single definition of C_SLA" invariant (CLAUDE.md Lưu ý
#5) auditable: the static baseline uses the *same* κ and deadline slack as the
CloudSim DES, so the two SLA-cost numbers are in the same units.
"""

from __future__ import annotations

# QoS → SLA penalty multiplier κ (higher ⇒ heavier deadline-miss penalty).
_QOS_WEIGHT = {
    "LS": 3.0,          # Latency-Sensitive
    "Guaranteed": 2.0,
    "Burstable": 1.0,
    "BE": 0.5,          # Best-Effort
}
_DEFAULT_WEIGHT = 1.0

# QoS → deadline slack factor: deadline = creation + duration × slack(qos).
_QOS_SLACK = {
    "LS": 1.1,          # tight — 10 % slack
    "Guaranteed": 1.3,
    "Burstable": 1.7,
    "BE": 3.0,          # essentially uncapped
}
_DEFAULT_SLACK = 1.5


def qos_weight(qos: str) -> float:
    """κ — SLA penalty multiplier for a QoS class (mirror of Java)."""
    return _QOS_WEIGHT.get(qos, _DEFAULT_WEIGHT)


def qos_slack_factor(qos: str) -> float:
    """Deadline slack factor for a QoS class (mirror of Java)."""
    return _QOS_SLACK.get(qos, _DEFAULT_SLACK)
