"""Deadline model — bounded slowdown plus a trace-derived absolute floor (PLAN §3.8).

    deadline = creation + duration * slack_factor(qos) + slack_floor(qos)
    slack_floor(qos) = p90(scheduling lag over the whole trace) * qos_factor(qos)

**Why the multiplicative term stays.** A deadline proportional to a job's own runtime is
the standard *bounded slowdown* SLO from the scheduling literature (Feitelson): "this job
may take at most 1.1x as long as it would on an idle cluster". Removing it, as an earlier
draft of the plan proposed, would have thrown away a citable formulation and made the
constraint depend only on a queueing delay that this simulator does not model.

**Why the additive floor is added.** Measured on the trace, the multiplicative term alone
gives 24.6 % of tasks an absolute budget below 60 s, and the median LS budget is 237 s.
For those jobs the fixed 5 s wake latency is a large share of the whole budget, so turning
a host off to save energy registers as an SLA violation regardless of load. The floor
decouples the two: energy decisions stop being punished by an artefact of job length.

**Why the floor is one global percentile times a QoS factor, not a per-class percentile.**
Per-class p90 scheduling lag is degenerate on this trace — Guaranteed has 7 samples
(p90 = 1 s) and Burstable has 98 (p90 = 1 s) — so a per-class estimate multiplied by a
per-class factor yields 2 s for Guaranteed and 5 s for Burstable against 107 s for LS,
i.e. it *inverts* the QoS ordering it was meant to express. One percentile estimated from
all 7 255 schedulable pods, scaled by the QoS factor, is both better estimated and
monotone by construction:

    LS 106 s  <  Guaranteed 212 s  <  Burstable 530 s  <  BE 2 120 s

Resulting absolute budgets: min 107 s, p10 123 s, p50 553 s, and 0 % below 60 s. The
constraint stays binding — violation rate runs 33-49 % across background-utilisation
levels — so C_SLA does not collapse to zero.

Every constant here is mirrored in ``SimulationConfig.java``; the two are checked against
each other by ``tests/test_workload_deadline.py`` and by ValidationRunner B16.
"""

from __future__ import annotations

from .schema import PHASE_PENDING, Task

# ── QoS coefficient maps (mirrors of SimulationConfig) ──────────────────────

#: Multiplicative bounded-slowdown allowance. Unchanged from Phase 1.
QOS_SLACK_FACTOR = {"LS": 1.1, "Guaranteed": 1.3, "Burstable": 1.7, "BE": 3.0}
DEFAULT_SLACK_FACTOR = 1.5

#: Multiplier applied to the global p90 scheduling lag to get the absolute floor.
#: Ordered so a stricter class tolerates less absolute lateness.
QOS_FLOOR_FACTOR = {"LS": 1.0, "Guaranteed": 2.0, "Burstable": 5.0, "BE": 20.0}
DEFAULT_FLOOR_FACTOR = 1.5

#: p90 of (scheduled_time - creation_time) over the 7 255 schedulable openb pods.
#: Measured, not chosen — see :func:`measure_scheduling_lag_p90`.
OPENB_SCHEDULING_LAG_P90_SEC = 106.0

#: Percentile used for the floor. Kept explicit so the choice is auditable.
LAG_PERCENTILE = 0.90


def qos_slack_factor(qos: str) -> float:
    return QOS_SLACK_FACTOR.get(qos, DEFAULT_SLACK_FACTOR)


def qos_floor_factor(qos: str) -> float:
    return QOS_FLOOR_FACTOR.get(qos, DEFAULT_FLOOR_FACTOR)


# ── Measurement ─────────────────────────────────────────────────────────────

def measure_scheduling_lag_p90(tasks: list[Task]) -> float:
    """p90 of the observed scheduling lag, over schedulable pods only.

    ``Pending`` pods are excluded because they were never scheduled: their
    ``scheduled_time`` is absent and parses to 0, which would contribute a large
    spurious negative lag.
    """
    lags = sorted(max(0.0, t.scheduled_time - t.creation_time)
                  for t in tasks if t.pod_phase != PHASE_PENDING)
    if not lags:
        raise ValueError("no schedulable tasks to measure scheduling lag from")
    return lags[int(LAG_PERCENTILE * (len(lags) - 1))]


# ── Deadline ────────────────────────────────────────────────────────────────

def slack_floor(qos: str, lag_p90: float = OPENB_SCHEDULING_LAG_P90_SEC) -> float:
    """Absolute lateness a class tolerates regardless of how short the job is."""
    return lag_p90 * qos_floor_factor(qos)


def slack_budget(task: Task, lag_p90: float = OPENB_SCHEDULING_LAG_P90_SEC) -> float:
    """Total absolute lateness allowed before C_SLA is charged."""
    return task.duration * (qos_slack_factor(task.qos) - 1.0) + slack_floor(task.qos, lag_p90)


def deadline(task: Task, lag_p90: float = OPENB_SCHEDULING_LAG_P90_SEC) -> float:
    """``creation + duration * slack_factor(qos) + slack_floor(qos)``."""
    return (task.creation_time
            + task.duration * qos_slack_factor(task.qos)
            + slack_floor(task.qos, lag_p90))
