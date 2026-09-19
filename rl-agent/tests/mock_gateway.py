"""Mock Py4J gateway for testing without a running Java process.

Simulates the GatewayEntryPoint / SimulationManager behaviour:
  - reset() → StepResult with initial observation
  - step(hostIndex) → StepResult with reward, advances task index
  - getActionMask() → boolean array
  - selectBaselineAction("k8s"|"random") → int

The mock uses deterministic values so tests are reproducible.
"""

from __future__ import annotations

import numpy as np


# Observation layout MUST mirror Java SimulationManager.buildObservation():
#   [0..H-1]    CPU util      [H..2H-1]   MEM util     [2H..3H-1]  GPU util
#   [3H..6H-1]  host state one-hot, HOST-MAJOR 3 per host (SUSPENDED, IDLE, ACTIVE)
#   [6H..6H+3]  task features (cpu_norm, mem_norm, gpu_norm, qos_weight_norm)
# Total = 6H + 4 (T8.9 state machine; CLAUDE.md Lưu ý #14). The mock previously
# emitted the pre-P1.8 3H+4 layout, which made every env/state_builder test fail
# against the real contract (tracking debt D1).
NUM_HOSTS = 10
PER_HOST_DIM = 6
TASK_FEATURE_DIM = 4
OBS_DIM = PER_HOST_DIM * NUM_HOSTS + TASK_FEATURE_DIM
NUM_TASKS = 5  # short episodes for fast tests

# Host-state codes, matching Java SimulationManager.HostState.
STATE_SUSPENDED = 0
STATE_IDLE = 1
STATE_ACTIVE = 2


class MockStepResult:
    """Mimics Java StepResult record accessed via Py4J."""

    def __init__(
        self,
        obs: list[float],
        rew: list[float],
        done: bool,
        task_index: int,
        task_name: str,
        cost: float = 0.0,
        dropped_tasks: int = 0,
    ) -> None:
        self._obs = obs
        self._rew = rew
        self._done = done
        self._task_index = task_index
        self._task_name = task_name
        self._cost = cost
        self._dropped_tasks = dropped_tasks

    def observation(self) -> list[float]:
        return self._obs

    def reward(self) -> list[float]:
        return self._rew

    def cost(self) -> float:
        """G1.1 — per-step CMDP constraint cost C_SLA ≥ 0 (= −R_sla)."""
        return self._cost

    def done(self) -> bool:
        return self._done

    def taskIndex(self) -> int:
        return self._task_index

    def taskName(self) -> str:
        return self._task_name

    def droppedTasks(self) -> int:
        """W3.1 — episode-cumulative count of tasks no host could accept."""
        return self._dropped_tasks


class MockEntryPoint:
    """Mimics GatewayEntryPoint exposed via Py4J."""

    def __init__(self, num_hosts: int = NUM_HOSTS, num_tasks: int = NUM_TASKS,
                 drop_at: set[int] | None = None):
        self._num_hosts = num_hosts
        self._num_tasks = num_tasks
        self._task_idx = 0
        self._done = False
        self._rng = np.random.default_rng(42)
        self._total_energy_kwh = 0.0
        self._sla_cost = 0.0
        self._rr_pointer = 0   # roundrobin is stateful, as in Java
        # W3.1 — 1-based task indices this mock reports as unplaceable. Empty by
        # default so existing tests see the unchanged trajectory; a test that
        # wants the drop path sets it explicitly.
        self._drop_at = drop_at or set()
        self._dropped_tasks = 0

    def _make_obs(self) -> list[float]:
        """Build a 6H+4 observation mirroring Java's buildObservation()."""
        h = self._num_hosts
        obs = [0.0] * (PER_HOST_DIM * h + TASK_FEATURE_DIM)
        for i in range(h):
            cpu = float(self._rng.uniform(0, 0.8))
            obs[i] = cpu                                          # CPU util
            obs[h + i] = float(self._rng.uniform(0, 0.6))         # MEM util
            obs[2 * h + i] = float(self._rng.uniform(0, 0.5))     # GPU util

            # Host-state one-hot, HOST-MAJOR (3 contiguous floats per host) —
            # exactly one slot set, as Java writes obs[3h + 3i + state.code].
            # Derive the state from load so the mock stays self-consistent:
            # carrying CPU load ⇒ ACTIVE, else IDLE/SUSPENDED.
            if cpu > 0.05:
                state = STATE_ACTIVE
            else:
                state = STATE_SUSPENDED if i % 2 == 0 else STATE_IDLE
            obs[3 * h + 3 * i + state] = 1.0

        if not self._done and self._task_idx < self._num_tasks:
            obs[6 * h] = float(self._rng.uniform(0.1, 0.5))       # task cpu_norm
            obs[6 * h + 1] = float(self._rng.uniform(0.1, 0.3))   # task mem_norm
            obs[6 * h + 2] = float(self._rng.uniform(0, 0.25))    # task gpu_norm
            obs[6 * h + 3] = float(self._rng.uniform(0.1, 1.0))   # qos_weight_norm
        return obs

    def reset(self, scenario: str = "HIGH", seed: int = 42) -> MockStepResult:
        self._task_idx = 0
        self._done = False
        self._rng = np.random.default_rng(seed)
        self._total_energy_kwh = 0.0
        self._sla_cost = 0.0
        self._rr_pointer = 0   # fresh pointer per episode, as Java re-creates it
        self._dropped_tasks = 0
        return MockStepResult(
            obs=self._make_obs(),
            rew=[0.0, 0.0],
            done=False,
            task_index=0,
            task_name="task-0",
            cost=0.0,
        )

    def step(self, host_index: int) -> MockStepResult:
        energy_delta = -float(self._rng.uniform(50, 200))
        sla_penalty = -float(self._rng.uniform(0, 5))
        step_cost = -sla_penalty  # C_SLA = −R_sla ≥ 0 (G1.1)
        self._total_energy_kwh += abs(energy_delta) / 3_600_000.0
        self._sla_cost += step_cost

        self._task_idx += 1
        self._done = self._task_idx >= self._num_tasks
        if self._task_idx in self._drop_at:
            self._dropped_tasks += 1

        return MockStepResult(
            obs=self._make_obs(),
            rew=[energy_delta, sla_penalty],
            done=self._done,
            task_index=self._task_idx,
            task_name="" if self._done else f"task-{self._task_idx}",
            cost=step_cost,
            dropped_tasks=self._dropped_tasks,
        )

    def getActionMask(self) -> list[bool]:
        mask = [True] * self._num_hosts
        # Mask out last 2 hosts to test masking logic
        mask[-1] = False
        mask[-2] = False
        return mask

    def getActionSize(self) -> int:
        return self._num_hosts

    def getObservationSize(self) -> int:
        return PER_HOST_DIM * self._num_hosts + TASK_FEATURE_DIM

    def isDone(self) -> bool:
        return self._done

    def getTotalEnergyKwh(self) -> float:
        return self._total_energy_kwh

    def getSlaCost(self) -> float:
        """G1.1 — episode-cumulative CMDP constraint cost C_SLA ≥ 0."""
        return self._sla_cost

    def getDroppedTasks(self) -> int:
        """W3.1 — tasks this episode that no host could accept."""
        return self._dropped_tasks

    def getTaskCount(self) -> int:
        return self._num_tasks

    def getCurrentTaskIndex(self) -> int:
        return self._task_idx

    def selectBaselineAction(self, policy_name: str) -> int:
        """Mirror Java GatewayEntryPoint.selectBaselineAction.

        Phase 1.8 added firstfit / bestfit / roundrobin alongside the original
        k8s / random; the mock only knew the latter two, so baseline_eval's real
        5-policy list (BASELINE_POLICIES) could not be exercised in tests.
        Names must match the Java switch, case-insensitively.
        """
        mask = self.getActionMask()
        feasible = [i for i, m in enumerate(mask) if m]
        if not feasible:
            return 0
        policy = policy_name.lower()
        if policy == "k8s":
            return feasible[0]          # deterministic: pick first feasible
        if policy == "random":
            return feasible[self._rng.integers(0, len(feasible))]
        if policy == "firstfit":
            return feasible[0]          # first feasible in index order
        if policy == "bestfit":
            return feasible[-1]         # stand-in for "tightest fit"
        if policy == "roundrobin":
            idx = self._rr_pointer % len(feasible)
            self._rr_pointer += 1
            return feasible[idx]
        raise ValueError(f"Unknown policy: {policy_name}")

    def exportMetrics(self, output_dir: str) -> None:
        pass  # no-op in tests

    def shutdown(self) -> None:
        pass

    def toString(self) -> str:
        return "MockEntryPoint"


class MockGateway:
    """Mimics py4j.java_gateway.JavaGateway."""

    def __init__(self, entry_point: MockEntryPoint | None = None):
        self.entry_point = entry_point or MockEntryPoint()

    def shutdown(self) -> None:
        pass
