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


NUM_HOSTS = 10
OBS_DIM = 3 * NUM_HOSTS + 4
NUM_TASKS = 5  # short episodes for fast tests


class MockStepResult:
    """Mimics Java StepResult record accessed via Py4J."""

    def __init__(
        self,
        obs: list[float],
        rew: list[float],
        done: bool,
        task_index: int,
        task_name: str,
    ) -> None:
        self._obs = obs
        self._rew = rew
        self._done = done
        self._task_index = task_index
        self._task_name = task_name

    def observation(self) -> list[float]:
        return self._obs

    def reward(self) -> list[float]:
        return self._rew

    def done(self) -> bool:
        return self._done

    def taskIndex(self) -> int:
        return self._task_index

    def taskName(self) -> str:
        return self._task_name


class MockEntryPoint:
    """Mimics GatewayEntryPoint exposed via Py4J."""

    def __init__(self, num_hosts: int = NUM_HOSTS, num_tasks: int = NUM_TASKS):
        self._num_hosts = num_hosts
        self._num_tasks = num_tasks
        self._task_idx = 0
        self._done = False
        self._rng = np.random.default_rng(42)
        self._total_energy_kwh = 0.0

    def _make_obs(self) -> list[float]:
        obs = [0.0] * (3 * self._num_hosts + 4)
        for i in range(self._num_hosts):
            obs[i] = float(self._rng.uniform(0, 0.8))                  # CPU
            obs[self._num_hosts + i] = float(self._rng.uniform(0, 0.6))  # MEM
            obs[2 * self._num_hosts + i] = float(self._rng.uniform(0, 0.5))  # GPU
        if not self._done and self._task_idx < self._num_tasks:
            obs[3 * self._num_hosts] = float(self._rng.uniform(0.1, 0.5))
            obs[3 * self._num_hosts + 1] = float(self._rng.uniform(0.1, 0.3))
            obs[3 * self._num_hosts + 2] = float(self._rng.uniform(0, 0.25))
            obs[3 * self._num_hosts + 3] = float(self._rng.uniform(0.1, 1.0))
        return obs

    def reset(self, scenario: str = "HIGH", seed: int = 42) -> MockStepResult:
        self._task_idx = 0
        self._done = False
        self._rng = np.random.default_rng(seed)
        self._total_energy_kwh = 0.0
        return MockStepResult(
            obs=self._make_obs(),
            rew=[0.0, 0.0],
            done=False,
            task_index=0,
            task_name="task-0",
        )

    def step(self, host_index: int) -> MockStepResult:
        energy_delta = -float(self._rng.uniform(50, 200))
        sla_penalty = -float(self._rng.uniform(0, 5))
        self._total_energy_kwh += abs(energy_delta) / 3_600_000.0

        self._task_idx += 1
        self._done = self._task_idx >= self._num_tasks

        return MockStepResult(
            obs=self._make_obs(),
            rew=[energy_delta, sla_penalty],
            done=self._done,
            task_index=self._task_idx,
            task_name="" if self._done else f"task-{self._task_idx}",
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
        return 3 * self._num_hosts + 4

    def isDone(self) -> bool:
        return self._done

    def getTotalEnergyKwh(self) -> float:
        return self._total_energy_kwh

    def getTaskCount(self) -> int:
        return self._num_tasks

    def getCurrentTaskIndex(self) -> int:
        return self._task_idx

    def selectBaselineAction(self, policy_name: str) -> int:
        mask = self.getActionMask()
        feasible = [i for i, m in enumerate(mask) if m]
        if policy_name == "k8s":
            return feasible[0]  # deterministic: pick first feasible
        elif policy_name == "random":
            return feasible[self._rng.integers(0, len(feasible))]
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
