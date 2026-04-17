"""
T4.4 — Experiment tracking via Weights & Biases (WandB).

Provides a thin wrapper around WandB that:
  1. Handles offline mode gracefully when ``WANDB_API_KEY`` is not set.
  2. Logs per-step and per-episode metrics with consistent naming.
  3. Supports comparison across schedulers (Random / K8s / MORL).
  4. Saves model checkpoints to WandB Artifacts.

Usage::

    tracker = ExperimentTracker(
        project="ELDAS",
        run_name="morl-high-w08",
        config={"scenario": "HIGH", "energy_weight": 0.8},
    )
    tracker.log_step(step=1, reward_vec=r, obs=obs)
    tracker.log_episode(episode=1, info=info)
    tracker.save_model("model.zip")
    tracker.finish()
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


# ── WandB availability ─────────────────────────────────────────────────────

def _wandb_available() -> bool:
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        return False


def _should_use_wandb() -> bool:
    """Determine if WandB should be active based on API key and availability."""
    if not _wandb_available():
        return False
    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    return len(api_key) > 0


# ── Experiment tracker ─────────────────────────────────────────────────────

class ExperimentTracker:
    """Unified experiment tracker with optional WandB backend.

    Falls back to console-only logging when WandB is unavailable or
    ``WANDB_API_KEY`` is empty (common during local development).

    Parameters
    ----------
    project : str
        WandB project name.
    run_name : str or None
        Human-readable run name (e.g. ``"morl-high-w08"``).
    config : dict or None
        Hyperparameters and experiment config to log.
    tags : list[str] or None
        WandB tags for filtering runs.
    enabled : bool or None
        Force enable/disable WandB.  ``None`` = auto-detect from env.
    """

    def __init__(
        self,
        project: str = "ELDAS",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._config = config or {}
        self._run_name = run_name
        self._wandb_run = None
        self._enabled = enabled if enabled is not None else _should_use_wandb()

        if self._enabled:
            import wandb

            self._wandb_run = wandb.init(
                project=project,
                name=run_name,
                config=self._config,
                tags=tags,
                reinit=True,
            )
            print(f"[tracker] WandB run started: {self._wandb_run.url}")
        else:
            print("[tracker] WandB disabled — logging to console only")

    @property
    def enabled(self) -> bool:
        return self._enabled and self._wandb_run is not None

    # ── Per-step logging ──────────────────────────────────────────────

    def log_step(
        self,
        step: int,
        reward_vec: np.ndarray,
        scalar_reward: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log metrics for a single environment step.

        Parameters
        ----------
        step : int
            Global step counter.
        reward_vec : np.ndarray
            Shape ``(2,)`` — ``[R_energy, R_sla]``.
        scalar_reward : float or None
            Scalarised reward (if applicable).
        extra : dict or None
            Additional key-value pairs to log.
        """
        metrics: dict[str, Any] = {
            "step/r_energy": float(reward_vec[0]),
            "step/r_sla": float(reward_vec[1]),
        }
        if scalar_reward is not None:
            metrics["step/r_scalar"] = scalar_reward
        if extra:
            metrics.update(extra)

        if self.enabled:
            import wandb
            wandb.log(metrics, step=step)

    # ── Per-episode logging ───────────────────────────────────────────

    def log_episode(
        self,
        episode: int,
        info: dict[str, Any],
        scheduler: str = "morl",
    ) -> None:
        """Log end-of-episode summary.

        Parameters
        ----------
        episode : int
            Episode number.
        info : dict
            Episode info dict from ``CloudSimEnv`` (contains ``"episode"``
            sub-dict with aggregated metrics).
        scheduler : str
            Scheduler name (``"morl"``, ``"k8s"``, ``"random"``).
        """
        ep_info = info.get("episode", info)

        metrics: dict[str, Any] = {
            "episode/number": episode,
            "episode/length": ep_info.get("length", 0),
            "episode/total_energy_reward": ep_info.get("total_energy_reward", 0),
            "episode/total_sla_reward": ep_info.get("total_sla_reward", 0),
            "episode/mean_energy_reward": ep_info.get("mean_energy_reward", 0),
            "episode/mean_sla_reward": ep_info.get("mean_sla_reward", 0),
            "episode/total_energy_kwh": ep_info.get("total_energy_kwh", 0),
            "episode/scheduler": scheduler,
        }

        if self.enabled:
            import wandb
            wandb.log(metrics, step=episode)

        # Console summary
        print(
            f"[tracker] Episode {episode} ({scheduler}): "
            f"len={ep_info.get('length', '?')}, "
            f"energy_kwh={ep_info.get('total_energy_kwh', '?'):.4f}, "
            f"r_energy={ep_info.get('total_energy_reward', '?'):.2f}, "
            f"r_sla={ep_info.get('total_sla_reward', '?'):.2f}"
        )

    # ── Baseline comparison table ─────────────────────────────────────

    def log_comparison(
        self,
        results: dict[str, dict[str, float]],
    ) -> None:
        """Log a scheduler comparison table.

        Parameters
        ----------
        results : dict
            ``{"random": {"energy_kwh": ..., "sla_violations": ...}, ...}``
        """
        if self.enabled:
            import wandb

            columns = ["scheduler"] + sorted(
                {k for v in results.values() for k in v}
            )
            table = wandb.Table(columns=columns)
            for scheduler, metrics in results.items():
                row = [scheduler] + [metrics.get(c, None) for c in columns[1:]]
                table.add_data(*row)
            wandb.log({"comparison": table})

        # Console
        print("[tracker] === Scheduler Comparison ===")
        for scheduler, metrics in results.items():
            metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
            print(f"  {scheduler}: {metrics_str}")

    # ── Model checkpointing ──────────────────────────────────────────

    def save_model(self, model_path: str, name: str = "rl-model") -> None:
        """Upload a model checkpoint to WandB Artifacts.

        Parameters
        ----------
        model_path : str
            Local path to the saved model (e.g. ``model.zip``).
        name : str
            Artifact name.
        """
        if self.enabled:
            import wandb

            artifact = wandb.Artifact(name, type="model")
            artifact.add_file(model_path)
            self._wandb_run.log_artifact(artifact)  # type: ignore[union-attr]
            print(f"[tracker] Model artifact '{name}' uploaded")
        else:
            print(f"[tracker] Model saved locally: {model_path} (WandB disabled)")

    # ── Lifecycle ─────────────────────────────────────────────────────

    def finish(self) -> None:
        """End the WandB run."""
        if self.enabled:
            import wandb
            wandb.finish()
            print("[tracker] WandB run finished")
