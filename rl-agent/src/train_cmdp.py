"""
G1.3 — Constrained-MDP training: MaskablePPO (primal) + PID-Lagrangian (dual).

Trains ONE policy at a fixed SLA budget ``d`` by two-timescale primal-dual:

  * **Primal** — ``MaskablePPO`` maximises the effective reward
    ``R_energy − λ·C_SLA`` (shaped by :class:`cmdp.CMDPRewardWrapper`), updating
    every rollout (``n_steps`` env steps).
  * **Dual** — :class:`optim.pid_lagrangian.PIDLagrangian` nudges λ so that the
    measured episodic cost ``J`` tracks the budget ``d``.  ``LambdaUpdateCallback``
    runs it on a SLOWER timescale (once every ``--dual-every`` completed
    episodes) — a correctness condition of primal-dual convergence, not an
    option (CLAUDE.md Lưu ý #6).

Sweeping ``--sla-budget d`` from tight → loose traces the Pareto front (G2.5);
each ``d`` yields one converged policy = one front point.

Usage::

    # Train at one budget
    docker compose run --rm rl-agent python src/train_cmdp.py \\
        --scenario HIGH --sla-budget 0.4 --total-timesteps 100000

    # Quick smoke (few steps, verify λ moves and value_loss is sane)
    docker compose run --rm rl-agent python src/train_cmdp.py \\
        --scenario LOW --sla-budget 0.5 --total-timesteps 5000 --dual-every 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# sb3-contrib hosts MaskablePPO; SB3 itself does not (CLAUDE.md Lưu ý #3 — we
# keep native action masking and wrap PID-Lagrangian around it rather than
# adopting OmniSafe's masking-free continuous-control loop).
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from cmdp import CMDPRewardWrapper
from environment import CloudSimEnv
from optim.pid_lagrangian import PIDLagrangian
from perf.tuning import DEFAULT_TORCH_THREADS, set_torch_threads

try:
    from tracker import ExperimentTracker
except Exception:  # tracker is optional
    ExperimentTracker = None  # type: ignore[assignment]

try:
    from metrics_exporter import CMDPMetricsExporter
except Exception:  # exporter is optional
    CMDPMetricsExporter = None  # type: ignore[assignment]


# ── Two-timescale dual-update callback ─────────────────────────────────────

class LambdaUpdateCallback(BaseCallback):
    """Advance the Lagrange multiplier λ on a slower timescale than the policy.

    The policy updates every rollout (``n_steps`` steps).  This callback waits
    until ``update_every_episodes`` episodes have completed, then feeds the mean
    episodic cost ``J`` to the PID controller — guaranteeing the dual is slower
    than the primal (between two dual steps the policy has updated
    ``episode_len / n_steps × update_every_episodes`` ≫ 1 times).

    Parameters
    ----------
    pid : PIDLagrangian
        Shared controller; its ``lambda_`` is read live by ``CMDPRewardWrapper``.
    budget_d : float
        SLA budget ``d`` (same units as the episodic cost statistic ``J``).
    update_every_episodes : int
        Number of completed episodes per dual update (≥ 1).
    tracker : ExperimentTracker or None
        Optional WandB logger.
    """

    def __init__(
        self,
        pid: PIDLagrangian,
        budget_d: float,
        update_every_episodes: int = 1,
        tracker=None,
        exporter=None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._pid = pid
        self._d = float(budget_d)
        self._every = max(1, int(update_every_episodes))
        self._tracker = tracker
        self._exporter = exporter
        self._costs: list[float] = []
        self._costs_raw: list[float] = []
        self._energy: list[float] = []
        self._dual_updates = 0
        # SYS.1 — set once the λ broadcast has been confirmed to reach workers.
        self._lambda_readback_ok = False
        # Per-dual-update trajectory, used by the G1.5 convergence check.
        self.history: list[dict] = []

    def _broadcast_lambda(self, lam: float) -> None:
        """Propagate λ to every (possibly remote) CMDP wrapper.

        Failing silently here would be the worst outcome: the dual logs would
        show λ moving while the policy kept training against a stale multiplier,
        producing a converged-looking run at the wrong constraint. So a missing
        hook is reported rather than swallowed.
        """
        env = self.training_env
        if env is None or not hasattr(env, "env_method"):
            return  # raw (non-vectorised) env: the wrapper reads the pid live.
        try:
            env.env_method("set_lambda", float(lam))
        except Exception as e:  # pragma: no cover - depends on wrapper stack
            raise RuntimeError(
                f"Could not broadcast λ to the vectorised envs ({e}). The CMDP "
                f"wrapper must be reachable as `set_lambda` on the outermost env "
                f"— build envs with vec_env.make_cmdp_vec_env()."
            ) from e

        # Read λ back from the workers ONCE, on the first dual update that moves
        # it off zero. env_method could dispatch to the wrong object and silently
        # do nothing, which is indistinguishable from a healthy run in the logs:
        # λ would climb here while every worker trained against λ=0, i.e. an
        # unconstrained policy reported as a converged CMDP one. Verified once,
        # not per update, so the check costs nothing in steady state.
        if not self._lambda_readback_ok and lam > 0:
            try:
                seen = env.env_method("get_lambda")
            except Exception:
                seen = None
            if seen is not None and not all(
                    abs(float(s) - float(lam)) < 1e-12 for s in seen):
                raise RuntimeError(
                    f"λ broadcast did not land: pushed {lam!r} but the workers "
                    f"report {seen!r}. The policy would train against a stale λ "
                    f"while the dual logs showed it moving."
                )
            self._lambda_readback_ok = True
            if self.verbose and seen is not None:
                print(f"[train_cmdp] SYS.1: λ broadcast verified across "
                      f"{len(seen)} env(s) (λ={lam:.6g})")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode_cost" in info:
                self._costs.append(float(info["episode_cost"]))
                self._costs_raw.append(float(info.get("episode_cost_raw", np.nan)))
                self._energy.append(float(info.get("episode_reward_energy", np.nan)))
                if len(self._costs) >= self._every:
                    self._dual_update()
        return True

    def _dual_update(self) -> None:
        j = float(np.mean(self._costs))
        lam = self._pid.update(j, self._d)

        # SYS.1 — under SubprocVecEnv the workers hold pickled copies of the env
        # stack and cannot see this controller's memory, so λ must be pushed to
        # them explicitly. In-process (DummyVecEnv / raw env) the wrapper may
        # read the pid live, but broadcasting is harmless and keeps one path.
        self._broadcast_lambda(lam)
        j_raw = float(np.nanmean(self._costs_raw)) if self._costs_raw else float("nan")
        e_mean = float(np.nanmean(self._energy)) if self._energy else float("nan")
        self._dual_updates += 1

        self.history.append({
            "update": self._dual_updates,
            "timesteps": int(self.num_timesteps),
            "lambda": lam,
            "J": j,
            "J_raw": j_raw,
            "energy": e_mean,
            "integral": self._pid.integral,
            "delta": j - self._d,
        })

        # SB3 logger (shows up in verbose output / any attached logger).
        self.logger.record("cmdp/lambda", lam)
        self.logger.record("cmdp/episode_cost_sla", j)
        self.logger.record("cmdp/episode_cost_sla_raw", j_raw)
        self.logger.record("cmdp/budget_d", self._d)
        self.logger.record("cmdp/pid_integral", self._pid.integral)

        if self._tracker is not None and getattr(self._tracker, "enabled", False):
            import wandb
            wandb.log({
                "eldas_lambda": lam,
                "eldas_episode_cost_sla": j,
                "eldas_episode_cost_sla_raw": j_raw,
                "eldas_episode_reward_energy": e_mean,
                "eldas_budget_d": self._d,
            })

        if self._exporter is not None:
            self._exporter.update(
                lambda_=lam, episode_cost_sla=j, episode_cost_sla_raw=j_raw,
                episode_reward_energy=e_mean, budget_d=self._d,
                pid_integral=self._pid.integral,
            )

        if self.verbose:
            print(f"[train_cmdp] dual #{self._dual_updates}: "
                  f"J={j:.4f} (raw={j_raw:.3e}) d={self._d:.4f} → λ={lam:.6g} "
                  f"(I={self._pid.integral:.6g})")

        self._costs.clear()
        self._costs_raw.clear()
        self._energy.clear()


# ── Convergence diagnostics (G1.5) ─────────────────────────────────────────

def convergence_summary(history: list[dict], budget_d: float,
                        tail_frac: float = 0.3) -> dict:
    """Quantify PID-Lagrangian convergence from a dual-update trajectory.

    Checks the empirical signatures of the CMDP KKT point (Paternain 2019):
      * **λ stability** — std of λ over the last ``tail_frac`` of updates,
        normalised by its mean, should be small (λ has settled, not oscillating).
      * **constraint tracking** — if the constraint is active (λ>0), the mean
        tail cost J should sit near the budget d (complementary slackness
        ``λ·(J−d)≈0``); if J<d the constraint is slack and λ should be ~0.
    Returns a dict of diagnostics (no assertion — the caller decides).
    """
    if not history:
        return {"n_updates": 0}
    lam = np.array([h["lambda"] for h in history], dtype=float)
    j = np.array([h["J"] for h in history], dtype=float)
    k = max(1, int(len(history) * tail_frac))
    lam_tail, j_tail = lam[-k:], j[-k:]
    lam_mean = float(lam_tail.mean())
    lam_std = float(lam_tail.std())
    j_mean = float(j_tail.mean())
    return {
        "n_updates": len(history),
        "lambda_final": float(lam[-1]),
        "lambda_tail_mean": lam_mean,
        "lambda_tail_std": lam_std,
        "lambda_rel_std": (lam_std / lam_mean) if lam_mean > 1e-9 else 0.0,
        "J_tail_mean": j_mean,
        "budget_d": float(budget_d),
        "constraint_gap": j_mean - float(budget_d),       # J − d
        "constraint_active": lam_mean > 1e-6,
    }


def print_convergence_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("  G1.5 — PID-Lagrangian convergence summary")
    print("=" * 60)
    if summary.get("n_updates", 0) == 0:
        print("  (no dual updates — episode too short for the run length)")
        return
    print(f"  dual updates       : {summary['n_updates']}")
    print(f"  λ final            : {summary['lambda_final']:.6g}")
    print(f"  λ tail mean±std    : {summary['lambda_tail_mean']:.6g} "
          f"± {summary['lambda_tail_std']:.6g} "
          f"(rel std {summary['lambda_rel_std']:.3f})")
    print(f"  J tail mean        : {summary['J_tail_mean']:.6f}")
    print(f"  budget d           : {summary['budget_d']:.6f}")
    print(f"  constraint gap J−d : {summary['constraint_gap']:+.6f}")
    active = summary["constraint_active"]
    print(f"  constraint         : {'ACTIVE (λ>0)' if active else 'SLACK (λ≈0)'}")
    # Complementary slackness reading.
    if active:
        ok = abs(summary["constraint_gap"]) < 0.05 * max(1e-9, summary["budget_d"]) + 0.02
        print(f"  compl. slackness   : {'OK — J≈d at λ>0' if ok else 'J not yet at d (train longer / retune)'}")
    else:
        ok = summary["constraint_gap"] <= 0.02
        print(f"  compl. slackness   : {'OK — J≤d so λ→0' if ok else 'λ≈0 but J>d (gains too small?)'}")
    print("=" * 60)


# ── Environment construction ───────────────────────────────────────────────

def _action_mask_fn(env):
    return env.action_masks()


def build_cmdp_env(scenario: str, seed: int, pid: PIDLagrangian,
                   normalize: bool = True) -> Monitor:
    """CloudSim → CMDPRewardWrapper → ActionMasker → Monitor.

    The base ``CloudSimEnv`` is built with ``normalize_reward=False``; the CMDP
    wrapper owns the *separate* normalisation of ``R_energy`` and ``C_SLA``.
    """
    base = CloudSimEnv(scenario=scenario, seed=seed, normalize_reward=False)
    shaped = CMDPRewardWrapper(base, pid=pid, normalize=normalize)
    masked = ActionMasker(shaped, action_mask_fn=_action_mask_fn)
    return Monitor(masked)


# ── Training driver ────────────────────────────────────────────────────────

def train_cmdp(
    scenario: str,
    seed: int,
    total_timesteps: int,
    budget_d: float,
    *,
    k_p: float = 1e-4,
    k_i: float = 1e-4,
    k_d: float = 0.0,
    lambda_init: float = 0.0,
    dual_every: int = 1,
    n_steps: int = 512,
    n_envs: int = 1,
    torch_threads: int = DEFAULT_TORCH_THREADS,
    out_path: Path,
    tracker=None,
    exporter=None,
    trajectory_csv: Path | None = None,
) -> tuple[Path, PIDLagrangian, dict]:
    """Train one CMDP policy at SLA budget ``d``.

    Parameters
    ----------
    n_envs : int
        SYS.1 — number of parallel CloudSim environments. ``1`` (default) keeps
        the exact single-env path every existing result was produced on. ``>1``
        needs the Java side started with ``NUM_GATEWAYS >= n_envs`` and changes
        rollout composition, so runs are not bit-comparable across different
        ``n_envs`` (see ``vec_env`` for the caveat).

    Returns ``(model_path, pid, convergence_summary)``.
    """
    print(f"[train_cmdp] scenario={scenario}, seed={seed}, d={budget_d}, "
          f"K_P={k_p}, K_I={k_i}, K_D={k_d}, dual_every={dual_every}, "
          f"n_envs={n_envs}, timesteps={total_timesteps:,}")

    # Speed-only knob (measured, see perf/tuning.py): 1 thread is fastest for a
    # [64,64] MLP, keeps parallel runs from oversubscribing the CPU, and fixes
    # the float reduction order so a seed reproduces exactly.
    set_torch_threads(torch_threads)

    pid = PIDLagrangian(k_p=k_p, k_i=k_i, k_d=k_d, lambda_init=lambda_init)

    if n_envs > 1:
        from vec_env import make_cmdp_vec_env
        env = make_cmdp_vec_env(scenario, seed, n_envs=n_envs, lambda_init=lambda_init)
        # With N envs, episodes complete ~N× faster in wall-clock terms. Scaling
        # dual_every by N keeps the dual on the same *episodes-per-update*
        # footing as a single-env run, preserving the two-timescale separation
        # the convergence argument depends on (Lưu ý #6).
        dual_every = dual_every * n_envs
        print(f"[train_cmdp] SYS.1: {n_envs} parallel envs; dual_every scaled "
              f"to {dual_every} episodes to hold the two-timescale separation.")
    else:
        env = build_cmdp_env(scenario, seed, pid)

    # Hyperparameters mirror train_min.py (Phase 1.8) so the only deliberate
    # difference vs the fixed-weight baseline is the CMDP reward + dual update.
    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
        device="cpu",
    )

    callback = LambdaUpdateCallback(
        pid, budget_d, update_every_episodes=dual_every,
        tracker=tracker, exporter=exporter, verbose=1,
    )
    model.learn(total_timesteps=total_timesteps, callback=callback,
                progress_bar=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"[train_cmdp] model saved -> {out_path} (final λ={pid.lambda_:.6g})")

    summary = convergence_summary(callback.history, budget_d)
    print_convergence_summary(summary)

    if trajectory_csv is not None and callback.history:
        import csv
        trajectory_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(trajectory_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(callback.history[0].keys()))
            w.writeheader()
            w.writerows(callback.history)
        print(f"[train_cmdp] λ/J trajectory -> {trajectory_csv}")

    env.close()
    return out_path, pid, summary


# ── Result persistence ─────────────────────────────────────────────────────

def _merge_result_json(path: Path, key: str, row: dict) -> None:
    """Merge one row into a shared JSON under an inter-process lock.

    The sweep can run several ``(d, seed)`` jobs concurrently (SYS.1 / C9), and
    they all merge into the same ``baseline_results.json``. Read-modify-write
    without a lock loses whichever writer read first — silently dropping other
    budgets' rows from the file. The lock makes the read-modify-write atomic
    across processes.

    The write itself goes through a temp file + ``os.replace`` so a crash mid-
    write cannot leave a truncated JSON behind (the file is small, but a lost
    results file at the end of a multi-hour sweep is not a cheap mistake).

    ``fcntl`` is POSIX-only; the container is Linux, and without it there is no
    parallel sweep to protect anyway, so the fallback is simply unlocked.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")

    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX host
        fcntl = None  # type: ignore[assignment]

    lock_file = open(lock_path, "w")
    try:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        existing: dict = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except Exception:
                # A corrupt file must not abort a finished training run; the
                # authoritative per-seed record is points.jsonl.
                existing = {}
        existing[key] = row

        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(existing, indent=2))
        os.replace(tmp, path)
    finally:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


# ── Evaluation: one greedy episode under the trained policy ─────────────────

def evaluate_cmdp(model_path: Path, scenario: str, seed: int, budget_d: float,
                  pid: PIDLagrangian, output_dir: Path) -> dict:
    """Run one greedy episode, export metrics, merge into baseline_results.json.

    Reports the physical CMDP outcome: energy (kWh) and the raw episodic
    constraint cost ``C_SLA`` (from ``getSlaCost()``), so a ``cmdp-d{d}`` row
    sits alongside the heuristics and the fixed-weight PPO for G2.6.
    """
    print(f"[train_cmdp] evaluating {model_path} on {scenario}/seed={seed}")
    env = build_cmdp_env(scenario, seed, pid)
    cloud_env = env.unwrapped  # Monitor/ActionMasker/CMDP → CloudSimEnv

    model = MaskablePPO.load(str(model_path), device="cpu")

    obs, _ = env.reset()
    total_energy_r = 0.0
    steps = 0
    while True:
        masks = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(int(action))
        raw = info.get("raw_reward")
        if raw is not None:
            total_energy_r += float(raw[0])
        steps += 1
        if terminated or truncated:
            break

    energy_kwh = float(cloud_env._ep.getTotalEnergyKwh())
    sla_cost = float(cloud_env._ep.getSlaCost())
    output_dir.mkdir(parents=True, exist_ok=True)
    cloud_env.export_metrics(str(output_dir))

    key = f"cmdp-d{budget_d:g}"
    parent = output_dir.parent  # e.g. /data/results/baseline-HIGH
    row = {
        "scheduler": key,
        "scenario": scenario,
        "seed": seed,
        "steps": steps,
        "total_energy_reward": total_energy_r,
        "total_energy_kwh": energy_kwh,
        "total_sla_cost": sla_cost,
        "sla_budget_d": budget_d,
        "lambda_final": float(pid.lambda_),
    }
    _merge_result_json(parent / "baseline_results.json", key, row)
    print(f"[train_cmdp] merged eval result ({key}) -> "
          f"{parent / 'baseline_results.json'}")

    summary = row
    print(f"[train_cmdp] eval: {summary}")
    env.close()
    return summary


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="HIGH", choices=["LOW", "HIGH", "BURST"])
    p.add_argument("--seed", type=int,
                   default=int(os.environ.get("RANDOM_SEED", "42")))
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--sla-budget", type=float,
                   default=float(os.environ.get("SLA_BUDGET_D", "0.5")),
                   help="Constraint budget d on the episodic SLA cost statistic J "
                        "(E[C_SLA] ≤ d). Sweep this to trace the Pareto front (G2.5).")
    # NOTE on gain scale: Stooke (2020) uses K_P=K_I=1e-4 for cost RETURNS of
    # order ~25 (Safety-Gym). Our constraint statistic J is the episode-MEAN of
    # the normalised per-step cost ~ O(0.05). A PID controller's gains scale
    # INVERSELY with the signal scale, so for this problem the *effective* gains
    # want to be ~25–100× larger (≈2.5e-3…1e-2+). 1e-4 is kept as the documented
    # default for fidelity to Stooke, but for real convergence pass e.g.
    # --k-i 0.05 --k-p 0.05 (see CLAUDE.md G1.5).
    p.add_argument("--k-p", type=float, default=1e-4, help="PID proportional gain")
    p.add_argument("--k-i", type=float, default=1e-4, help="PID integral gain")
    p.add_argument("--k-d", type=float, default=0.0, help="PID derivative gain")
    p.add_argument("--lambda-init", type=float, default=0.0)
    p.add_argument("--dual-every", type=int, default=1,
                   help="Episodes per dual (λ) update — the slower timescale.")
    p.add_argument("--n-steps", type=int, default=512,
                   help="PPO rollout length (the faster, primal timescale).")
    p.add_argument("--torch-threads", type=int, default=DEFAULT_TORCH_THREADS,
                   help="PyTorch intra-op threads. 1 (default) measured fastest for "
                        "this MLP size, avoids oversubscription when sweep runs are "
                        "parallel, and fixes float reduction order for reproducibility. "
                        "Speed-only: never trade hyperparameters for wall-time.")
    p.add_argument("--n-envs", type=int, default=1,
                   help="SYS.1 — parallel CloudSim envs (SubprocVecEnv). Needs the "
                        "Java container started with NUM_GATEWAYS >= n-envs. "
                        "Default 1 = the exact path existing results used; >1 "
                        "changes rollout composition (not bit-comparable).")
    p.add_argument("--model-out", default="/data/models/cmdp.zip", type=Path)
    p.add_argument("--eval-out", default=None,
                   help="Eval export dir (default: /data/results/baseline-<SCENARIO>/cmdp-d<d>)")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--wandb", action="store_true",
                   help="Enable WandB tracking (auto-disabled without WANDB_API_KEY).")
    p.add_argument("--prometheus", action="store_true",
                   help="Enable the Prometheus exporter on port 8000 (G1.6).")
    p.add_argument("--prometheus-port", type=int, default=8000)
    p.add_argument("--trajectory-csv", default=None,
                   help="Dump the per-dual-update λ/J trajectory to this CSV (G1.5).")
    args = p.parse_args()

    tracker = None
    if args.wandb and ExperimentTracker is not None:
        tracker = ExperimentTracker(
            project="ELDAS",
            run_name=f"cmdp-{args.scenario}-d{args.sla_budget:g}-s{args.seed}",
            config=vars(args) | {"phase": "G1.3"},
            tags=["cmdp", "pid-lagrangian", args.scenario],
        )

    exporter = None
    if CMDPMetricsExporter is not None:
        exporter = CMDPMetricsExporter(
            port=args.prometheus_port,
            enabled=True if args.prometheus else None,
        )

    eval_out = Path(args.eval_out) if args.eval_out else Path(
        f"/data/results/baseline-{args.scenario}/cmdp-d{args.sla_budget:g}")
    traj_csv = Path(args.trajectory_csv) if args.trajectory_csv else None

    pid = PIDLagrangian(k_p=args.k_p, k_i=args.k_i, k_d=args.k_d,
                        lambda_init=args.lambda_init)

    if not args.skip_train:
        _, pid, _ = train_cmdp(
            args.scenario, args.seed, args.total_timesteps, args.sla_budget,
            k_p=args.k_p, k_i=args.k_i, k_d=args.k_d, lambda_init=args.lambda_init,
            dual_every=args.dual_every, n_steps=args.n_steps, n_envs=args.n_envs,
            torch_threads=args.torch_threads,
            out_path=args.model_out, tracker=tracker, exporter=exporter,
            trajectory_csv=traj_csv,
        )
    elif not args.model_out.exists():
        print(f"[train_cmdp] --skip-train set but {args.model_out} missing",
              file=sys.stderr)
        return 2

    evaluate_cmdp(args.model_out, args.scenario, args.seed, args.sla_budget,
                  pid, eval_out)

    if tracker is not None:
        tracker.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
