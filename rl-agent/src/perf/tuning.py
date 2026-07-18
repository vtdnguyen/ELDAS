"""SYS — Runtime tuning knobs that change speed only, never the algorithm.

Everything here is deliberately restricted to *how* the computation runs, not
*what* it computes. Hyperparameters (``n_epochs``, ``batch_size``, network size)
would all make training faster if lowered — and would change the science. They
are out of scope by design.
"""

from __future__ import annotations

import os

DEFAULT_TORCH_THREADS = 1


def set_torch_threads(n: int = DEFAULT_TORCH_THREADS, verbose: bool = True) -> int:
    """Pin PyTorch's intra-op thread pool.

    Why 1 is the default here, measured rather than assumed
    (``perf/profile_train.py`` on a 12-CPU container, torch defaulting to 6):

    ============ ===============
    torch threads  learn steps/s
    ============ ===============
    1              623.8  ← best
    2              595.5
    6 (default)    572.0
    ============ ===============

    PPO trains a ``[64, 64]`` MLP on batches of 64. At that size the cost of
    fanning each op out across threads and joining them back exceeds the
    arithmetic, so more threads is slower.

    Two further reasons this matters beyond the ~9%:

    * **Oversubscription.** Running R training processes in parallel (the sweep)
      with T threads each wants R×T ≤ cores. At the default 6, four concurrent
      runs would ask for 24 threads on 12 cores and thrash.
    * **Reproducibility.** Multi-threaded float reductions accumulate in
      non-deterministic order, so the same seed can drift between runs. One
      thread makes the reduction order fixed — a *stricter* guarantee, which is
      what Lưu ý #10 (≥5 seeds, reproducible) needs.

    Note this changes speed, not semantics — but a thread-count change can
    perturb the last bits of a float sum, so runs are only bit-comparable at a
    fixed thread count. Keep it constant across the seeds of one comparison.

    Returns the thread count actually applied.
    """
    n = max(1, int(n))
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is always present in the image
        return n

    previous = torch.get_num_threads()
    torch.set_num_threads(n)

    # BLAS backends read these at first use and ignore torch's setting for some
    # ops; set them too so a spawned worker inherits a consistent policy.
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))

    if verbose and previous != n:
        print(f"[tuning] torch threads {previous} → {n} "
              f"(measured fastest for this MLP size; also avoids "
              f"oversubscription when runs are parallel)")
    return n
