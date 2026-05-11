"""
Container entry point for the rl-agent.

Phase 1 default: runs the smoke test (T4.5) so that ``docker compose up``
end-to-end-validates the stack out of the box.

Phase 2 will replace this with a full PPO + MaskablePPO training loop
(see CLAUDE.md → P2.1).

Override at runtime::

    docker compose run --rm rl-agent python src/baseline_eval.py
    docker compose run --rm rl-agent python src/smoke_test.py --scenario HIGH
"""

from __future__ import annotations

import sys

import smoke_test


if __name__ == "__main__":
    sys.exit(smoke_test.main())
