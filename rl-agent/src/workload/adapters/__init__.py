"""Trace adapters — the only place that knows about a *specific* source trace.

PLAN-Workload-Model.md §7. Everything else in ``workload/`` operates on
:class:`workload.schema.Task` and is deliberately ignorant of where the jobs came from,
so switching to Alibaba GPU v2020 (P2) means adding one module here and registering it —
no change to ``jobsize``, ``arrivals``, ``calibrate`` or ``wm1``.

Capabilities are the safety interlock. A trace that lacks memory and QoS columns cannot
serve as a job-size source, and the failure mode if that is not enforced is silent: WM-1
would happily emit a trace with ``memory_mib=0`` everywhere and the simulator would
schedule it without complaint. ``require()`` turns that into an exception at the call
site instead.

    >>> from workload import adapters
    >>> adapters.list_adapters()
    ['helios', 'openb', 'philly']
    >>> tasks = adapters.load("openb", "/data/trace/openb_pod_list_default.csv")

``philly`` and ``helios`` are the live demonstration of why the interlock is here: they
carry a real submission log but no memory and no QoS, so they declare ``arrivals`` and
``gpu`` and withhold ``jobsize`` (W4.2).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..schema import Task

# ── Capability vocabulary ───────────────────────────────────────────────────

CAP_JOBSIZE = "jobsize"    #: can supply the job-size pool (needs cpu, memory, gpu, qos, duration)
CAP_ARRIVALS = "arrivals"  #: has real per-job arrival timestamps (REPLAY / arrival fitting)
CAP_QOS = "qos"            #: has a QoS class per job
CAP_MEMORY = "memory"      #: has a memory request per job
CAP_GPU = "gpu"            #: has a GPU request per job

ALL_CAPABILITIES = frozenset({CAP_JOBSIZE, CAP_ARRIVALS, CAP_QOS, CAP_MEMORY, CAP_GPU})


class AdapterError(RuntimeError):
    """Raised for unknown adapters and for capability violations."""


@runtime_checkable
class TraceAdapter(Protocol):
    """Interface every source trace must implement."""

    name: str

    def capabilities(self) -> frozenset[str]:
        """What this trace can actually supply, from :data:`ALL_CAPABILITIES`."""

    def load(self, path: str) -> list[Task]:
        """Parse the source file into canonical tasks, sorted by ``creation_time``."""

    def native_arrivals(self, tasks: list[Task]) -> list[float] | None:
        """Real arrival timestamps, or ``None`` if the trace has no usable arrival stream."""


# ── Registry ────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, TraceAdapter] = {}


def register(adapter: TraceAdapter) -> TraceAdapter:
    """Register an adapter. Re-registering the same name is an error, not a silent swap."""
    name = getattr(adapter, "name", None)
    if not name:
        raise AdapterError(f"adapter {adapter!r} has no name")
    if name in _REGISTRY and _REGISTRY[name] is not adapter:
        raise AdapterError(f"adapter name already registered: {name!r}")
    unknown = set(adapter.capabilities()) - ALL_CAPABILITIES
    if unknown:
        raise AdapterError(f"adapter {name!r} declares unknown capabilities: {sorted(unknown)}")
    _REGISTRY[name] = adapter
    return adapter


def get(name: str) -> TraceAdapter:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise AdapterError(
            f"unknown adapter {name!r}; available: {list_adapters()}") from None


def list_adapters() -> list[str]:
    return sorted(_REGISTRY)


def load(name: str, path: str) -> list[Task]:
    """Convenience: ``get(name).load(path)``."""
    return get(name).load(path)


def require(name: str, *capabilities: str) -> TraceAdapter:
    """Fetch an adapter, asserting it supplies every capability the caller needs.

    Call this — not :func:`get` — whenever the result feeds a model, so that using a
    trace for something it cannot support fails at the boundary with a readable message.
    """
    adapter = get(name)
    have = adapter.capabilities()
    missing = [c for c in capabilities if c not in have]
    if missing:
        raise AdapterError(
            f"adapter {name!r} cannot supply {missing}; it provides {sorted(have)}. "
            f"See PLAN-Workload-Model.md §7/§8 for why this trace is limited.")
    return adapter


# ── Built-in adapters ───────────────────────────────────────────────────────
# Imported for their registration side effect; keep at the bottom so the registry
# helpers above are defined first.

from . import openb as _openb    # noqa: E402,F401  (side effect: register)
from . import philly as _philly  # noqa: E402,F401  (registers "philly" and "helios")

__all__ = [
    "TraceAdapter", "AdapterError",
    "CAP_JOBSIZE", "CAP_ARRIVALS", "CAP_QOS", "CAP_MEMORY", "CAP_GPU", "ALL_CAPABILITIES",
    "register", "get", "list_adapters", "load", "require",
]
