package sim;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;

import sim.AlibabaTraceReader.TaskRecord;

import java.util.*;

/**
 * T8.4 — First-Fit VM allocation policy (Phase 1.8 baseline).
 *
 * <p>The most primitive of the classical bin-packing baselines: walk the
 * host list in index order and pick the first host whose live resources
 * (CPU PEs + RAM + GPUs) satisfy the task.
 *
 * <p>Behavioural expectations under the Phase 1.8 energy model
 * (linear + idle-threshold + host shutdown):
 * <ul>
 *   <li>Tasks gravitate toward the low-index hosts → those hosts stay
 *       ACTIVE; high-index hosts that never get an allocation auto-suspend
 *       past {@code IDLE_THRESHOLD_SEC}.</li>
 *   <li>Energy outcome: better than Random/RoundRobin (because the tail
 *       hosts suspend), worse than BestFit (because FirstFit doesn't try
 *       to pack tightly within an active host before moving on).</li>
 *   <li>SLA outcome: heavy contention on the low-index hosts → worse
 *       than K8s's spread, but better than BestFit's even tighter pack.</li>
 * </ul>
 *
 * <p>Same {@link SimulationManager}-reference rationale as
 * {@link VmAllocationPolicyK8sDefault} — live state lives there, not in
 * CloudSim's Host API.
 */
public class VmAllocationPolicyFirstFit extends VmAllocationPolicyAbstract {

    private final SimulationManager mgr;

    public VmAllocationPolicyFirstFit(SimulationManager mgr) {
        this.mgr = mgr;
    }

    // ── CloudSim integration (unused while DES is not started — see §B8) ──

    @Override
    protected Optional<Host> defaultFindHostForVm(Vm vm) {
        for (Host host : getHostList()) {
            if (host.isSuitableForVm(vm)) return Optional.of(host);
        }
        return Optional.empty();
    }

    // ── Baseline evaluation API ───────────────────────────────────────────

    /**
     * Pick the first feasible host in index order, or 0 if none fit
     * (matches the convention used by K8s/Random — the manager will then
     * defensively redirect or drop the task in {@code SimulationManager}).
     */
    public int selectHostForTask(List<Host> hosts, TaskRecord task) {
        for (int i = 0; i < hosts.size(); i++) {
            if (mgr.canHost(hosts.get(i), task)) return i;
        }
        return 0;
    }
}
