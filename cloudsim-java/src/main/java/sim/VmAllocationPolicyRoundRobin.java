package sim;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;

import sim.AlibabaTraceReader.TaskRecord;

import java.util.*;

/**
 * T8.6 — Round-Robin VM allocation policy (Phase 1.8 baseline).
 *
 * <p>Cycle through hosts in index order, advancing a state-ful pointer
 * after every placement attempt. When the chosen host cannot fit the
 * task, skip forward to the next feasible host (still advancing the
 * pointer past it) — this matches the textbook RR semantics where
 * "infeasible" is treated like "busy".
 *
 * <h3>Pointer reset</h3>
 * The pointer is owned by this instance, so it survives across
 * {@code selectHostForTask} calls within one episode. Each new episode
 * builds a fresh {@code VmAllocationPolicyRoundRobin} in
 * {@link GatewayEntryPoint#reset(String, long)}, so the pointer
 * automatically starts at 0 again — no explicit reset hook needed.
 *
 * <h3>Expected behaviour under the Phase 1.8 energy model</h3>
 * Round-robin spreads allocations evenly. Hosts seldom stay empty long
 * enough to cross the idle threshold, so few SUSPENDED transitions occur,
 * and energy is among the highest of the five baselines (similar to
 * Random in the limit of large task counts). SLA, on the other hand,
 * benefits from low per-host contention.
 *
 * <p>Same {@link SimulationManager}-reference rationale as
 * {@link VmAllocationPolicyK8sDefault}.
 */
public class VmAllocationPolicyRoundRobin extends VmAllocationPolicyAbstract {

    private final SimulationManager mgr;

    /** Next host index to TRY first (advances modulo host count). */
    private int nextHostIndex = 0;

    public VmAllocationPolicyRoundRobin(SimulationManager mgr) {
        this.mgr = mgr;
    }

    // ── CloudSim integration (unused while DES is not started — see §B8) ──

    @Override
    protected Optional<Host> defaultFindHostForVm(Vm vm) {
        List<Host> hosts = getHostList();
        int n = hosts.size();
        if (n == 0) return Optional.empty();
        for (int probe = 0; probe < n; probe++) {
            int idx = (nextHostIndex + probe) % n;
            Host h = hosts.get(idx);
            if (h.isSuitableForVm(vm)) {
                nextHostIndex = (idx + 1) % n;
                return Optional.of(h);
            }
        }
        return Optional.empty();
    }

    // ── Baseline evaluation API ───────────────────────────────────────────

    /**
     * Walk forward from {@link #nextHostIndex}; return the first feasible
     * host's index, then advance the pointer one past it. If no host fits,
     * return 0 without advancing the pointer (caller's defensive path will
     * then drop or redirect — see {@link SimulationManager#steppingLoop}).
     */
    public int selectHostForTask(List<Host> hosts, TaskRecord task) {
        int n = hosts.size();
        if (n == 0) return 0;

        for (int probe = 0; probe < n; probe++) {
            int idx = (nextHostIndex + probe) % n;
            if (mgr.canHost(hosts.get(idx), task)) {
                nextHostIndex = (idx + 1) % n;
                return idx;
            }
        }
        // No feasible host. Don't advance — the next task gets a fresh
        // sweep from the same starting point, preserving fairness.
        return 0;
    }
}
