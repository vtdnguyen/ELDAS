package sim;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;

import sim.AlibabaTraceReader.TaskRecord;

import java.util.*;

/**
 * T3.3 — Random VM allocation policy (baseline #2).
 *
 * Filters hosts by resource feasibility (CPU PEs, RAM, GPUs), then selects
 * uniformly at random among the feasible set. Serves as a lower-bound
 * baseline for comparison against K8s-style and RL-driven scheduling.
 *
 * <p>Same {@link SimulationManager}-reference rationale as
 * {@link VmAllocationPolicyK8sDefault} — see that class for details.
 */
public class VmAllocationPolicyRandom extends VmAllocationPolicyAbstract {

    private final SimulationManager mgr;
    private final Random random;

    public VmAllocationPolicyRandom(SimulationManager mgr, long seed) {
        this.mgr    = mgr;
        this.random = new Random(seed);
    }

    // ── CloudSim integration (unused while DES is not started — see §B8) ──

    @Override
    protected Optional<Host> defaultFindHostForVm(Vm vm) {
        List<Host> feasible = getHostList().stream()
                .filter(h -> h.isSuitableForVm(vm))
                .toList();
        if (feasible.isEmpty()) return Optional.empty();
        return Optional.of(feasible.get(random.nextInt(feasible.size())));
    }

    // ── Baseline evaluation API ───────────────────────────────────────────

    /**
     * Pick a host uniformly at random among those that can actually fit the
     * task (CPU + RAM + GPU, all read from live SimulationManager state).
     */
    public int selectHostForTask(List<Host> hosts, TaskRecord task) {
        List<Integer> feasible = new ArrayList<>();
        for (int i = 0; i < hosts.size(); i++) {
            if (mgr.canHost(hosts.get(i), task)) {
                feasible.add(i);
            }
        }
        if (feasible.isEmpty()) return 0;
        return feasible.get(random.nextInt(feasible.size()));
    }
}
