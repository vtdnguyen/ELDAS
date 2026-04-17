package sim;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;

import sim.AlibabaTraceReader.TaskRecord;
import sim.DatacenterFactory.GpuState;

import java.util.*;

/**
 * T3.3 — Random VM allocation policy (baseline #2).
 *
 * Filters hosts by resource feasibility (CPU PEs, RAM, GPUs), then selects
 * uniformly at random among the feasible set.  Serves as a lower-bound
 * baseline for comparison against K8s-style and RL-driven scheduling.
 *
 * <p>Two entry points:
 * <ol>
 *   <li>{@link #defaultFindHostForVm(Vm)} — CloudSim internal path
 *       (CPU + RAM only, no GPU info on {@code Vm})</li>
 *   <li>{@link #selectHostForTask(List, TaskRecord)} — baseline evaluation
 *       path, GPU-aware, returns a 0-based host index</li>
 * </ol>
 */
public class VmAllocationPolicyRandom extends VmAllocationPolicyAbstract {

    private final Map<Host, GpuState> gpuRegistry;
    private final Random random;

    public VmAllocationPolicyRandom(Map<Host, GpuState> gpuRegistry, long seed) {
        this.gpuRegistry = gpuRegistry;
        this.random      = new Random(seed);
    }

    // ── CloudSim integration ──────────────────────────────────────────────

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
     * Select a host index for the given task using random policy.
     * Filters by CPU PEs, RAM, and GPU availability, then picks at random.
     *
     * @param hosts ordered host list (same ordering as {@code SimulationManager})
     * @param task  the task to schedule
     * @return 0-based host index, or 0 if no feasible host found
     */
    public int selectHostForTask(List<Host> hosts, TaskRecord task) {
        List<Integer> feasible = new ArrayList<>();
        for (int i = 0; i < hosts.size(); i++) {
            if (canHost(hosts.get(i), task)) {
                feasible.add(i);
            }
        }
        if (feasible.isEmpty()) return 0;
        return feasible.get(random.nextInt(feasible.size()));
    }

    // ── Feasibility check ─────────────────────────────────────────────────

    private boolean canHost(Host host, TaskRecord task) {
        long freePes = host.getFreePesNumber();
        long freeRam = host.getRam().getAvailableResource();
        GpuState gpu = gpuRegistry.get(host);
        int freeGpus = (gpu != null) ? gpu.available() : 0;

        return freePes >= task.pesNeeded()
            && freeRam >= task.memoryMib()
            && freeGpus >= task.numGpu();
    }
}
