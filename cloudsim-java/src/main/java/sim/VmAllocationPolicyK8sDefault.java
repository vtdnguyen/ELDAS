package sim;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;

import sim.AlibabaTraceReader.TaskRecord;
import sim.DatacenterFactory.GpuState;

import java.util.*;

/**
 * T3.2 — Kubernetes-style default scheduler: Filter → Score → Select.
 *
 * Mimics the real K8s default scheduling algorithm:
 * <ol>
 *   <li><b>Filter</b> (Predicates) — exclude hosts that cannot satisfy the
 *       task's resource requirements (CPU PEs, RAM, GPUs)</li>
 *   <li><b>Score</b> (Priorities) — rank remaining hosts using
 *       <em>LeastRequestedPriority</em>:
 *       {@code score = (freeCpu/totalCpu + freeMem/totalMem) / 2}.
 *       GPU is treated as an extended resource (filter-only, not scored),
 *       matching real K8s behavior.</li>
 *   <li><b>Select</b> — pick the host with the highest score.
 *       Ties are broken by lowest index for determinism.</li>
 * </ol>
 *
 * <p>Two entry points:
 * <ol>
 *   <li>{@link #defaultFindHostForVm(Vm)} — CloudSim internal path</li>
 *   <li>{@link #selectHostForTask(List, TaskRecord)} — baseline evaluation
 *       path, GPU-aware, returns a 0-based host index</li>
 * </ol>
 */
public class VmAllocationPolicyK8sDefault extends VmAllocationPolicyAbstract {

    private final Map<Host, GpuState> gpuRegistry;

    public VmAllocationPolicyK8sDefault(Map<Host, GpuState> gpuRegistry) {
        this.gpuRegistry = gpuRegistry;
    }

    // ── CloudSim integration ──────────────────────────────────────────────

    @Override
    protected Optional<Host> defaultFindHostForVm(Vm vm) {
        Host best = null;
        double bestScore = -1;

        for (Host host : getHostList()) {
            if (!host.isSuitableForVm(vm)) continue;

            double score = leastRequestedScore(host);
            if (score > bestScore) {
                bestScore = score;
                best = host;
            }
        }

        return Optional.ofNullable(best);
    }

    // ── Baseline evaluation API ───────────────────────────────────────────

    /**
     * Select a host index for the given task using K8s-style scheduling.
     *
     * <ol>
     *   <li>Filter hosts by CPU, RAM, and GPU feasibility</li>
     *   <li>Score feasible hosts with LeastRequestedPriority</li>
     *   <li>Return the index of the highest-scoring host</li>
     * </ol>
     *
     * @param hosts ordered host list (same ordering as {@code SimulationManager})
     * @param task  the task to schedule
     * @return 0-based host index, or 0 if no feasible host found
     */
    public int selectHostForTask(List<Host> hosts, TaskRecord task) {
        int bestIdx = -1;
        double bestScore = -1;

        for (int i = 0; i < hosts.size(); i++) {
            Host host = hosts.get(i);
            if (!canHost(host, task)) continue;

            double score = leastRequestedScore(host);
            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }

        return (bestIdx >= 0) ? bestIdx : 0;
    }

    // ── LeastRequestedPriority scoring ────────────────────────────────────

    /**
     * K8s LeastRequestedPriority: prefer hosts with the most free resources.
     * <pre>
     *   score = (freeCpu / totalCpu + freeMem / totalMem) / 2
     * </pre>
     * Range: [0.0, 1.0] where 1.0 = completely idle host.
     */
    private double leastRequestedScore(Host host) {
        double cpuFraction = (double) host.getFreePesNumber() / host.getPesNumber();
        double memFraction = (double) host.getRam().getAvailableResource()
                                     / host.getRam().getCapacity();
        return (cpuFraction + memFraction) / 2.0;
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
