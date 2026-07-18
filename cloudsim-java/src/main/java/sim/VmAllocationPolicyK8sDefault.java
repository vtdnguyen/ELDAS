package sim;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;

import sim.AlibabaTraceReader.TaskRecord;

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
 * <p><b>Why it takes a {@link SimulationManager} reference.</b> CloudSim's
 * {@code Host.getFreePesNumber()} / {@code getRam().getAvailableResource()}
 * always report full capacity because we never submit Vms (manual
 * allocation only — see java-validation-report §B1). Reading those would
 * make every host score 1.0 ⇒ tie-break picks index 0 every time. This
 * class therefore queries live state via {@link SimulationManager#freePes},
 * {@link SimulationManager#freeRam}, {@link SimulationManager#canHost}.
 */
public class VmAllocationPolicyK8sDefault extends VmAllocationPolicyAbstract {

    private final SimulationManager mgr;

    public VmAllocationPolicyK8sDefault(SimulationManager mgr) {
        this.mgr = mgr;
    }

    // ── CloudSim integration (unused while DES is not started — see §B8) ──

    @Override
    protected Optional<Host> defaultFindHostForVm(Vm vm) {
        Host best = null;
        double bestScore = -1;

        for (Host host : getHostList()) {
            if (!host.isSuitableForVm(vm)) continue;
            double score = leastRequestedScoreFromCloudSim(host);
            if (score > bestScore) {
                bestScore = score;
                best = host;
            }
        }
        return Optional.ofNullable(best);
    }

    // ── Baseline evaluation API (the only path actually exercised) ────────

    /**
     * Select a host index for the given task using K8s-style scheduling.
     *
     * @param hosts ordered host list (same ordering as {@link SimulationManager#getHosts})
     * @param task  the task to schedule
     * @return 0-based host index, or 0 if no feasible host found
     */
    public int selectHostForTask(List<Host> hosts, TaskRecord task) {
        int bestIdx = -1;
        double bestScore = -1;

        for (int i = 0; i < hosts.size(); i++) {
            Host host = hosts.get(i);
            if (!mgr.canHost(host, task)) continue;

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
     * K8s LeastRequestedPriority over live SimulationManager state:
     * <pre>
     *   score = (freePes/pesCount + freeRam/ramMb) / 2
     * </pre>
     * Range: [0, 1]; 1 = completely idle host.
     */
    private double leastRequestedScore(Host host) {
        // G2.1 — per-host spec so scoring is correct under heterogeneity.
        SimulationConfig.HostSpec hs = mgr.hostSpec(host);
        double cpuFraction = (double) mgr.freePes(host) / hs.pesCount();
        double memFraction = (double) mgr.freeRam(host) / hs.ramMb();
        return (cpuFraction + memFraction) / 2.0;
    }

    /** Score variant for the CloudSim-internal path (dead code today). */
    private double leastRequestedScoreFromCloudSim(Host host) {
        double cpuFraction = (double) host.getFreePesNumber() / host.getPesNumber();
        double memFraction = (double) host.getRam().getAvailableResource()
                                     / host.getRam().getCapacity();
        return (cpuFraction + memFraction) / 2.0;
    }
}
