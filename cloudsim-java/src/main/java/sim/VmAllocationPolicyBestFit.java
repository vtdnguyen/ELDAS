package sim;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;

import sim.AlibabaTraceReader.TaskRecord;

import java.util.*;

/**
 * T8.5 — Best-Fit VM allocation policy (Phase 1.8 baseline).
 *
 * <p>Classical bin-packing best-fit: among hosts that can fit the task,
 * pick the one whose resource utilisation will be the <b>highest</b> after
 * the placement (i.e. leaves the least free room). This packs the cluster
 * as tightly as possible, leaving more hosts to fall below the idle
 * threshold and auto-SUSPEND.
 *
 * <h3>Scoring</h3>
 * Per-host fit score is the max of the three post-allocation utilisations:
 * <pre>
 *   cpu_after = (usedPes + task.pes) / totalPes
 *   ram_after = (usedRam + task.ram) / totalRam
 *   gpu_after = (usedGpu + task.gpu) / totalGpu     (omitted when totalGpu = 0)
 *   score     = max(cpu_after, ram_after, gpu_after)
 * </pre>
 * We pick the host with the <b>largest</b> score (closest to 1.0 = tightest
 * pack) among feasible hosts. Range [0, 1]; the {@code max} is preferred
 * over an average because a near-saturated dimension is more "useful pack"
 * evidence than a balanced low-load one.
 *
 * <h3>Tie-break</h3>
 * Deterministic by lowest {@code host_id} — ensures reproducibility across
 * runs with the same trace+seed.
 *
 * <h3>Expected ordering (Phase 1.8)</h3>
 * With the state machine in place:
 * <pre>
 *   Energy:  BestFit  &lt;  K8s ≤ FirstFit  &lt;  Random ≈ RoundRobin
 *   SLA:     BestFit  &gt;  K8s             &gt;  FirstFit  &gt; Random ≈ RR
 *          (pack tight → contention)        (spread     → less contention)
 * </pre>
 * If BestFit does NOT beat K8s on energy, the state machine is broken
 * (no idle threshold, hosts staying ACTIVE forever) — investigate before
 * proceeding to PPO-min training in T8.9.
 */
public class VmAllocationPolicyBestFit extends VmAllocationPolicyAbstract {

    private final SimulationManager mgr;

    public VmAllocationPolicyBestFit(SimulationManager mgr) {
        this.mgr = mgr;
    }

    // ── CloudSim integration (unused while DES is not started — see §B8) ──

    @Override
    protected Optional<Host> defaultFindHostForVm(Vm vm) {
        Host bestHost = null;
        double bestScore = -1;
        for (Host host : getHostList()) {
            if (!host.isSuitableForVm(vm)) continue;
            // Approximate with the CloudSim API; not exercised in practice.
            double cpuAfter = 1.0 - (double) (host.getFreePesNumber() - vm.getPesNumber())
                                   / host.getPesNumber();
            double ramAfter = 1.0 - (double) (host.getRam().getAvailableResource()
                                              - vm.getRam().getCapacity())
                                   / host.getRam().getCapacity();
            double score = Math.max(cpuAfter, ramAfter);
            if (score > bestScore) {
                bestScore = score;
                bestHost = host;
            }
        }
        return Optional.ofNullable(bestHost);
    }

    // ── Baseline evaluation API ───────────────────────────────────────────

    /**
     * Pick the feasible host that will be most fully utilised after the
     * placement. See class javadoc for the scoring rule.
     */
    public int selectHostForTask(List<Host> hosts, TaskRecord task) {
        int bestIdx = -1;
        double bestScore = -1;

        for (int i = 0; i < hosts.size(); i++) {
            Host host = hosts.get(i);
            if (!mgr.canHost(host, task)) continue;

            // G2.1 — per-host capacity so tightness is measured against THIS
            // host's SKU (a task fills a small CPU-only host more than a big
            // GPU node), which is exactly what best-fit packing should reward.
            SimulationConfig.HostSpec hs = mgr.hostSpec(host);
            int  totalPes = hs.pesCount();
            long totalRam = hs.ramMb();
            int  totalGpu = hs.gpuCount();

            // Post-placement utilisation per dimension. mgr.freePes returns
            // remaining capacity, so (totalPes − free + task.pes) = usage after.
            int  usedPesAfter = totalPes  - mgr.freePes(host) + task.pesNeeded();
            long usedRamAfter = totalRam  - mgr.freeRam(host) + task.memoryMib();

            double cpuAfter = (double) usedPesAfter / totalPes;
            double ramAfter = (double) usedRamAfter / totalRam;
            double score = Math.max(cpuAfter, ramAfter);

            if (totalGpu > 0) {
                int usedGpuAfter = totalGpu - mgr.freeGpus(host) + task.numGpu();
                double gpuAfter = (double) usedGpuAfter / totalGpu;
                score = Math.max(score, gpuAfter);
            }

            // Strict > preserves the lowest-index tie-break.
            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }
        return (bestIdx >= 0) ? bestIdx : 0;
    }
}
