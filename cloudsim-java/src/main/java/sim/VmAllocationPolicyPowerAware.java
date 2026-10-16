package sim;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;

import sim.AlibabaTraceReader.TaskRecord;
import sim.SimulationManager.HostState;

import java.util.*;

/**
 * Energy-aware baseline — PABFD (Power-Aware Best Fit Decreasing),
 * Beloglazov, Abawajy &amp; Buyya, FGCS 28(5), 2012, Alg. 1.
 *
 * <p>PABFD places each VM on the host whose <b>power draw increases least</b>.
 * Unlike FirstFit/BestFit/K8s it reads the power model directly: per-SKU
 * {@code P_idle}/{@code P_max}, per-card GPU power, and whether a host is asleep.
 * The "Decreasing" half (sort VMs by size before placing) has no online
 * counterpart — tasks arrive one at a time — so only the per-task rule is kept.
 *
 * <h3>Power increase of placing task t on host h (watts)</h3>
 * <pre>
 *   ΔP_cpu = (P_max − P_idle) · (min(1, (used+pes)/P) − min(1, used/P))
 *   ΔP_gpu = numGpu · (gpuMax − gpuIdle)                (cards switching idle → busy)
 *   ΔP_on  = P_idle + gpuTotal·gpuIdle − P_suspended    (only if h is IDLE or SUSPENDED)
 *   ΔP     = ΔP_cpu + ΔP_gpu + ΔP_on
 * </pre>
 * {@code ΔP_on} is the extension the state machine needs. The original PABFD
 * switches empty hosts off, so any host it can choose is already on; here an
 * empty host is either SUSPENDED or about to be, and placing on it keeps it
 * awake at full idle bill. Charging it makes the rule prefer hosts that are
 * already paying their idle power — the consolidation PABFD is known for — and
 * lets a heterogeneous cluster wake the cheapest SKU when it must wake one.
 *
 * <p>Ties (common: on a homogeneous cluster every ACTIVE host has the same
 * ΔP) are broken by (1) avoiding a wake-up, then (2) the tightest best-fit
 * pack, then (3) the lowest index — deterministic, so CI = 0 across reruns
 * of the same workload.
 *
 * <p>Information parity with the RL agent: host state comes from
 * {@link SimulationManager#hostState}, i.e. the snapshot the observation
 * one-hot encodes; no future arrivals or completions are read.
 */
public class VmAllocationPolicyPowerAware extends VmAllocationPolicyAbstract {

    /** ΔP within this many watts counts as a tie. */
    private static final double EPS_W = 1e-9;

    private final SimulationManager mgr;

    public VmAllocationPolicyPowerAware(SimulationManager mgr) {
        this.mgr = mgr;
    }

    // ── CloudSim Plus allocation API (unused by the RL path) ──────────────

    @Override
    protected Optional<Host> defaultFindHostForVm(Vm vm) {
        for (Host host : getHostList()) {
            if (host.isSuitableForVm(vm)) return Optional.of(host);
        }
        return Optional.empty();
    }

    // ── RL-gateway entry point ────────────────────────────────────────────

    public int selectHostForTask(List<Host> hosts, TaskRecord task) {
        int    bestIdx  = -1;
        double bestDp   = Double.POSITIVE_INFINITY;
        boolean bestWake = true;
        double bestFit  = -1;

        for (int i = 0; i < hosts.size(); i++) {
            Host host = hosts.get(i);
            if (!mgr.canHost(host, task)) continue;

            double dp   = powerIncreaseWatt(host, task);
            boolean wake = mgr.hostState(host) == HostState.SUSPENDED;
            double fit  = fitScore(host, task);

            boolean better;
            if (dp < bestDp - EPS_W)       better = true;
            else if (dp > bestDp + EPS_W)  better = false;
            else if (wake != bestWake)     better = !wake;
            else                           better = fit > bestFit;

            if (better) {
                bestIdx = i; bestDp = dp; bestWake = wake; bestFit = fit;
            }
        }
        return (bestIdx >= 0) ? bestIdx : 0;
    }

    /** Steady-state power increase (W) of placing {@code task} on {@code host}. */
    double powerIncreaseWatt(Host host, TaskRecord task) {
        SimulationConfig.PowerSpec ps = mgr.powerOf(host);
        int pes   = mgr.specOf(host).pesCount();
        int used  = pes - mgr.freePes(host);

        double uBefore = Math.min(1.0, (double) used / pes);
        double uAfter  = Math.min(1.0, (double) (used + task.pesNeeded()) / pes);
        double dCpu = (ps.cpuMaxPowerWatt() - ps.cpuIdlePowerWatt()) * (uAfter - uBefore);

        int gpuTotal = mgr.gpuTotal(host);
        double dGpu = (gpuTotal > 0)
                ? task.numGpu() * (ps.gpuMaxPowerWatt() - ps.gpuIdlePowerWatt())
                : 0.0;

        double dOn = 0.0;
        if (mgr.hostState(host) != HostState.ACTIVE) {
            dOn = ps.cpuIdlePowerWatt() + gpuTotal * ps.gpuIdlePowerWatt()
                - ps.suspendedPowerWatt();
        }
        return dCpu + dGpu + dOn;
    }

    /** Same post-placement tightness score as {@link VmAllocationPolicyBestFit}. */
    private double fitScore(Host host, TaskRecord task) {
        SimulationConfig.HostSpec hs = mgr.hostSpec(host);
        double cpu = (double) (hs.pesCount() - mgr.freePes(host) + task.pesNeeded()) / hs.pesCount();
        double ram = (double) (hs.ramMb() - mgr.freeRam(host) + task.memoryMib()) / hs.ramMb();
        double s = Math.max(cpu, ram);
        int g = hs.gpuCount();
        if (g > 0) s = Math.max(s, (double) (g - mgr.freeGpus(host) + task.numGpu()) / g);
        return s;
    }
}
