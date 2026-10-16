package sim;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;

import sim.AlibabaTraceReader.TaskRecord;

import java.util.*;

/**
 * Kubernetes scheduler with the bin-packing scoring strategy — the
 * {@code NodeResourcesFit} plugin configured with
 * {@code scoringStrategy.type: MostAllocated} (Kubernetes docs,
 * "Resource Bin Packing"). This is the setting Kubernetes documents for
 * consolidating pods onto fewer nodes so the rest can be scaled down, i.e.
 * its energy/cost-oriented mode. {@link VmAllocationPolicyK8sDefault} is the
 * stock {@code LeastAllocated} strategy, which spreads.
 *
 * <h3>Score</h3>
 * Per resource r, after adding the pod: {@code requested_r / allocatable_r}.
 * Node score = weighted mean over the configured resources (all weights 1).
 * Resources: {@code cpu}, {@code memory}, plus {@code nvidia.com/gpu} on GPU
 * nodes — the docs' example adds extended resources to the list the same way,
 * and on this workload GPUs carry most of the power bill. Highest score wins;
 * ties go to the lowest index.
 */
public class VmAllocationPolicyK8sMostAllocated extends VmAllocationPolicyAbstract {

    private final SimulationManager mgr;

    public VmAllocationPolicyK8sMostAllocated(SimulationManager mgr) {
        this.mgr = mgr;
    }

    @Override
    protected Optional<Host> defaultFindHostForVm(Vm vm) {
        for (Host host : getHostList()) {
            if (host.isSuitableForVm(vm)) return Optional.of(host);
        }
        return Optional.empty();
    }

    public int selectHostForTask(List<Host> hosts, TaskRecord task) {
        int bestIdx = -1;
        double bestScore = -1;
        for (int i = 0; i < hosts.size(); i++) {
            Host host = hosts.get(i);
            if (!mgr.canHost(host, task)) continue;
            double score = mostAllocatedScore(host, task);
            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }
        return (bestIdx >= 0) ? bestIdx : 0;
    }

    double mostAllocatedScore(Host host, TaskRecord task) {
        SimulationConfig.HostSpec hs = mgr.hostSpec(host);
        double cpu = (double) (hs.pesCount() - mgr.freePes(host) + task.pesNeeded()) / hs.pesCount();
        double ram = (double) (hs.ramMb() - mgr.freeRam(host) + task.memoryMib()) / hs.ramMb();
        double sum = cpu + ram;
        int n = 2;
        int g = hs.gpuCount();
        if (g > 0) {
            sum += (double) (g - mgr.freeGpus(host) + task.numGpu()) / g;
            n++;
        }
        return sum / n;
    }
}
