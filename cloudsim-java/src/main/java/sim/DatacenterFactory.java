package sim;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicy;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.power.models.PowerModelHostSimple;
import org.cloudsimplus.provisioners.ResourceProvisionerSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.vm.VmSchedulerTimeShared;

import java.util.*;
import java.util.stream.IntStream;

/**
 * T2.1 — Factory that builds a CloudSim Plus {@link Datacenter} from
 * {@link SimulationConfig.DatacenterSpec} and tracks GPU state per host.
 *
 * CloudSim Plus has no native GPU concept, so we maintain a parallel
 * {@code Map<Host, GpuState>} that records total / used GPU counts.
 */
public final class DatacenterFactory {

    // ── GPU bookkeeping ────────────────────────────────────────────────────

    /** Mutable GPU allocation state for a single host. */
    public static final class GpuState {
        private final int total;
        private int used;

        public GpuState(int total) {
            this.total = total;
            this.used  = 0;
        }

        public int  total()     { return total; }
        public int  used()      { return used;  }
        public int  available() { return total - used; }

        /** Try to allocate {@code n} GPUs. Returns {@code true} on success. */
        public boolean allocate(int n) {
            if (n < 0 || n > available()) return false;
            used += n;
            return true;
        }

        /** Release {@code n} GPUs back to the pool. */
        public void release(int n) {
            used = Math.max(0, used - n);
        }

        public double utilization() {
            return total == 0 ? 0.0 : (double) used / total;
        }
    }

    // ── Factory method ─────────────────────────────────────────────────────

    /**
     * Creates a {@link Datacenter} with the default {@link VmAllocationPolicySimple}.
     * T3.2 / T3.3 will supply custom policies (K8s-style, Random, RL-driven).
     */
    public static Datacenter create(CloudSimPlus simulation,
                                    SimulationConfig.DatacenterSpec dcSpec,
                                    Map<Host, GpuState> gpuRegistry) {
        return create(simulation, dcSpec, gpuRegistry, new VmAllocationPolicySimple());
    }

    /**
     * Creates a {@link Datacenter} with a caller-supplied allocation policy,
     * discarding the per-host spec maps (homogeneous callers / tests).
     */
    public static Datacenter create(CloudSimPlus simulation,
                                    SimulationConfig.DatacenterSpec dcSpec,
                                    Map<Host, GpuState> gpuRegistry,
                                    VmAllocationPolicy policy) {
        return create(simulation, dcSpec, gpuRegistry,
                new java.util.IdentityHashMap<>(), new java.util.IdentityHashMap<>(),
                policy);
    }

    /**
     * Creates a {@link Datacenter}, populating GPU state <b>and</b> per-host
     * spec maps so that heterogeneous topologies (G2.1) can be accounted
     * host-by-host downstream.
     *
     * <p>When {@code dcSpec.isHeterogeneous()} is {@code true}, hosts are built
     * per {@link SimulationConfig.HostSku} (SKU order preserved). Otherwise the
     * cluster is homogeneous: {@code hostCount} identical hosts from
     * {@code dcSpec.hostSpec()}/{@code dcSpec.powerSpec()} — every host maps to
     * the same spec, so downstream code is oblivious to the distinction.
     *
     * @param simulation   the running CloudSimPlus instance
     * @param dcSpec       topology + power config (homogeneous or multi-SKU)
     * @param gpuRegistry  (output) one {@link GpuState} per host (SKU gpu count)
     * @param hostSpecOut  (output) per-host {@link SimulationConfig.HostSpec}
     * @param powerSpecOut (output) per-host {@link SimulationConfig.PowerSpec}
     * @param policy       VM-to-Host allocation policy
     * @return fully configured Datacenter
     */
    public static Datacenter create(CloudSimPlus simulation,
                                    SimulationConfig.DatacenterSpec dcSpec,
                                    Map<Host, GpuState> gpuRegistry,
                                    Map<Host, SimulationConfig.HostSpec> hostSpecOut,
                                    Map<Host, SimulationConfig.PowerSpec> powerSpecOut,
                                    VmAllocationPolicy policy) {

        // Build hosts and a parallel list of per-host specs, in the SAME order.
        List<Host> hosts = new ArrayList<>();
        List<SimulationConfig.HostSpec>  perHostSpec  = new ArrayList<>();
        List<SimulationConfig.PowerSpec> perHostPower = new ArrayList<>();

        if (dcSpec.isHeterogeneous()) {
            for (SimulationConfig.HostSku sku : dcSpec.skus()) {
                for (int i = 0; i < sku.count(); i++) {
                    hosts.add(createHost(sku.hostSpec(), sku.powerSpec()));
                    perHostSpec.add(sku.hostSpec());
                    perHostPower.add(sku.powerSpec());
                }
            }
        } else {
            SimulationConfig.HostSpec  hs = dcSpec.hostSpec();
            SimulationConfig.PowerSpec ps = dcSpec.powerSpec();
            for (int i = 0; i < dcSpec.hostCount(); i++) {
                hosts.add(createHost(hs, ps));
                perHostSpec.add(hs);
                perHostPower.add(ps);
            }
        }

        DatacenterSimple dc = new DatacenterSimple(simulation, hosts, policy);
        dc.setSchedulingInterval(dcSpec.schedulingIntervalSec());

        // Populate the registries AFTER datacenter creation, aligning by index:
        // DatacenterSimple preserves the host list order it was given.
        List<Host> ordered = dc.getHostList();
        for (int i = 0; i < ordered.size(); i++) {
            Host host = ordered.get(i);
            SimulationConfig.HostSpec  hs = perHostSpec.get(i);
            SimulationConfig.PowerSpec ps = perHostPower.get(i);
            gpuRegistry .put(host, new GpuState(hs.gpuCount()));
            hostSpecOut .put(host, hs);
            powerSpecOut.put(host, ps);
        }

        if (dcSpec.isHeterogeneous()) {
            System.out.printf("[DatacenterFactory] Created HETEROGENEOUS datacenter: "
                            + "%d hosts across %d SKU(s)%n",
                    ordered.size(), dcSpec.skus().size());
        } else {
            SimulationConfig.HostSpec hs = dcSpec.hostSpec();
            System.out.printf("[DatacenterFactory] Created datacenter: %d hosts, "
                            + "%d PEs/host, %d GPUs/host%n",
                    dcSpec.hostCount(), hs.pesCount(), hs.gpuCount());
        }

        return dc;
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    private static Host createHost(SimulationConfig.HostSpec hs,
                                   SimulationConfig.PowerSpec ps) {

        List<Pe> peList = IntStream.range(0, hs.pesCount())
                .mapToObj(i -> (Pe) new PeSimple(hs.mips()))
                .toList();

        HostSimple host = new HostSimple(hs.ramMb(), hs.bwMbps(),
                                         hs.storageMb(), peList);
        host.setRamProvisioner(new ResourceProvisionerSimple())
            .setBwProvisioner(new ResourceProvisionerSimple())
            .setVmScheduler(new VmSchedulerTimeShared());

        // CPU power model (linear interpolation between idle and max)
        host.setPowerModel(new PowerModelHostSimple(
                ps.cpuMaxPowerWatt(), ps.cpuIdlePowerWatt()));

        // NOTE: host.enableUtilizationStats() removed — we never submit
        // Vms to hosts, so CloudSim logs a repetitive INFO warning each
        // time a host is built. Utilisation is tracked manually in
        // SimulationManager.hostPeUsage instead.
        return host;
    }

    // ── GPU power helper (called externally — not part of CloudSim model) ─

    /**
     * Compute instantaneous GPU power for one host based on its allocation.
     * Uses the same linear model as CPU: P = idle + (max − idle) × utilisation.
     *
     * GPU utilisation is binary per card: a card is either 100 % (allocated)
     * or 0 % (idle), which reflects typical deep-learning workloads.
     */
    public static double gpuPowerWatt(GpuState gpu, SimulationConfig.PowerSpec ps) {
        int activeCards = gpu.used();
        int idleCards   = gpu.total() - activeCards;
        return activeCards * ps.gpuMaxPowerWatt()
             + idleCards   * ps.gpuIdlePowerWatt();
    }

    private DatacenterFactory() {} // utility class
}
