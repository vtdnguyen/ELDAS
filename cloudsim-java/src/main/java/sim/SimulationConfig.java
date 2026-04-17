package sim;

/**
 * T2.0 — Central configuration for datacenter topology, host specs, and power model.
 *
 * All simulation parameters live here so that every other class pulls from
 * one source of truth. Adjust the DEFAULT_* constants or build a custom
 * {@link DatacenterSpec} for experiments.
 */
public final class SimulationConfig {

    private SimulationConfig() {} // utility class — no instances

    // ── Host hardware specification ────────────────────────────────────────
    public record HostSpec(
        int    pesCount,     // number of CPU cores (PEs)
        long   mips,         // MIPS per PE
        long   ramMb,        // RAM in MB
        long   bwMbps,       // network bandwidth in Mbps
        long   storageMb,    // disk storage in MB
        int    gpuCount,     // GPUs per host
        long   gpuMemoryMb   // VRAM per GPU in MB
    ) {}

    // ── Power model specification (linear model) ──────────────────────────
    public record PowerSpec(
        double cpuMaxPowerWatt,   // CPU power at 100 % utilization
        double cpuIdlePowerWatt,  // CPU power at   0 % utilization
        double gpuMaxPowerWatt,   // power per GPU card at 100 %
        double gpuIdlePowerWatt   // power per GPU card at   0 %
    ) {}

    // ── Datacenter specification ──────────────────────────────────────────
    public record DatacenterSpec(
        int          hostCount,
        HostSpec     hostSpec,
        PowerSpec    powerSpec,
        double       schedulingIntervalSec
    ) {}

    // ── Defaults — modeled after a typical 8-GPU training node ────────────
    public static final HostSpec DEFAULT_HOST = new HostSpec(
        64,          // 64 vCPUs
        10_000,      // 10 000 MIPS each
        262_144,     // 256 GB RAM
        10_000,      // 10 Gbps NIC
        1_000_000,   // 1 TB SSD
        8,           // 8× GPU
        32_768       // 32 GB VRAM (V100-32 GB class)
    );

    public static final PowerSpec DEFAULT_POWER = new PowerSpec(
        400.0,   // CPU TDP
        120.0,   // CPU idle
        300.0,   // GPU TDP (per card)
         30.0    // GPU idle (per card)
    );

    public static final DatacenterSpec DEFAULT_DC = new DatacenterSpec(
        10,               // 10 nodes
        DEFAULT_HOST,
        DEFAULT_POWER,
        1.0               // 1-second scheduling interval
    );

    // ── Trace file path (inside Docker container) ─────────────────────────
    public static final String TRACE_FILE = "/data/trace/openb_pod_list_default.csv";

    // ── QoS → SLA penalty multiplier (λ) ─────────────────────────────────
    // Higher λ  ⇒  heavier penalty for deadline miss
    public static double qosToLambda(String qos) {
        return switch (qos) {
            case "LS"         -> 3.0;   // Latency-Sensitive
            case "Guaranteed" -> 2.0;
            case "Burstable"  -> 1.0;
            case "BE"         -> 0.5;   // Best-Effort
            default           -> 1.0;
        };
    }
}
