package sim;

/**
 * T2.0 / T7.1 — Central configuration for datacenter topology, host specs,
 * power model and the host state machine (T8.1).
 *
 * <p>Every other class pulls parameters from one source of truth. Two ways
 * to override the defaults:
 * <ol>
 *   <li><b>Environment variables</b> (preferred for Docker): see
 *       {@link #fromEnv()} for the supported keys.</li>
 *   <li><b>JVM system properties</b> (e.g. for unit tests):
 *       {@code -Deldas.num_hosts=20}.</li>
 * </ol>
 * Env wins over system properties; both fall back to the {@code DEFAULT_*}
 * constants below.
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

    // ── Power model specification (linear model + idle threshold) ─────────
    public record PowerSpec(
        double cpuMaxPowerWatt,       // CPU power at 100 % utilization
        double cpuIdlePowerWatt,      // CPU power at   0 % utilization
        double gpuMaxPowerWatt,       // power per GPU card at 100 %
        double gpuIdlePowerWatt,      // power per GPU card at   0 %
        // T8.1 — Host state machine. When a host carries no allocation for
        // longer than `idleThresholdSec`, it transitions IDLE → SUSPENDED
        // and draws `suspendedPowerWatt` (typically ~10 W) instead of the
        // full CPU+GPU idle bill. Waking a SUSPENDED host costs
        // `wakeEnergyKwh` (one-shot) and adds `wakeLatencySec` to the
        // arriving task's completion time.
        double idleThresholdSec,
        double suspendedPowerWatt,
        double wakeEnergyKwh,
        double wakeLatencySec
    ) {}

    // ── Host SKU (G2.1 — one heterogeneous machine class) ─────────────────
    /**
     * G2.1 — A group of {@code count} identical hosts sharing one
     * {@link HostSpec} + {@link PowerSpec}. A heterogeneous datacenter is a
     * list of SKUs (e.g. 3× GPU-heavy, 3× balanced, 4× CPU-only). The
     * homogeneous case is simply a single implicit SKU and is still the
     * default (see {@link DatacenterSpec#skus}).
     */
    public record HostSku(
        String    name,
        int       count,
        HostSpec  hostSpec,
        PowerSpec powerSpec
    ) {}

    // ── Datacenter specification ──────────────────────────────────────────
    /**
     * Datacenter topology. When {@link #skus} is {@code null} or empty the
     * cluster is <b>homogeneous</b>: {@code hostCount} identical hosts built
     * from {@code hostSpec}/{@code powerSpec} — the reproducibility default.
     * When {@code skus} is present the cluster is <b>heterogeneous</b>
     * (G2.1): hosts are built per-SKU and {@code hostSpec}/{@code powerSpec}
     * carry the first SKU's specs only as a reference/fallback (e.g. for
     * top-level printouts); per-host specs are resolved by
     * {@link DatacenterFactory} and {@link SimulationManager}.
     */
    public record DatacenterSpec(
        int          hostCount,
        HostSpec     hostSpec,
        PowerSpec    powerSpec,
        double       schedulingIntervalSec,
        java.util.List<HostSku> skus
    ) {
        /** Homogeneous convenience constructor ({@code skus = null}). */
        public DatacenterSpec(int hostCount, HostSpec hostSpec,
                              PowerSpec powerSpec, double schedulingIntervalSec) {
            this(hostCount, hostSpec, powerSpec, schedulingIntervalSec, null);
        }

        /** True when this spec describes a heterogeneous (multi-SKU) cluster. */
        public boolean isHeterogeneous() {
            return skus != null && !skus.isEmpty();
        }
    }

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
        400.0,    // CPU TDP
        120.0,    // CPU idle
        300.0,    // GPU TDP (per card)
         30.0,    // GPU idle (per card)
         30.0,    // idle → suspended threshold (sim-seconds)
         10.0,    // suspended power draw (W) — BMC + memory refresh only
         0.0005,  // wake-up energy (kWh) ≈ 5 s × 360 W
          5.0     // wake-up latency added to the arriving task (sim-seconds)
    );

    public static final DatacenterSpec DEFAULT_DC = new DatacenterSpec(
        10,               // 10 nodes
        DEFAULT_HOST,
        DEFAULT_POWER,
        1.0               // 1-second scheduling interval
    );

    // ── Trace file path (inside Docker container) ─────────────────────────
    //
    // W2.1 — env-overridable. Leaving TRACE_FILE unset reproduces the Phase-1
    // behaviour byte for byte, which is what every existing result depends on.

    /** Fallback when neither TRACE_FILE nor TRACE_PATTERN is set. */
    public static final String DEFAULT_TRACE_FILE = "/data/trace/openb_pod_list_default.csv";

    public static final String TRACE_FILE =
            stringParam("TRACE_FILE", "eldas.trace_file", DEFAULT_TRACE_FILE);

    // ── W2.2 — per-scenario trace resolution ──────────────────────────────
    //
    // WM-1 emits one file per (arm, scenario, seed), so a run needs to pick the
    // right one at reset() time rather than read a single fixed path:
    //
    //     TRACE_PATTERN=/data/wm1/homo/{scenario}/seed{seed}.csv
    //
    // Unset ⇒ resolveTracePath() returns TRACE_FILE and nothing changes.

    /** Placeholder replaced with the scenario label, e.g. HIGH / OVERLOAD / REPLAY. */
    public static final String TRACE_PATTERN_SCENARIO = "{scenario}";
    /** Placeholder replaced with the episode seed. */
    public static final String TRACE_PATTERN_SEED = "{seed}";

    /** The configured pattern, or {@code null} when running on a single trace file. */
    public static String tracePattern() {
        return resolve("TRACE_PATTERN", "eldas.trace_pattern");
    }

    /** True when traces are selected per scenario rather than read from one file. */
    public static boolean usesTracePattern() {
        return tracePattern() != null;
    }

    /**
     * Resolve the trace for one episode.
     *
     * <p>Two failure modes are turned into exceptions rather than fallbacks, because
     * both would otherwise produce a complete, plausible-looking run against the wrong
     * workload — the most expensive kind of mistake this project can make:
     *
     * <ul>
     *   <li>a pattern without {@code {scenario}} would resolve every scenario to the
     *       same file, so LOW, HIGH and BURST would silently be the same experiment;</li>
     *   <li>a resolved path that does not exist would, under a silent fallback, quietly
     *       run the legacy trace while the operator believes WM-1 is in use.</li>
     * </ul>
     *
     * @param scenario scenario label (case preserved as given by the caller)
     * @param seed     episode seed
     * @return absolute path of the trace to load
     */
    public static String resolveTracePath(String scenario, long seed) {
        return resolveTracePath(tracePattern(), scenario, seed);
    }

    /**
     * Same as {@link #resolveTracePath(String, long)} but with the pattern supplied
     * explicitly, so the resolution rules can be exercised without depending on the
     * ambient environment. {@code pattern == null} means "no pattern configured".
     */
    static String resolveTracePath(String pattern, String scenario, long seed) {
        if (pattern == null || pattern.isBlank()) {
            return TRACE_FILE;
        }
        if (!pattern.contains(TRACE_PATTERN_SCENARIO)) {
            throw new IllegalStateException(
                    "TRACE_PATTERN='" + pattern + "' has no " + TRACE_PATTERN_SCENARIO
                  + " placeholder, so every scenario would resolve to the same file. "
                  + "Use e.g. /data/wm1/homo/" + TRACE_PATTERN_SCENARIO
                  + "/seed" + TRACE_PATTERN_SEED + ".csv");
        }
        String path = pattern
                .replace(TRACE_PATTERN_SCENARIO, scenario)
                .replace(TRACE_PATTERN_SEED, Long.toString(seed));
        if (!java.nio.file.Files.isReadable(java.nio.file.Path.of(path))) {
            throw new IllegalStateException(
                    "TRACE_PATTERN resolved to '" + path + "' which is not readable "
                  + "(scenario=" + scenario + ", seed=" + seed + "). Generate it with "
                  + "scripts/gen-workloads.sh, or unset TRACE_PATTERN to use "
                  + TRACE_FILE + ".");
        }
        return path;
    }

    // ── Env-var driven builder (T7.1) ─────────────────────────────────────

    /**
     * Build a {@link DatacenterSpec} from environment variables, falling
     * back to JVM system properties, then to the {@code DEFAULT_*} constants.
     *
     * <p>Supported keys (env name → property name → default):
     * <table border="1">
     *   <tr><th>Env</th><th>Sys-prop</th><th>Default</th></tr>
     *   <tr><td>{@code NUM_HOSTS}</td>          <td>{@code eldas.num_hosts}</td>          <td>10</td></tr>
     *   <tr><td>{@code VCPU_PER_HOST}</td>      <td>{@code eldas.vcpu_per_host}</td>      <td>64</td></tr>
     *   <tr><td>{@code GPU_PER_HOST}</td>       <td>{@code eldas.gpu_per_host}</td>       <td>8</td></tr>
     *   <tr><td>{@code RAM_PER_HOST_GB}</td>    <td>{@code eldas.ram_per_host_gb}</td>    <td>256</td></tr>
     *   <tr><td>{@code MIPS_PER_PE}</td>        <td>{@code eldas.mips_per_pe}</td>        <td>10000</td></tr>
     *   <tr><td>{@code CPU_MAX_WATT}</td>       <td>{@code eldas.cpu_max_watt}</td>       <td>400</td></tr>
     *   <tr><td>{@code CPU_IDLE_WATT}</td>      <td>{@code eldas.cpu_idle_watt}</td>      <td>120</td></tr>
     *   <tr><td>{@code GPU_MAX_WATT}</td>       <td>{@code eldas.gpu_max_watt}</td>       <td>300</td></tr>
     *   <tr><td>{@code GPU_IDLE_WATT}</td>      <td>{@code eldas.gpu_idle_watt}</td>      <td>30</td></tr>
     *   <tr><td>{@code IDLE_THRESHOLD_SEC}</td> <td>{@code eldas.idle_threshold_sec}</td> <td>30</td></tr>
     *   <tr><td>{@code P_SUSPENDED_W}</td>      <td>{@code eldas.p_suspended_w}</td>      <td>10</td></tr>
     *   <tr><td>{@code WAKE_ENERGY_KWH}</td>    <td>{@code eldas.wake_energy_kwh}</td>    <td>0.0005</td></tr>
     *   <tr><td>{@code WAKE_LATENCY_SEC}</td>   <td>{@code eldas.wake_latency_sec}</td>   <td>5</td></tr>
     * </table>
     *
     * <p>Invalid values fall through to the default silently with a warning
     * — the simulation never aborts on misconfigured tuning knobs.
     */
    public static DatacenterSpec fromEnv() {
        int  numHosts     = intParam ("NUM_HOSTS",       "eldas.num_hosts",       DEFAULT_DC.hostCount());
        int  vcpu         = intParam ("VCPU_PER_HOST",   "eldas.vcpu_per_host",   DEFAULT_HOST.pesCount());
        int  gpuPerHost   = intParam ("GPU_PER_HOST",    "eldas.gpu_per_host",    DEFAULT_HOST.gpuCount());
        int  ramGb        = intParam ("RAM_PER_HOST_GB", "eldas.ram_per_host_gb", (int) (DEFAULT_HOST.ramMb() / 1024));
        long mips         = longParam("MIPS_PER_PE",     "eldas.mips_per_pe",     DEFAULT_HOST.mips());

        double cpuMax     = doubleParam("CPU_MAX_WATT",       "eldas.cpu_max_watt",       DEFAULT_POWER.cpuMaxPowerWatt());
        double cpuIdle    = doubleParam("CPU_IDLE_WATT",      "eldas.cpu_idle_watt",      DEFAULT_POWER.cpuIdlePowerWatt());
        double gpuMax     = doubleParam("GPU_MAX_WATT",       "eldas.gpu_max_watt",       DEFAULT_POWER.gpuMaxPowerWatt());
        double gpuIdle    = doubleParam("GPU_IDLE_WATT",      "eldas.gpu_idle_watt",      DEFAULT_POWER.gpuIdlePowerWatt());
        double idleThresh = doubleParam("IDLE_THRESHOLD_SEC", "eldas.idle_threshold_sec", DEFAULT_POWER.idleThresholdSec());
        double pSusp      = doubleParam("P_SUSPENDED_W",      "eldas.p_suspended_w",      DEFAULT_POWER.suspendedPowerWatt());
        double wakeKwh    = doubleParam("WAKE_ENERGY_KWH",    "eldas.wake_energy_kwh",    DEFAULT_POWER.wakeEnergyKwh());
        double wakeLat    = doubleParam("WAKE_LATENCY_SEC",   "eldas.wake_latency_sec",   DEFAULT_POWER.wakeLatencySec());

        // Light sanity floors — protect downstream math from absurd values.
        numHosts   = Math.max(1, numHosts);
        vcpu       = Math.max(1, vcpu);
        gpuPerHost = Math.max(0, gpuPerHost);
        ramGb      = Math.max(1, ramGb);
        mips       = Math.max(100, mips);

        HostSpec hs = new HostSpec(
                vcpu, mips, ((long) ramGb) * 1024L,
                DEFAULT_HOST.bwMbps(), DEFAULT_HOST.storageMb(),
                gpuPerHost, DEFAULT_HOST.gpuMemoryMb());

        PowerSpec ps = new PowerSpec(
                cpuMax, cpuIdle, gpuMax, gpuIdle,
                idleThresh, pSusp, wakeKwh, wakeLat);

        DatacenterSpec spec = new DatacenterSpec(
                numHosts, hs, ps, DEFAULT_DC.schedulingIntervalSec());

        System.out.printf(
            "[SimulationConfig] fromEnv: hosts=%d, vcpu=%d, gpu=%d, ramGb=%d "
          + "| power: cpu[%.0f/%.0f] gpu[%.0f/%.0f] | "
          + "idle→suspend %.1fs, P_sus=%.0fW, wake=%.4f kWh + %.1fs%n",
            numHosts, vcpu, gpuPerHost, ramGb,
            cpuIdle, cpuMax, gpuIdle, gpuMax,
            idleThresh, pSusp, wakeKwh, wakeLat);

        // G2.1 — Optional heterogeneous topology. When TOPOLOGY_CONFIG points
        // at a readable JSON file, its SKUs override the homogeneous cluster
        // above. Homogeneous stays the default (Lưu ý reproducibility): any
        // load/parse error falls through to `spec` with a warning — the
        // simulation never aborts on a bad topology file.
        String topoPath = resolve("TOPOLOGY_CONFIG", "eldas.topology_config");
        if (topoPath != null) {
            try {
                DatacenterSpec hetero = TopologyConfig.load(topoPath, spec);
                java.util.List<HostSku> skus = hetero.skus();
                System.out.printf(
                    "[SimulationConfig] TOPOLOGY_CONFIG=%s → heterogeneous: "
                  + "%d hosts across %d SKU(s)%n",
                    topoPath, hetero.hostCount(), skus.size());
                for (HostSku s : skus) {
                    System.out.printf(
                        "    SKU %-12s ×%-2d | vcpu=%d ram=%dGB gpu=%d | "
                      + "cpu[%.0f/%.0f]W gpu[%.0f/%.0f]W%n",
                        s.name(), s.count(), s.hostSpec().pesCount(),
                        s.hostSpec().ramMb() / 1024, s.hostSpec().gpuCount(),
                        s.powerSpec().cpuIdlePowerWatt(), s.powerSpec().cpuMaxPowerWatt(),
                        s.powerSpec().gpuIdlePowerWatt(), s.powerSpec().gpuMaxPowerWatt());
                }
                return hetero;
            } catch (Exception e) {
                System.err.printf(
                    "[SimulationConfig] Failed to load TOPOLOGY_CONFIG='%s' (%s) "
                  + "— falling back to homogeneous topology%n", topoPath, e.getMessage());
            }
        }

        return spec;
    }

    // ── Param-resolution helpers ──────────────────────────────────────────
    //
    //   resolution order:  System.getenv("ENV_NAME")
    //                   →  System.getProperty("prop.name")
    //                   →  default
    //
    // Returns the default on null / blank / NumberFormatException so a
    // typo in `.env` never aborts startup.

    private static String resolve(String envName, String propName) {
        String v = System.getenv(envName);
        if (v != null && !v.isBlank()) return v.trim();
        v = System.getProperty(propName);
        if (v != null && !v.isBlank()) return v.trim();
        return null;
    }

    private static String stringParam(String envName, String propName, String dflt) {
        String s = resolve(envName, propName);
        return (s == null) ? dflt : s;
    }

    private static int intParam(String envName, String propName, int dflt) {
        String s = resolve(envName, propName);
        if (s == null) return dflt;
        try { return Integer.parseInt(s); }
        catch (NumberFormatException e) {
            System.err.printf("[SimulationConfig] %s='%s' is not an int — using default %d%n",
                    envName, s, dflt);
            return dflt;
        }
    }

    private static long longParam(String envName, String propName, long dflt) {
        String s = resolve(envName, propName);
        if (s == null) return dflt;
        try { return Long.parseLong(s); }
        catch (NumberFormatException e) {
            System.err.printf("[SimulationConfig] %s='%s' is not a long — using default %d%n",
                    envName, s, dflt);
            return dflt;
        }
    }

    private static double doubleParam(String envName, String propName, double dflt) {
        String s = resolve(envName, propName);
        if (s == null) return dflt;
        try { return Double.parseDouble(s); }
        catch (NumberFormatException e) {
            System.err.printf("[SimulationConfig] %s='%s' is not a number — using default %s%n",
                    envName, s, dflt);
            return dflt;
        }
    }

    // ── QoS → SLA penalty multiplier (κ) ─────────────────────────────────
    // Higher κ  ⇒  heavier penalty for deadline miss.
    //
    // Phase 2 (G1.0): renamed from {@code qosToLambda} to free the symbol λ
    // for the Lagrangian multiplier of the Constrained-MDP formulation. This
    // coefficient is the QoS weight κ in C_SLA = κ·max(0, completion − deadline).
    public static double qosToWeight(String qos) {
        return switch (qos) {
            case "LS"         -> 3.0;   // Latency-Sensitive
            case "Guaranteed" -> 2.0;
            case "Burstable"  -> 1.0;
            case "BE"         -> 0.5;   // Best-Effort
            default           -> 1.0;
        };
    }

    // ── QoS → deadline slack factor ──────────────────────────────────────
    // deadline = creationTime + duration × slackFactor(qos).
    //
    // The looser the slack, the more tolerant the QoS class is of
    // contention-induced slowdown. Tightly coupled to {@link #qosToWeight}:
    // strict classes (LS) get strict deadlines AND high penalty multipliers.
    public static double qosToSlackFactor(String qos) {
        return switch (qos) {
            case "LS"         -> 1.1;   // Latency-Sensitive — 10 % slack only
            case "Guaranteed" -> 1.3;
            case "Burstable"  -> 1.7;
            case "BE"         -> 3.0;   // Best-Effort — essentially uncapped
            default           -> 1.5;
        };
    }

    // ── W1.5 — absolute deadline floor (PLAN-Workload-Model.md §3.8) ──────
    //
    // The multiplicative slack above is a *bounded slowdown* SLO ("this job may
    // take at most f(qos)× as long as on an idle cluster"), which is the standard
    // formulation. On its own, though, it gives short jobs a vanishing absolute
    // budget: measured on the openb trace, 24.6 % of tasks tolerate under 60 s and
    // the median LS budget is 237 s. For those the fixed 5 s wake latency is a
    // large share of the whole budget, so suspending a host to save energy scores
    // as an SLA violation regardless of load — the energy axis and the SLA axis
    // become entangled through an artefact of job length.
    //
    // The floor decouples them:
    //
    //     deadline = creation + duration·slackFactor(qos) + slackFloor(qos)
    //     slackFloor(qos) = SCHEDULING_LAG_P90_SEC · floorFactor(qos)
    //
    // SCHEDULING_LAG_P90_SEC is measured, not chosen: the p90 of
    // (scheduled_time − creation_time) over the 7 255 schedulable openb pods.
    //
    // One GLOBAL percentile scaled per class, not a per-class percentile: the rare
    // classes are too thin to estimate from (Guaranteed n=7 with p90 = 1 s,
    // Burstable n=98 with p90 = 1 s), and a per-class estimate times a per-class
    // factor yields 2 s for Guaranteed against 107 s for LS — it *inverts* the QoS
    // ordering it exists to express.
    //
    // Resulting budgets: LS 106 s < Guaranteed 212 s < Burstable 530 s < BE 2120 s;
    // over the trace min 107 s, p50 553 s, 0 % below 60 s. The constraint stays
    // binding (violation rate 33–49 % across background-utilisation levels), so
    // C_SLA does not collapse to zero.
    //
    // MIRRORED IN rl-agent/src/workload/deadline.py AND rl-agent/src/eval/qos.py —
    // change all three together or ValidationRunner B16b fails.

    /** p90 of the observed scheduling lag in the openb trace (seconds). */
    public static final double SCHEDULING_LAG_P90_SEC = 106.0;

    /** Multiplier on {@link #SCHEDULING_LAG_P90_SEC}; stricter class ⇒ less slack. */
    public static double qosToFloorFactor(String qos) {
        return switch (qos) {
            case "LS"         -> 1.0;
            case "Guaranteed" -> 2.0;
            case "Burstable"  -> 5.0;
            case "BE"         -> 20.0;
            default           -> 1.5;
        };
    }

    /** Absolute lateness a class tolerates regardless of how short the job is. */
    public static double qosToSlackFloorSec(String qos) {
        return SCHEDULING_LAG_P90_SEC * qosToFloorFactor(qos);
    }
}
