package sim;

import io.prometheus.client.CollectorRegistry;
import io.prometheus.client.Counter;
import io.prometheus.client.Gauge;
import io.prometheus.client.exporter.HTTPServer;

import java.io.IOException;
import java.util.Locale;

/**
 * T5.3 — Prometheus exporter façade (opt-in observability).
 *
 * <p>This class is a stateless static façade over the Prometheus simpleclient.
 * It exposes ELDAS-specific metrics on an embedded HTTP endpoint that
 * Prometheus scrapes (default {@code 0.0.0.0:9091/metrics}).
 *
 * <h2>Activation</h2>
 * The exporter is <b>off by default</b>. It only starts when the environment
 * variable {@code MONITORING_ENABLED=true} is set at JVM startup. When off,
 * every public method on this class is a fast no-op — no allocations, no
 * thread switches, no I/O.
 *
 * <h2>Safety guarantees</h2>
 * <ul>
 *   <li><b>No exceptions leak.</b> Every metric mutation is wrapped in a
 *       broad catch — a bug in monitoring cannot crash the simulation.</li>
 *   <li><b>Idempotent {@link #start()}.</b> Calling twice is harmless.</li>
 *   <li><b>Bounded cardinality.</b> Per-host gauges carry only
 *       {@code host_id × scenario × scheduler}; episode dimension lives in
 *       Prometheus time, not in labels.</li>
 *   <li><b>Thread-safe.</b> Prometheus simpleclient gauges/counters use
 *       internal {@code DoubleAdder} — safe to mutate from the sim thread.</li>
 * </ul>
 *
 * <h2>Wiring</h2>
 * <ol>
 *   <li>{@link Main} calls {@link #start()} once on JVM boot.</li>
 *   <li>{@link GatewayEntryPoint#reset} calls {@link #setContext} when
 *       Python begins a new episode.</li>
 *   <li>{@link GatewayEntryPoint#selectBaselineAction} updates only the
 *       scheduler tag, leaving scenario untouched.</li>
 *   <li>{@link SimulationManager} pushes per-snapshot host utilisation and
 *       events (SLA violations, task placements) into the registry.</li>
 * </ol>
 */
public final class MetricsRegistry {

    private MetricsRegistry() {} // façade — no instances

    // ── State ──────────────────────────────────────────────────────────────

    /**
     * Flipped to {@code true} only after a successful {@link #start()}.
     * Volatile so the sim thread observes the change without locking.
     */
    private static volatile boolean enabled = false;

    private static HTTPServer server;

    // Per-host gauges (low cardinality, see class javadoc).
    private static Gauge hostCpuUtil;
    private static Gauge hostMemUtil;
    private static Gauge hostGpuUtil;
    private static Gauge hostPowerWatt;
    // T8.2 — Host state (numeric: 0=SUSPENDED, 1=IDLE, 2=ACTIVE).
    private static Gauge hostState;

    // Episode-scoped gauges (reset by SimulationManager on each new episode).
    private static Gauge totalEnergyKwh;
    private static Gauge pendingTasks;
    private static Gauge simClockSeconds;
    // T8.2 — Cluster-wide state-count gauges, useful for "how many hosts off?"
    private static Gauge hostStateCount;   // labels: state, scenario, scheduler

    // Cumulative counters (lifetime of JVM — use rate()/increase() in PromQL).
    private static Counter slaViolationsTotal;
    private static Counter tasksScheduledTotal;
    // T8.2 — Cumulative SUSPENDED → ACTIVE transitions.
    private static Counter wakeupsTotal;

    // Context labels (updated atomically from gateway thread).
    private static volatile String scenarioTag  = "unknown";
    private static volatile String schedulerTag = "rl";

    // ── Lifecycle ─────────────────────────────────────────────────────────

    /**
     * Start the Prometheus HTTP endpoint if {@code MONITORING_ENABLED=true}.
     *
     * <p>Idempotent: subsequent calls are no-ops once the server is running.
     * Any failure to bind or register is logged and swallowed — the
     * simulation continues without monitoring.
     */
    public static synchronized void start() {
        if (enabled || server != null) return; // already started

        String env = System.getenv("MONITORING_ENABLED");
        if (env == null || !"true".equalsIgnoreCase(env.trim())) {
            System.out.println(
                "[MetricsRegistry] Monitoring disabled "
              + "(set MONITORING_ENABLED=true to enable Prometheus exporter).");
            return;
        }

        int port;
        try {
            port = Integer.parseInt(
                    System.getenv().getOrDefault("METRICS_PORT", "9091"));
        } catch (NumberFormatException nfe) {
            System.err.println(
                "[MetricsRegistry] Invalid METRICS_PORT — falling back to 9091.");
            port = 9091;
        }

        try {
            registerMetrics();
            server = new HTTPServer.Builder().withPort(port).build();
            enabled = true;
            System.out.printf(
                "[MetricsRegistry] Prometheus exporter listening on 0.0.0.0:%d%n",
                port);
        } catch (Throwable t) {
            // Catch Throwable — even an OOM or LinkageError must not kill main.
            System.err.printf(
                "[MetricsRegistry] Failed to start exporter: %s — "
              + "continuing without monitoring.%n", t);
            enabled = false;
            safeStop();
        }
    }

    /** Stop the HTTP endpoint and clear registered metrics. */
    public static synchronized void stop() {
        safeStop();
    }

    private static void safeStop() {
        enabled = false;
        try {
            if (server != null) server.close();
        } catch (Throwable t) {
            System.err.printf("[MetricsRegistry] Error closing server: %s%n", t);
        } finally {
            server = null;
        }
        // Clear default registry so a subsequent start() can re-register.
        try {
            CollectorRegistry.defaultRegistry.clear();
        } catch (Throwable ignored) { /* defensive */ }
        hostCpuUtil = hostMemUtil = hostGpuUtil = hostPowerWatt = null;
        hostState = null;
        totalEnergyKwh = pendingTasks = simClockSeconds = null;
        hostStateCount = null;
        slaViolationsTotal = tasksScheduledTotal = null;
        wakeupsTotal = null;
    }

    /** Whether the exporter is currently active. Useful for tests. */
    public static boolean isEnabled() { return enabled; }

    /** Effective port the HTTP server is bound to, or -1 if not running. */
    public static int port() {
        return (server != null) ? server.getPort() : -1;
    }

    // ── Metric registration ───────────────────────────────────────────────

    private static void registerMetrics() {
        // G2.1 — `sku` identifies the hardware class (gpu-heavy / balanced /
        // cpu-only, or "homogeneous"). Without it Grafana cannot separate a
        // GPU-heavy host's 500 W idle draw from a CPU-only host's 60 W, which
        // is the whole point of the heterogeneous topology. Cardinality cost is
        // nil: sku is functionally determined by host_id.
        String[] hostLabels = {"host_id", "sku", "scenario", "scheduler"};
        String[] ctxLabels  = {"scenario", "scheduler"};

        hostCpuUtil = Gauge.build()
            .name("eldas_host_cpu_util")
            .help("Per-host CPU utilisation, range [0, 1].")
            .labelNames(hostLabels).register();

        hostMemUtil = Gauge.build()
            .name("eldas_host_mem_util")
            .help("Per-host memory utilisation, range [0, 1].")
            .labelNames(hostLabels).register();

        hostGpuUtil = Gauge.build()
            .name("eldas_host_gpu_util")
            .help("Per-host GPU utilisation, range [0, 1].")
            .labelNames(hostLabels).register();

        hostPowerWatt = Gauge.build()
            .name("eldas_host_power_watt")
            .help("Per-host instantaneous power draw (CPU + GPU), Watts.")
            .labelNames(hostLabels).register();

        // T8.2 — Host state code. Plot as a step function in Grafana to
        // visualise the ACTIVE / IDLE / SUSPENDED timeline per host.
        hostState = Gauge.build()
            .name("eldas_host_state")
            .help("Per-host state: 0=SUSPENDED, 1=IDLE, 2=ACTIVE.")
            .labelNames(hostLabels).register();

        totalEnergyKwh = Gauge.build()
            .name("eldas_total_energy_kwh")
            .help("Cumulative energy consumed in the current episode (kWh). "
                + "Reset to 0 by SimulationManager on each resetSimulation().")
            .labelNames(ctxLabels).register();

        pendingTasks = Gauge.build()
            .name("eldas_pending_tasks")
            .help("Number of tasks remaining to be scheduled in the current episode.")
            .labelNames(ctxLabels).register();

        simClockSeconds = Gauge.build()
            .name("eldas_sim_clock_seconds")
            .help("Simulated clock time (seconds). Use vs wall-time to spot stalls.")
            .labelNames(ctxLabels).register();

        // T8.2 — Per-state cluster count (labels.state ∈ {active, idle, suspended}).
        // Stacked area in Grafana = instant "how many hosts are off" view.
        hostStateCount = Gauge.build()
            .name("eldas_host_state_count")
            .help("Number of hosts in each state at the current snapshot.")
            .labelNames("state", "scenario", "scheduler").register();

        slaViolationsTotal = Counter.build()
            .name("eldas_sla_violations_total")
            .help("Cumulative SLA deadline violations since JVM start. "
                + "Monotonic — use increase() / rate() in PromQL.")
            .labelNames("scenario", "scheduler", "qos").register();

        tasksScheduledTotal = Counter.build()
            .name("eldas_tasks_scheduled_total")
            .help("Cumulative successful task placements since JVM start.")
            .labelNames("scenario", "scheduler").register();

        // T8.2 — SUSPENDED → ACTIVE transitions. Use increase() in PromQL
        // to see wake-up rate per scenario/scheduler.
        wakeupsTotal = Counter.build()
            .name("eldas_total_wakeups_total")
            .help("Cumulative SUSPENDED→ACTIVE host wake-up events.")
            .labelNames("scenario", "scheduler").register();
    }

    // ── Context (called from GatewayEntryPoint) ───────────────────────────

    /**
     * Update the scenario + scheduler tags applied to all subsequent metric
     * writes. Safe to call from any thread; volatile writes propagate to the
     * sim thread without locking.
     */
    public static void setContext(String scenario, String scheduler) {
        if (scenario  != null) scenarioTag  = scenario.toUpperCase(Locale.ROOT);
        if (scheduler != null) schedulerTag = scheduler.toLowerCase(Locale.ROOT);
    }

    /** Update only the scheduler tag (used by selectBaselineAction). */
    public static void setSchedulerTag(String scheduler) {
        if (scheduler != null) schedulerTag = scheduler.toLowerCase(Locale.ROOT);
    }

    public static String scenarioTag()  { return scenarioTag; }
    public static String schedulerTag() { return schedulerTag; }

    // ── Metric mutators (each is a no-op when disabled, never throws) ─────

    /**
     * Reset per-episode gauges to 0 for the current context. Counters are
     * untouched (Prometheus semantics: counters never decrease).
     */
    public static void onEpisodeStart() {
        if (!enabled) return;
        try {
            totalEnergyKwh.labels(scenarioTag, schedulerTag).set(0);
            pendingTasks  .labels(scenarioTag, schedulerTag).set(0);
            simClockSeconds.labels(scenarioTag, schedulerTag).set(0);
        } catch (Throwable t) { logSwallow("onEpisodeStart", t); }
    }

    /** Legacy 5-arg form — labels the host as {@code sku="homogeneous"}. */
    public static void recordHost(int hostIdx,
                                  double cpuUtil, double memUtil,
                                  double gpuUtil, double powerWatt) {
        recordHost(hostIdx, "homogeneous", cpuUtil, memUtil, gpuUtil, powerWatt);
    }

    public static void recordHost(int hostIdx, String sku,
                                  double cpuUtil, double memUtil,
                                  double gpuUtil, double powerWatt) {
        if (!enabled) return;
        try {
            String hid = Integer.toString(hostIdx);
            String s   = skuOrDefault(sku);
            hostCpuUtil  .labels(hid, s, scenarioTag, schedulerTag).set(clamp01(cpuUtil));
            hostMemUtil  .labels(hid, s, scenarioTag, schedulerTag).set(clamp01(memUtil));
            hostGpuUtil  .labels(hid, s, scenarioTag, schedulerTag).set(clamp01(gpuUtil));
            hostPowerWatt.labels(hid, s, scenarioTag, schedulerTag).set(Math.max(0, powerWatt));
        } catch (Throwable t) { logSwallow("recordHost", t); }
    }

    public static void setTotalEnergyKwh(double kwh) {
        if (!enabled) return;
        try {
            totalEnergyKwh.labels(scenarioTag, schedulerTag).set(Math.max(0, kwh));
        } catch (Throwable t) { logSwallow("setTotalEnergyKwh", t); }
    }

    public static void setPendingTasks(int count) {
        if (!enabled) return;
        try {
            pendingTasks.labels(scenarioTag, schedulerTag).set(Math.max(0, count));
        } catch (Throwable t) { logSwallow("setPendingTasks", t); }
    }

    public static void setSimClock(double seconds) {
        if (!enabled) return;
        try {
            simClockSeconds.labels(scenarioTag, schedulerTag).set(Math.max(0, seconds));
        } catch (Throwable t) { logSwallow("setSimClock", t); }
    }

    public static void incTasksScheduled() {
        if (!enabled) return;
        try {
            tasksScheduledTotal.labels(scenarioTag, schedulerTag).inc();
        } catch (Throwable t) { logSwallow("incTasksScheduled", t); }
    }

    /** T8.2 — Set per-host state code (0/1/2). No-op when disabled. */
    public static void setHostState(int hostIdx, int stateCode) {
        setHostState(hostIdx, "homogeneous", stateCode);
    }

    public static void setHostState(int hostIdx, String sku, int stateCode) {
        if (!enabled) return;
        try {
            hostState.labels(Integer.toString(hostIdx), skuOrDefault(sku),
                             scenarioTag, schedulerTag)
                     .set(stateCode);
        } catch (Throwable t) { logSwallow("setHostState", t); }
    }

    /**
     * T8.2 — Update the cluster-wide state-count snapshot. Called once per
     * simulation snapshot; the three values together must equal the host count
     * (the registry does not enforce this — caller is trusted).
     */
    public static void setHostStateCounts(int active, int idle, int suspended) {
        if (!enabled) return;
        try {
            hostStateCount.labels("active",    scenarioTag, schedulerTag).set(active);
            hostStateCount.labels("idle",      scenarioTag, schedulerTag).set(idle);
            hostStateCount.labels("suspended", scenarioTag, schedulerTag).set(suspended);
        } catch (Throwable t) { logSwallow("setHostStateCounts", t); }
    }

    /** T8.2 — Increment the cumulative wake-up counter. */
    public static void incWakeup() {
        if (!enabled) return;
        try {
            wakeupsTotal.labels(scenarioTag, schedulerTag).inc();
        } catch (Throwable t) { logSwallow("incWakeup", t); }
    }

    public static void incSlaViolation(String qos) {
        if (!enabled) return;
        try {
            String q = (qos != null && !qos.isBlank()) ? qos : "unknown";
            slaViolationsTotal.labels(scenarioTag, schedulerTag, q).inc();
        } catch (Throwable t) { logSwallow("incSlaViolation", t); }
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    /** Prometheus rejects null label values — never let one through. */
    private static String skuOrDefault(String sku) {
        return (sku != null && !sku.isBlank()) ? sku : "unknown";
    }

    private static double clamp01(double v) {
        if (Double.isNaN(v) || v < 0) return 0;
        return Math.min(1.0, v);
    }

    private static void logSwallow(String op, Throwable t) {
        // Throttle log spam: print only once per minute per op would be nicer,
        // but a single line per failure is acceptable in practice.
        System.err.printf("[MetricsRegistry] %s failed: %s (swallowed)%n", op, t);
    }
}
