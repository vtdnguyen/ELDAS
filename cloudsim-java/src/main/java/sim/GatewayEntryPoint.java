package sim;

import py4j.GatewayServer;

import sim.AlibabaTraceReader.TaskRecord;
import sim.DatacenterFactory.GpuState;
import sim.ScenarioFilter.Scenario;
import sim.SimulationManager.StepResult;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.*;

/**
 * T3.1 — Py4J gateway entry point exposing {@link SimulationManager} to Python.
 *
 * <p>Python connects via:
 * <pre>
 *   from py4j.java_gateway import JavaGateway
 *   gw  = JavaGateway(address="cloudsim-java", port=25333)
 *   ep  = gw.entry_point
 *   res = ep.reset("HIGH", 42)
 *   obs = list(res.observation())   # double[] → Python list
 * </pre>
 *
 * <p>All public methods return Py4J-friendly types: primitives, arrays,
 * and Java objects whose methods Python can call transparently.
 *
 * <p><b>Baseline evaluation</b> is supported via {@link #selectBaselineAction(String)}:
 * Python calls this to get a host index from a non-RL policy, then passes it
 * to {@link #step(int)}.
 */
public class GatewayEntryPoint {

    private SimulationManager manager;

    // Baseline policy instances (rebuilt on each reset)
    private VmAllocationPolicyK8sDefault    k8sPolicy;
    private VmAllocationPolicyRandom        randomPolicy;
    // T8.4 / T8.5 / T8.6 — three classical baselines added in Phase 1.8.
    private VmAllocationPolicyFirstFit      firstFitPolicy;
    private VmAllocationPolicyBestFit       bestFitPolicy;
    private VmAllocationPolicyRoundRobin    roundRobinPolicy;

    // ════════════════════════════════════════════════════════════════════════
    //  LIFECYCLE
    // ════════════════════════════════════════════════════════════════════════

    /**
     * (Re)initialise the simulation and return the first {@link StepResult}.
     *
     * @param scenarioName "LOW", "HIGH", or "BURST"
     * @param seed         random seed for reproducibility
     * @return StepResult with initial observation (reward = [0,0], done = false)
     */
    public StepResult reset(String scenarioName, long seed) {
        Scenario scenario = Scenario.valueOf(scenarioName.toUpperCase(Locale.ROOT));

        // T5.3 — Tag the upcoming episode for Prometheus. Scheduler defaults
        // to "rl"; selectBaselineAction overrides it if a baseline drives the
        // run. Safe no-op when monitoring is disabled.
        MetricsRegistry.setContext(scenario.name(), "rl");

        // Reclaim the PREVIOUS episode's manager before replacing it.
        //
        // This reset builds a *new* SimulationManager (see below), so the old
        // one's terminateExisting() never runs — the new instance's simThread is
        // null, and the old stepping thread is left blocked forever on an
        // actionQueue nobody will ever write to. Being a daemon it never blocks
        // JVM exit, so it stays invisible: measured +1 live "cloudsim-step"
        // thread and ~1.4 MB RSS *per reset* (60 resets ⇒ 94→154 threads,
        // 256→340 MB), because each stranded thread also pins its whole CloudSim
        // graph against GC. Harmless over a handful of episodes; fatal for a
        // budget sweep that runs hundreds of episodes across dozens of runs.
        //
        // shutdown() interrupts the stepping thread and terminates that
        // episode's CloudSim instance. Episodes are independent, so reclaiming
        // the finished one cannot affect any result.
        if (manager != null) {
            manager.shutdown();
        }

        // T7.1 — Use the env-driven constructor so NUM_HOSTS / VCPU_PER_HOST /
        // GPU_PER_HOST / RAM_PER_HOST_GB (and T8.1 state-machine knobs) are
        // re-read on every resetSimulation(). The fromEnv() call inside
        // SimulationManager.buildSimulation handles parsing and fallbacks.
        manager = new SimulationManager(
                SimulationConfig.TRACE_FILE, scenario, seed);

        StepResult result = manager.resetSimulation();

        // Rebuild baseline policies. They read live resource state through
        // the manager (see java-validation-report §B1), not from CloudSim's
        // Host API which always reports full capacity. RoundRobin is also
        // state-ful (rotating pointer) — re-creating it on every reset gives
        // a fresh pointer at index 0, matching reproducibility expectations.
        k8sPolicy        = new VmAllocationPolicyK8sDefault(manager);
        randomPolicy     = new VmAllocationPolicyRandom(manager, seed);
        firstFitPolicy   = new VmAllocationPolicyFirstFit(manager);
        bestFitPolicy    = new VmAllocationPolicyBestFit(manager);
        roundRobinPolicy = new VmAllocationPolicyRoundRobin(manager);

        System.out.printf("[GatewayEntryPoint] Reset: scenario=%s, seed=%d, "
                        + "hosts=%d, tasks=%d%n",
                scenario, seed, manager.getHostCount(), manager.getTasks().size());

        return result;
    }

    /**
     * Convenience reset with defaults (HIGH scenario, seed from env or 42).
     */
    public StepResult reset() {
        long seed = Long.parseLong(
                System.getenv().getOrDefault("RANDOM_SEED", "42"));
        return reset("HIGH", seed);
    }

    /**
     * Submit the RL agent's action and receive the next {@link StepResult}.
     *
     * @param hostIndex 0-based index into the host list
     * @return StepResult with observation, reward, and done flag
     */
    public StepResult step(int hostIndex) {
        requireManager();
        return manager.step(hostIndex);
    }

    /**
     * SYS.2 — No-op round-trip probe used to measure raw Py4J RPC latency.
     *
     * <p>Does no simulation work whatsoever, so the wall-clock time Python
     * measures around this call is (by construction) the pure JVM↔Python
     * round-trip cost. Subtracting it from the measured {@link #step(int)} time
     * decomposes a step into <em>transport</em> vs <em>simulation</em> without
     * needing any timing instrumentation on the Java side.
     *
     * <p>This is the evidence gate for SYS.2: batching several {@code step}
     * calls into one RPC only pays off if transport is a real share of the step
     * budget. See {@code rl-agent/src/perf/profile_step.py}.
     *
     * @return the argument, echoed back (keeps the call from being optimised away)
     */
    public int ping(int value) {
        return value;
    }

    /**
     * SYS.2 — {@link #step(int)} with the whole payload packed into one blob.
     *
     * <p>Returns exactly the same numbers as {@link #step(int)}; the difference
     * is purely transport. Py4J passes {@code byte[]} by value, so Python gets
     * the observation, reward, cost, done flag, task index, task name AND the
     * next action mask in a <b>single</b> round trip instead of ~76 (one per
     * array element — see {@link StepCodec} for the measurement). Python decodes
     * it with {@code state_builder.decode_packed}.
     *
     * @param hostIndex 0-based index into the host list
     * @return {@link StepCodec} blob for the resulting state
     */
    public byte[] stepPacked(int hostIndex) {
        requireManager();
        StepResult result = manager.step(hostIndex);
        return StepCodec.encode(result, manager.getActionMask());
    }

    /**
     * SYS.2 — {@link #reset(String, long)} returning a {@link StepCodec} blob.
     * Same values as {@code reset}, one round trip.
     */
    public byte[] resetPacked(String scenarioName, long seed) {
        StepResult result = reset(scenarioName, seed);
        return StepCodec.encode(result, manager.getActionMask());
    }

    /**
     * SYS.2 — Capability probe. Lets Python detect the packed transport at
     * connect time and fall back to the per-element path against an older
     * gateway (or a test double) instead of crashing on a missing method.
     */
    public boolean supportsPackedTransport() {
        return true;
    }

    /**
     * Shut down the simulation cleanly.
     * Called from Python or from the JVM shutdown hook.
     */
    public void shutdown() {
        if (manager != null) {
            manager.shutdown();
        }
        System.out.println("[GatewayEntryPoint] Shutdown complete.");
    }

    // ════════════════════════════════════════════════════════════════════════
    //  STATE QUERIES
    // ════════════════════════════════════════════════════════════════════════

    /**
     * Action mask: {@code true} at index {@code i} if host {@code i} can
     * accept the current task.
     */
    public boolean[] getActionMask() {
        requireManager();
        return manager.getActionMask();
    }

    /** Number of hosts = size of the discrete action space. */
    public int getActionSize() {
        requireManager();
        return manager.getHostCount();
    }

    /** Length of the observation vector ({@code 6H + 4}, T8.9). */
    public int getObservationSize() {
        requireManager();
        return 6 * manager.getHostCount() + 4;
    }

    /** Whether the current episode has finished. */
    public boolean isDone() {
        requireManager();
        return manager.isDone();
    }

    /** Total energy consumed so far (kWh). */
    public double getTotalEnergyKwh() {
        requireManager();
        return manager.getTotalEnergyKwh();
    }

    /**
     * G1.1 — Episode-cumulative Constrained-MDP constraint cost
     * {@code C_SLA = Σ κ·max(0, completion − deadline)} (≥ 0). Consumed by the
     * Python CMDP layer (PID-Lagrangian) as the constrained quantity
     * {@code E[C_SLA] ≤ d}.
     */
    public double getSlaCost() {
        requireManager();
        return manager.getSlaCost();
    }

    /** Number of tasks in the current episode. */
    public int getTaskCount() {
        requireManager();
        return manager.getTasks().size();
    }

    /** Index of the current task being scheduled. */
    public int getCurrentTaskIndex() {
        requireManager();
        return manager.getCurrentTaskIndex();
    }

    // ════════════════════════════════════════════════════════════════════════
    //  BASELINE ACTION SELECTION
    // ════════════════════════════════════════════════════════════════════════

    /**
     * Ask a baseline policy to select a host for the current task.
     * Used by {@code baseline_eval.py} to run episodes with non-RL schedulers.
     *
     * <p>Supported policy names (case-insensitive):
     * <ul>
     *   <li>{@code "k8s"} — Kubernetes default (Filter + LeastRequestedPriority)</li>
     *   <li>{@code "random"} — uniform random among feasible hosts</li>
     *   <li>{@code "firstfit"} — first feasible host in index order (T8.4)</li>
     *   <li>{@code "bestfit"} — pack tightest-fit feasible host (T8.5)</li>
     *   <li>{@code "roundrobin"} — rotate through hosts in index order (T8.6)</li>
     * </ul>
     *
     * @param policyName policy identifier (case-insensitive)
     * @return 0-based host index
     * @throws IllegalArgumentException if policyName is unknown
     */
    public int selectBaselineAction(String policyName) {
        requireManager();

        int taskIdx = manager.getCurrentTaskIndex();
        if (taskIdx >= manager.getTasks().size()) {
            return 0;
        }
        TaskRecord task = manager.getTasks().get(taskIdx);
        List<org.cloudsimplus.hosts.Host> hosts = manager.getHosts();

        // T5.3 — Reflect the active baseline policy in Prometheus labels.
        // Called every step; the underlying assignment is a cheap volatile write.
        MetricsRegistry.setSchedulerTag(policyName);

        return switch (policyName.toLowerCase(Locale.ROOT)) {
            case "k8s"        -> k8sPolicy       .selectHostForTask(hosts, task);
            case "random"     -> randomPolicy    .selectHostForTask(hosts, task);
            case "firstfit"   -> firstFitPolicy  .selectHostForTask(hosts, task);
            case "bestfit"    -> bestFitPolicy   .selectHostForTask(hosts, task);
            case "roundrobin" -> roundRobinPolicy.selectHostForTask(hosts, task);
            default -> throw new IllegalArgumentException(
                    "Unknown policy: " + policyName
                  + " (expected 'k8s', 'random', 'firstfit', 'bestfit', or 'roundrobin')");
        };
    }

    /**
     * Override the scheduler tag used by the Prometheus exporter. Useful when
     * Python drives a custom policy (e.g. a trained MORL agent) and wants to
     * distinguish its runs from baselines in Grafana. Safe no-op when
     * monitoring is disabled.
     */
    public void setSchedulerTag(String tag) {
        MetricsRegistry.setSchedulerTag(tag);
    }

    // ════════════════════════════════════════════════════════════════════════
    //  METRICS EXPORT
    // ════════════════════════════════════════════════════════════════════════

    /**
     * Export CSV + JSON metrics to the given directory.
     * Call after the episode is done.
     */
    public void exportMetrics(String outputDir) {
        requireManager();
        manager.exportMetrics(outputDir);
    }

    // ════════════════════════════════════════════════════════════════════════
    //  INTERNAL
    // ════════════════════════════════════════════════════════════════════════

    private void requireManager() {
        if (manager == null) {
            throw new IllegalStateException("Call reset() before using the gateway");
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  GATEWAY SERVER BOOTSTRAP
    // ════════════════════════════════════════════════════════════════════════

    /**
     * Start the Py4J {@link GatewayServer} on the configured port.
     * Binds to {@code 0.0.0.0} so that the rl-agent Docker container
     * can reach the gateway over the bridge network.
     *
     * @param entryPoint the entry point object exposed to Python
     * @return the running GatewayServer instance
     */
    public static GatewayServer startServer(GatewayEntryPoint entryPoint) {
        int port = Integer.parseInt(
                System.getenv().getOrDefault("PY4J_PORT", "25333"));
        return startServer(entryPoint, port);
    }

    /**
     * SYS.1 — Start a GatewayServer on an explicit port.
     *
     * <p>Used by {@link Main} to bring up {@code NUM_GATEWAYS} independent
     * gateways in a single JVM (ports {@code PY4J_PORT … PY4J_PORT+N−1}), one
     * per {@code SubprocVecEnv} worker. This is safe because
     * {@link SimulationManager} holds <b>no static state</b> — every gateway
     * owns its own manager, CloudSim instance, host list, energy accumulators
     * and stepping thread, so the simulations cannot interfere.
     *
     * <p><b>Caveat:</b> {@link MetricsRegistry} <i>is</i> a static façade. With
     * more than one gateway its gauges are written by every simulation at once
     * and become meaningless (they are not labelled per gateway). Monitoring is
     * opt-in and off the critical path (CLAUDE.md Lưu ý #16), so {@code Main}
     * warns rather than failing — but do not read Grafana during a parallel run.
     *
     * @param entryPoint the entry point object exposed to Python
     * @param port       TCP port to bind
     * @return the running GatewayServer instance
     */
    public static GatewayServer startServer(GatewayEntryPoint entryPoint, int port) {
        InetAddress bindAddress;
        try {
            bindAddress = InetAddress.getByName("0.0.0.0");
        } catch (UnknownHostException e) {
            // "0.0.0.0" is always valid — this cannot happen
            throw new RuntimeException(e);
        }

        GatewayServer server = new GatewayServer.GatewayServerBuilder(entryPoint)
                .javaPort(port)
                .javaAddress(bindAddress)
                .build();

        server.start();

        System.out.printf("[GatewayEntryPoint] Py4J GatewayServer listening on 0.0.0.0:%d%n",
                port);
        return server;
    }
}
