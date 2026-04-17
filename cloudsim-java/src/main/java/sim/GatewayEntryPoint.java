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
    private VmAllocationPolicyK8sDefault k8sPolicy;
    private VmAllocationPolicyRandom     randomPolicy;

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

        manager = new SimulationManager(
                SimulationConfig.DEFAULT_DC,
                SimulationConfig.TRACE_FILE,
                scenario,
                seed);

        StepResult result = manager.resetSimulation();

        // Rebuild baseline policies with the fresh GPU registry
        k8sPolicy    = new VmAllocationPolicyK8sDefault(manager.getGpuRegistry());
        randomPolicy = new VmAllocationPolicyRandom(manager.getGpuRegistry(), seed);

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

    /** Length of the observation vector ({@code 3H + 4}). */
    public int getObservationSize() {
        requireManager();
        return 3 * manager.getHostCount() + 4;
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
     * <p>Supported policy names:
     * <ul>
     *   <li>{@code "k8s"} — Kubernetes default (Filter + LeastRequestedPriority)</li>
     *   <li>{@code "random"} — uniform random among feasible hosts</li>
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

        return switch (policyName.toLowerCase(Locale.ROOT)) {
            case "k8s"    -> k8sPolicy.selectHostForTask(manager.getHosts(), task);
            case "random" -> randomPolicy.selectHostForTask(manager.getHosts(), task);
            default -> throw new IllegalArgumentException(
                    "Unknown policy: " + policyName + " (expected 'k8s' or 'random')");
        };
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
