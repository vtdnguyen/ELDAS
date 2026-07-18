package sim;

import py4j.GatewayServer;

import java.util.ArrayList;
import java.util.List;

/**
 * Application entry point.
 *
 * <p>Starts one or more Py4J {@link GatewayServer}s exposing
 * {@link GatewayEntryPoint} to the Python rl-agent container.
 * The process stays alive until the JVM is shut down (e.g. Docker stop).
 *
 * <p><b>SYS.1 — parallel envs.</b> {@code NUM_GATEWAYS} (default 1) controls how
 * many <i>independent</i> gateways this JVM serves, on consecutive ports
 * {@code PY4J_PORT … PY4J_PORT + NUM_GATEWAYS − 1}. Each gateway owns a separate
 * {@link GatewayEntryPoint} → {@link SimulationManager} → CloudSim instance, so
 * a Python {@code SubprocVecEnv} can drive N simulations concurrently with
 * worker {@code i} connecting to port {@code PY4J_PORT + i}.
 *
 * <p>Keeping the default at 1 means the single-gateway topology every Phase-1/
 * Phase-2 result was produced under is completely unchanged unless opted in.
 */
public class Main {

    public static void main(String[] args) throws InterruptedException {
        System.out.println("CloudSim simulation container started.");

        // T5.3 — Optional Prometheus exporter. No-op unless MONITORING_ENABLED=true.
        // Started BEFORE the gateway so the /metrics endpoint is up before any
        // simulation activity. Failures here cannot abort startup — see
        // MetricsRegistry.start() (catches Throwable, logs, returns).
        MetricsRegistry.start();

        final int basePort = Integer.parseInt(
                System.getenv().getOrDefault("PY4J_PORT", "25333"));
        final int numGateways = parseNumGateways();

        // MetricsRegistry is a static façade: its gauges carry no per-gateway
        // label, so N concurrent simulations would all write the same series and
        // produce a meaningless mixture. Monitoring is opt-in and must never
        // abort the run (Lưu ý #16) — warn and carry on.
        if (numGateways > 1 && MetricsRegistry.isEnabled()) {
            System.err.printf("[Main] WARNING: NUM_GATEWAYS=%d with monitoring "
                            + "enabled. MetricsRegistry gauges are global (not "
                            + "labelled per gateway) so Prometheus/Grafana output "
                            + "will interleave %d simulations and is NOT "
                            + "interpretable. Energy/SLA results are unaffected "
                            + "(they are read per-gateway over Py4J).%n",
                    numGateways, numGateways);
        }

        final List<GatewayEntryPoint> entryPoints = new ArrayList<>();
        final List<GatewayServer> servers = new ArrayList<>();

        for (int i = 0; i < numGateways; i++) {
            GatewayEntryPoint entryPoint = new GatewayEntryPoint();
            GatewayServer server = GatewayEntryPoint.startServer(entryPoint, basePort + i);
            entryPoints.add(entryPoint);
            servers.add(server);
        }

        if (numGateways > 1) {
            System.out.printf("[Main] SYS.1: %d independent gateways on ports %d–%d "
                            + "(each with its own SimulationManager).%n",
                    numGateways, basePort, basePort + numGateways - 1);
        }

        // Clean teardown on SIGTERM / Docker stop
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("[Main] Shutting down...");
            for (GatewayEntryPoint ep : entryPoints) {
                try {
                    ep.shutdown();
                } catch (Exception e) {
                    System.err.println("[Main] entry point shutdown failed: " + e);
                }
            }
            for (GatewayServer s : servers) {
                try {
                    s.shutdown();
                } catch (Exception e) {
                    System.err.println("[Main] server shutdown failed: " + e);
                }
            }
            MetricsRegistry.stop();
        }, "shutdown-hook"));

        // Keep process alive — gateways run on their own daemon threads
        Thread.currentThread().join();
    }

    /**
     * Read {@code NUM_GATEWAYS}, falling back to 1 on anything unparseable or
     * out of range. A bad value must not stop the container from serving the
     * default single gateway.
     */
    private static int parseNumGateways() {
        String raw = System.getenv().getOrDefault("NUM_GATEWAYS", "1");
        try {
            int n = Integer.parseInt(raw.trim());
            if (n < 1) {
                System.err.printf("[Main] NUM_GATEWAYS=%s < 1 — using 1.%n", raw);
                return 1;
            }
            return n;
        } catch (NumberFormatException e) {
            System.err.printf("[Main] NUM_GATEWAYS=%s is not an integer — using 1.%n", raw);
            return 1;
        }
    }
}
