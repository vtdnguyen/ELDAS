package sim;

import py4j.GatewayServer;

/**
 * Application entry point.
 *
 * Starts the Py4J {@link GatewayServer} that exposes
 * {@link GatewayEntryPoint} to the Python rl-agent container.
 * The process stays alive until the JVM is shut down (e.g. Docker stop).
 */
public class Main {

    public static void main(String[] args) throws InterruptedException {
        System.out.println("CloudSim simulation container started.");

        // Start Py4J gateway — Python can now connect
        GatewayEntryPoint entryPoint = new GatewayEntryPoint();
        GatewayServer server = GatewayEntryPoint.startServer(entryPoint);

        // Clean teardown on SIGTERM / Docker stop
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("[Main] Shutting down...");
            entryPoint.shutdown();
            server.shutdown();
        }, "shutdown-hook"));

        // Keep process alive — gateway runs on its own daemon threads
        Thread.currentThread().join();
    }
}
