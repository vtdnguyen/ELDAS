package sim;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.stream.Collectors;

/**
 * Standalone test harness for {@link MetricsRegistry}.
 *
 * <p>Follows the same convention as {@link ValidationRunner}: prints
 * {@code [PASS]} / {@code [FAIL]} lines, exits 0 if every check passes,
 * 1 otherwise. No external test framework needed.
 *
 * <p>Run inside the {@code cloudsim-java} image after rebuilding:
 * <pre>
 *   docker compose build cloudsim-java
 *   docker compose run --rm --no-deps --entrypoint java cloudsim-java \
 *       -cp simulation.jar sim.MetricsRegistryTest
 * </pre>
 *
 * <p>Or directly on host with Maven:
 * <pre>
 *   cd cloudsim-java
 *   mvn package -q
 *   java -cp target/simulation-1.0-SNAPSHOT.jar sim.MetricsRegistryTest
 * </pre>
 *
 * <h2>Coverage</h2>
 * <ol>
 *   <li>Disabled by default — no server, mutators are no-ops, no exceptions.</li>
 *   <li>start() with MONITORING_ENABLED=true binds to METRICS_PORT and
 *       serves /metrics with the registered metric names.</li>
 *   <li>start() is idempotent.</li>
 *   <li>setContext + recordHost + incSlaViolation surface in /metrics output.</li>
 *   <li>stop() releases the port; subsequent start() works again.</li>
 *   <li>Bad input (NaN, negative, null QoS) does not throw.</li>
 * </ol>
 */
public final class MetricsRegistryTest {

    private static int passes = 0;
    private static int fails  = 0;

    public static void main(String[] args) throws Exception {
        banner("MetricsRegistry tests");

        try {
            testDisabledByDefault();
            testStartAndScrape();
            testStartIsIdempotent();
            testLabelsInOutput();
            testBadInputDoesNotThrow();
            testStopReleasesPort();
        } finally {
            // Always tear down — leaving the HTTP server running would block
            // subsequent test invocations on the same port.
            MetricsRegistry.stop();
        }

        System.out.printf("%n=== %d passed, %d failed ===%n", passes, fails);
        System.exit(fails == 0 ? 0 : 1);
    }

    // ── Tests ─────────────────────────────────────────────────────────────

    /** Without MONITORING_ENABLED=true, everything must be a silent no-op. */
    private static void testDisabledByDefault() {
        // Make sure nothing is lingering from a previous run inside the same JVM.
        MetricsRegistry.stop();
        // Confirm we cannot start without the env var. We cannot mutate the
        // process environment portably from Java, so this test relies on the
        // ABSENCE of MONITORING_ENABLED in the launcher. If the env var IS
        // set in this JVM, skip — the next test covers the enabled path.
        if ("true".equalsIgnoreCase(
                System.getenv().getOrDefault("MONITORING_ENABLED", ""))) {
            System.out.println("[SKIP] testDisabledByDefault — "
                + "MONITORING_ENABLED is true in this JVM");
            return;
        }
        MetricsRegistry.start();
        check("disabled flag is false after start() with env unset",
              !MetricsRegistry.isEnabled());
        check("port() returns -1 when disabled",
              MetricsRegistry.port() == -1);

        // Every mutator must be safe to call. If any throws, this entire
        // test fails with an uncaught exception → marked FAIL by main.
        MetricsRegistry.setContext("HIGH", "rl");
        MetricsRegistry.onEpisodeStart();
        MetricsRegistry.recordHost(0, 0.5, 0.5, 0.5, 250.0);
        MetricsRegistry.setTotalEnergyKwh(1.2);
        MetricsRegistry.setPendingTasks(42);
        MetricsRegistry.setSimClock(123.4);
        MetricsRegistry.incTasksScheduled();
        MetricsRegistry.incSlaViolation("LS");
        check("mutators are no-op when disabled (no exception thrown)", true);
    }

    /** With MONITORING_ENABLED=true at JVM launch, /metrics must serve. */
    private static void testStartAndScrape() throws IOException {
        // This test only runs when the launcher set MONITORING_ENABLED=true.
        // We use ProcessBuilder-style relaunch in the wrapper script, but for
        // a single-JVM run we can simulate by manually re-enabling via the
        // setter (only available in test code paths if we exposed one) —
        // here we honour the original env-var contract and just skip when off.
        if (!"true".equalsIgnoreCase(
                System.getenv().getOrDefault("MONITORING_ENABLED", ""))) {
            System.out.println("[SKIP] testStartAndScrape — "
                + "set MONITORING_ENABLED=true to run");
            return;
        }
        MetricsRegistry.start();
        check("isEnabled() is true after start()",
              MetricsRegistry.isEnabled());
        check("port() returns a positive number",
              MetricsRegistry.port() > 0);

        // Push a few values so /metrics has content to verify.
        MetricsRegistry.setContext("HIGH", "k8s");
        MetricsRegistry.recordHost(0, 0.3, 0.4, 0.5, 280.0);
        MetricsRegistry.setTotalEnergyKwh(0.123);

        String body = scrape("/metrics");
        check("scrape body includes eldas_host_cpu_util",
              body.contains("eldas_host_cpu_util"));
        check("scrape body includes eldas_total_energy_kwh",
              body.contains("eldas_total_energy_kwh"));
        check("scrape body includes the just-set value 0.123",
              body.contains("0.123"));
    }

    private static void testStartIsIdempotent() {
        if (!MetricsRegistry.isEnabled()) {
            System.out.println("[SKIP] testStartIsIdempotent — exporter not running");
            return;
        }
        int portBefore = MetricsRegistry.port();
        // Second call must NOT re-register metrics (would throw
        // "Collector already registered" inside simpleclient).
        MetricsRegistry.start();
        check("start() is idempotent — exporter still enabled",
              MetricsRegistry.isEnabled());
        check("start() is idempotent — port unchanged",
              MetricsRegistry.port() == portBefore);
    }

    private static void testLabelsInOutput() throws IOException {
        if (!MetricsRegistry.isEnabled()) {
            System.out.println("[SKIP] testLabelsInOutput — exporter not running");
            return;
        }
        MetricsRegistry.setContext("BURST", "random");
        MetricsRegistry.recordHost(7, 0.9, 0.1, 0.0, 350.0);
        MetricsRegistry.incSlaViolation("LS");
        MetricsRegistry.incTasksScheduled();

        String body = scrape("/metrics");
        check("output contains scenario=BURST label",
              body.contains("scenario=\"BURST\""));
        check("output contains scheduler=random label",
              body.contains("scheduler=\"random\""));
        check("output contains host_id=7 series",
              body.contains("host_id=\"7\""));
        check("output contains qos=LS series for SLA counter",
              body.contains("qos=\"LS\""));
    }

    private static void testBadInputDoesNotThrow() {
        // Negative / NaN / null are all valid call shapes — must be clamped
        // or substituted, never thrown.
        MetricsRegistry.recordHost(-1, Double.NaN, -0.5, 2.0, -100.0);
        MetricsRegistry.setTotalEnergyKwh(Double.NaN);
        MetricsRegistry.setSimClock(-1.0);
        MetricsRegistry.setPendingTasks(-5);
        MetricsRegistry.incSlaViolation(null);
        MetricsRegistry.incSlaViolation("");
        MetricsRegistry.setContext(null, null);
        check("bad inputs are handled without exception", true);
    }

    private static void testStopReleasesPort() throws IOException {
        if (!MetricsRegistry.isEnabled()) {
            System.out.println("[SKIP] testStopReleasesPort — exporter not running");
            return;
        }
        int port = MetricsRegistry.port();
        MetricsRegistry.stop();
        check("isEnabled() is false after stop()",
              !MetricsRegistry.isEnabled());

        // A fresh start() on the same port must succeed (default registry
        // was cleared in stop()). Skip if the env var was unset by the time
        // we get here.
        if ("true".equalsIgnoreCase(
                System.getenv().getOrDefault("MONITORING_ENABLED", ""))) {
            MetricsRegistry.start();
            check("re-start after stop() rebinds successfully",
                  MetricsRegistry.isEnabled());
            check("re-bound port matches original",
                  MetricsRegistry.port() == port);
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static String scrape(String path) throws IOException {
        int port = MetricsRegistry.port();
        URI uri = URI.create("http://127.0.0.1:" + port + path);
        HttpURLConnection conn = (HttpURLConnection) uri.toURL().openConnection();
        conn.setRequestMethod("GET");
        conn.setConnectTimeout(2000);
        conn.setReadTimeout(2000);
        int status = conn.getResponseCode();
        if (status != 200) {
            throw new IOException("Unexpected HTTP status: " + status);
        }
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
            return br.lines().collect(Collectors.joining("\n"));
        }
    }

    private static void check(String description, boolean condition) {
        if (condition) {
            passes++;
            System.out.println("[PASS] " + description);
        } else {
            fails++;
            System.out.println("[FAIL] " + description);
        }
    }

    private static void banner(String title) {
        String bar = "═".repeat(Math.max(20, title.length() + 4));
        System.out.println(bar);
        System.out.println("  " + title);
        System.out.println(bar);
    }
}
