package sim;

import org.cloudsimplus.hosts.Host;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * T2.4 / T8.2 — Exports energy and SLA metrics to CSV and JSON.
 *
 * <p>Two export formats are supported:
 * <ul>
 *   <li><b>CSV</b> — one row per metric snapshot, per-host columns appended
 *       dynamically (one {@code h{i}_cpu} and {@code h{i}_state} per host)
 *       so the post-hoc diagnostic notebook can plot per-host timelines
 *       without round-tripping through Prometheus.</li>
 *   <li><b>JSON</b> — single summary object, easy to consume from Python or dashboards</li>
 * </ul>
 */
public final class MetricsExporter {

    // ── Metric snapshot (one per simulation tick or per task) ──────────────

    public record Snapshot(
        double timestamp,           // simulation time (seconds)
        double totalCpuEnergyKwh,   // cumulative CPU energy (kWh)
        double totalGpuEnergyKwh,   // cumulative GPU energy (kWh)
        double totalEnergyKwh,      // CPU + GPU
        int    tasksScheduled,      // tasks scheduled so far
        int    slaViolations,       // tasks that missed their deadline
        double avgSlaSlack,         // mean (deadline − completion), negative = late
        double clusterCpuUtil,      // average CPU utilisation across all hosts
        double clusterGpuUtil,      // average GPU utilisation across all hosts
        // ── T8.1 / T8.2: host state machine telemetry ────────────────────
        int    activeHosts,         // hosts with U > 0
        int    idleHosts,           // hosts with U = 0 since < IDLE_THRESHOLD
        int    suspendedHosts,      // hosts with U = 0 since ≥ IDLE_THRESHOLD
        int    totalWakeups,        // cumulative SUSPENDED → ACTIVE transitions
        double wakeEnergyKwh,       // cumulative one-shot wake-up energy
        double[] hostCpuUtil,       // per-host CPU utilisation (length = host count)
        int[]    hostState          // per-host state code: 0=SUSPENDED, 1=IDLE, 2=ACTIVE
    ) {}

    // ── Summary (end-of-simulation aggregate) ─────────────────────────────

    public record Summary(
        double totalEnergyKwh,
        double totalCpuEnergyKwh,
        double totalGpuEnergyKwh,
        int    totalTasks,
        int    slaViolations,
        double slaViolationRate,
        double avgCpuUtil,
        double avgGpuUtil,
        double makespan,                  // wall-clock length of simulation (sec)
        // ── T8.2 additions ────────────────────────────────────────────
        int    totalWakeups,
        double wakeEnergyKwh,
        double avgActiveHosts,
        double avgIdleHosts,
        double avgSuspendedHosts
    ) {}

    // ── CSV export ─────────────────────────────────────────────────────────

    private static final String CSV_BASE_HEADER =
            "timestamp,cpu_energy_kwh,gpu_energy_kwh,total_energy_kwh,"
          + "tasks_scheduled,sla_violations,avg_sla_slack,"
          + "cluster_cpu_util,cluster_gpu_util,"
          + "active_hosts,idle_hosts,suspended_hosts,"
          + "total_wakeups,wake_energy_kwh";

    /**
     * Write a list of snapshots to a CSV file. The header is extended with
     * per-host columns ({@code h{i}_cpu_util}, {@code h{i}_state}) based on
     * the first snapshot's host array length. Creates parent directories
     * if needed.
     */
    public static void writeCsv(String filePath, List<Snapshot> snapshots) throws IOException {
        Path path = Path.of(filePath);
        Files.createDirectories(path.getParent());

        int hostCount = snapshots.isEmpty() || snapshots.getFirst().hostCpuUtil() == null
                ? 0 : snapshots.getFirst().hostCpuUtil().length;

        try (BufferedWriter w = Files.newBufferedWriter(path)) {
            w.write(CSV_BASE_HEADER);
            for (int i = 0; i < hostCount; i++) {
                w.write(",h" + i + "_cpu_util,h" + i + "_state");
            }
            w.newLine();

            for (Snapshot s : snapshots) {
                w.write(String.format(Locale.US,
                        "%.1f,%.6f,%.6f,%.6f,%d,%d,%.2f,%.4f,%.4f,%d,%d,%d,%d,%.6f",
                        s.timestamp, s.totalCpuEnergyKwh, s.totalGpuEnergyKwh,
                        s.totalEnergyKwh, s.tasksScheduled, s.slaViolations,
                        s.avgSlaSlack, s.clusterCpuUtil, s.clusterGpuUtil,
                        s.activeHosts, s.idleHosts, s.suspendedHosts,
                        s.totalWakeups, s.wakeEnergyKwh));
                double[] util  = s.hostCpuUtil();
                int[]    state = s.hostState();
                for (int i = 0; i < hostCount; i++) {
                    // Defensive: tolerate snapshots from mid-reset edge cases
                    // where the host array hasn't been (re)populated yet.
                    double u = (util  != null && i < util .length) ? util [i] : 0.0;
                    int    st= (state != null && i < state.length) ? state[i] : 1;
                    w.write(String.format(Locale.US, ",%.4f,%d", u, st));
                }
                w.newLine();
            }
        }

        System.out.printf("[MetricsExporter] CSV written: %s (%d rows, %d hosts)%n",
                filePath, snapshots.size(), hostCount);
    }

    // ── JSON export ────────────────────────────────────────────────────────

    /**
     * Write an end-of-simulation summary to a JSON file.
     * Hand-rolled JSON to avoid adding a dependency (jackson / gson).
     */
    public static void writeJson(String filePath, Summary summary) throws IOException {
        Path path = Path.of(filePath);
        Files.createDirectories(path.getParent());

        String json = String.format(Locale.US, """
                {
                  "total_energy_kwh":     %.6f,
                  "cpu_energy_kwh":       %.6f,
                  "gpu_energy_kwh":       %.6f,
                  "total_tasks":          %d,
                  "sla_violations":       %d,
                  "sla_violation_rate":   %.4f,
                  "avg_cpu_utilization":  %.4f,
                  "avg_gpu_utilization":  %.4f,
                  "makespan_sec":         %.1f,
                  "total_wakeups":        %d,
                  "wake_energy_kwh":      %.6f,
                  "avg_active_hosts":     %.4f,
                  "avg_idle_hosts":       %.4f,
                  "avg_suspended_hosts":  %.4f
                }
                """,
                summary.totalEnergyKwh, summary.totalCpuEnergyKwh,
                summary.totalGpuEnergyKwh, summary.totalTasks,
                summary.slaViolations, summary.slaViolationRate,
                summary.avgCpuUtil, summary.avgGpuUtil, summary.makespan,
                summary.totalWakeups, summary.wakeEnergyKwh,
                summary.avgActiveHosts, summary.avgIdleHosts, summary.avgSuspendedHosts);

        Files.writeString(path, json);
        System.out.printf("[MetricsExporter] JSON written: %s%n", filePath);
    }

    // ── Utility: build a summary from final state ─────────────────────────

    /**
     * Convenience method to build a {@link Summary} from a list of snapshots.
     * Takes the last snapshot as the final cumulative state and averages
     * across all snapshots for utilisation / state-count metrics.
     */
    public static Summary buildSummary(List<Snapshot> snapshots) {
        if (snapshots.isEmpty()) {
            return new Summary(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        }

        Snapshot last = snapshots.getLast();
        double avgCpuUtil   = mean(snapshots, Snapshot::clusterCpuUtil);
        double avgGpuUtil   = mean(snapshots, Snapshot::clusterGpuUtil);
        double avgActive    = mean(snapshots, s -> (double) s.activeHosts());
        double avgIdle      = mean(snapshots, s -> (double) s.idleHosts());
        double avgSuspended = mean(snapshots, s -> (double) s.suspendedHosts());

        double violationRate = last.tasksScheduled == 0
                ? 0.0
                : (double) last.slaViolations / last.tasksScheduled;

        return new Summary(
                last.totalEnergyKwh, last.totalCpuEnergyKwh, last.totalGpuEnergyKwh,
                last.tasksScheduled, last.slaViolations, violationRate,
                avgCpuUtil, avgGpuUtil, last.timestamp,
                last.totalWakeups, last.wakeEnergyKwh,
                avgActive, avgIdle, avgSuspended);
    }

    private static double mean(List<Snapshot> snapshots,
                               java.util.function.ToDoubleFunction<Snapshot> f) {
        return snapshots.stream().mapToDouble(f).average().orElse(0.0);
    }

    // ── Utility: compute cluster-wide utilisation ─────────────────────────

    public static double averageCpuUtilization(List<Host> hosts) {
        return hosts.stream()
                .mapToDouble(h -> h.getCpuPercentUtilization())
                .average()
                .orElse(0.0);
    }

    public static double averageGpuUtilization(Map<Host, DatacenterFactory.GpuState> gpuMap) {
        if (gpuMap.isEmpty()) return 0.0;
        return gpuMap.values().stream()
                .mapToDouble(DatacenterFactory.GpuState::utilization)
                .average()
                .orElse(0.0);
    }

    private MetricsExporter() {} // utility class
}
