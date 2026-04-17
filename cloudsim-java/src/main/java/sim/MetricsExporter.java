package sim;

import org.cloudsimplus.hosts.Host;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * T2.4 — Exports energy and SLA metrics to CSV and JSON.
 *
 * Two export formats are supported:
 * <ul>
 *   <li><b>CSV</b> — one row per metric snapshot, easy to plot with pandas/matplotlib</li>
 *   <li><b>JSON</b> — single summary object, easy to consume from Python or dashboards</li>
 * </ul>
 *
 * All export methods are static and take the data they need as parameters
 * (no internal state), so they can be called at any point during or after
 * the simulation.
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
        double clusterGpuUtil       // average GPU utilisation across all hosts
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
        double makespan              // wall-clock length of simulation (sec)
    ) {}

    // ── CSV export ─────────────────────────────────────────────────────────

    private static final String CSV_HEADER =
            "timestamp,cpu_energy_kwh,gpu_energy_kwh,total_energy_kwh,"
          + "tasks_scheduled,sla_violations,avg_sla_slack,"
          + "cluster_cpu_util,cluster_gpu_util";

    /**
     * Write a list of snapshots to a CSV file.
     * Creates parent directories if needed.
     */
    public static void writeCsv(String filePath, List<Snapshot> snapshots) throws IOException {
        Path path = Path.of(filePath);
        Files.createDirectories(path.getParent());

        try (BufferedWriter w = Files.newBufferedWriter(path)) {
            w.write(CSV_HEADER);
            w.newLine();

            for (Snapshot s : snapshots) {
                w.write(String.format(Locale.US,
                        "%.1f,%.6f,%.6f,%.6f,%d,%d,%.2f,%.4f,%.4f",
                        s.timestamp, s.totalCpuEnergyKwh, s.totalGpuEnergyKwh,
                        s.totalEnergyKwh, s.tasksScheduled, s.slaViolations,
                        s.avgSlaSlack, s.clusterCpuUtil, s.clusterGpuUtil));
                w.newLine();
            }
        }

        System.out.printf("[MetricsExporter] CSV written: %s (%d rows)%n",
                filePath, snapshots.size());
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
                  "makespan_sec":         %.1f
                }
                """,
                summary.totalEnergyKwh, summary.totalCpuEnergyKwh,
                summary.totalGpuEnergyKwh, summary.totalTasks,
                summary.slaViolations, summary.slaViolationRate,
                summary.avgCpuUtil, summary.avgGpuUtil, summary.makespan);

        Files.writeString(path, json);
        System.out.printf("[MetricsExporter] JSON written: %s%n", filePath);
    }

    // ── Utility: build a summary from final state ─────────────────────────

    /**
     * Convenience method to build a {@link Summary} from a list of snapshots.
     * Takes the last snapshot as the final cumulative state.
     */
    public static Summary buildSummary(List<Snapshot> snapshots) {
        if (snapshots.isEmpty()) {
            return new Summary(0, 0, 0, 0, 0, 0, 0, 0, 0);
        }

        Snapshot last = snapshots.getLast();
        double avgCpuUtil = snapshots.stream()
                .mapToDouble(Snapshot::clusterCpuUtil).average().orElse(0);
        double avgGpuUtil = snapshots.stream()
                .mapToDouble(Snapshot::clusterGpuUtil).average().orElse(0);

        double violationRate = last.tasksScheduled == 0
                ? 0.0
                : (double) last.slaViolations / last.tasksScheduled;

        return new Summary(
                last.totalEnergyKwh, last.totalCpuEnergyKwh, last.totalGpuEnergyKwh,
                last.tasksScheduled, last.slaViolations, violationRate,
                avgCpuUtil, avgGpuUtil, last.timestamp);
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
