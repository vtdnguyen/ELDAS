package sim;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * T2.2 — Parses the Alibaba OpenB trace CSV into a list of {@link TaskRecord}s.
 *
 * CSV columns (from alibaba/clusterdata cluster-trace-gpu-v2023):
 *   name, cpu_milli, memory_mib, num_gpu, gpu_milli,
 *   gpu_spec, qos, pod_phase, creation_time, deletion_time, scheduled_time
 *
 * Each row becomes an immutable {@link TaskRecord}.  Records with invalid or
 * unparseable fields are skipped with a warning rather than aborting the whole
 * import — the trace is large enough that a few bad rows are tolerable.
 */
public final class AlibabaTraceReader {

    // ── Data record ────────────────────────────────────────────────────────

    public record TaskRecord(
        String name,
        int    cpuMilli,        // CPU request in milli-cores  (e.g. 12 000 = 12 cores)
        int    memoryMib,       // memory in MiB
        int    numGpu,          // GPU count (0-8)
        int    gpuMilli,        // GPU share in milli (1000 = 1 full card)
        String gpuSpec,         // GPU type constraint (often empty)
        String qos,             // BE | LS | Burstable | Guaranteed
        String podPhase,        // Running | Succeeded | Failed | Pending
        double creationTime,    // seconds from trace start
        double deletionTime,    // seconds from trace start
        double scheduledTime,   // seconds from trace start
        double deadline,        // derived: expected completion time
        double qosWeight        // derived: QoS → penalty multiplier κ (G1.0)
    ) {
        /** Estimated execution duration (seconds). */
        public double duration() {
            return Math.max(0, deletionTime - Math.max(creationTime, scheduledTime));
        }

        /** Number of CPU PEs (at least 1). */
        public int pesNeeded() {
            return Math.max(1, cpuMilli / 1000);
        }
    }

    // ── CSV column indices ─────────────────────────────────────────────────

    private static final int COL_NAME           = 0;
    private static final int COL_CPU_MILLI      = 1;
    private static final int COL_MEMORY_MIB     = 2;
    private static final int COL_NUM_GPU        = 3;
    private static final int COL_GPU_MILLI      = 4;
    private static final int COL_GPU_SPEC       = 5;
    private static final int COL_QOS            = 6;
    private static final int COL_POD_PHASE      = 7;
    private static final int COL_CREATION_TIME  = 8;
    private static final int COL_DELETION_TIME  = 9;
    private static final int COL_SCHEDULED_TIME = 10;
    private static final int MIN_COLUMNS        = 11;

    // ── Public API ─────────────────────────────────────────────────────────

    /**
     * Read and parse the CSV file into an unmodifiable list of {@link TaskRecord}s,
     * sorted by {@code creationTime} ascending.
     *
     * @param csvPath path to the CSV (e.g. {@code /data/trace/openb_pod_list_default.csv})
     * @return sorted, immutable list of task records
     * @throws IOException if the file cannot be read
     */
    public static List<TaskRecord> read(String csvPath) throws IOException {
        Path path = Path.of(csvPath);
        List<TaskRecord> records = new ArrayList<>();
        int skipped = 0;

        try (BufferedReader br = Files.newBufferedReader(path)) {
            String header = br.readLine(); // skip header
            if (header == null) {
                throw new IOException("Trace file is empty: " + csvPath);
            }

            String line;
            int lineNum = 1;
            while ((line = br.readLine()) != null) {
                lineNum++;
                TaskRecord rec = parseLine(line, lineNum);
                if (rec != null) {
                    records.add(rec);
                } else {
                    skipped++;
                }
            }
        }

        records.sort(Comparator.comparingDouble(TaskRecord::creationTime));

        System.out.printf("[AlibabaTraceReader] Loaded %d tasks from %s (skipped %d)%n",
                records.size(), csvPath, skipped);

        return Collections.unmodifiableList(records);
    }

    // ── Parser ─────────────────────────────────────────────────────────────

    private static TaskRecord parseLine(String line, int lineNum) {
        String[] cols = line.split(",", -1);   // -1 keeps trailing empties
        if (cols.length < MIN_COLUMNS) {
            System.err.printf("[AlibabaTraceReader] Line %d: expected %d columns, got %d — skipped%n",
                    lineNum, MIN_COLUMNS, cols.length);
            return null;
        }

        try {
            String name      = cols[COL_NAME].trim();
            int    cpuMilli   = parseIntSafe(cols[COL_CPU_MILLI],  0);
            int    memoryMib  = parseIntSafe(cols[COL_MEMORY_MIB], 0);
            int    numGpu     = parseIntSafe(cols[COL_NUM_GPU],    0);
            int    gpuMilli   = parseIntSafe(cols[COL_GPU_MILLI],  0);
            String gpuSpec    = cols[COL_GPU_SPEC].trim();
            String qos        = cols[COL_QOS].trim();
            String podPhase   = cols[COL_POD_PHASE].trim();
            double creation   = parseDoubleSafe(cols[COL_CREATION_TIME],  0);
            double deletion   = parseDoubleSafe(cols[COL_DELETION_TIME],  0);
            double scheduled  = parseDoubleSafe(cols[COL_SCHEDULED_TIME], 0);

            // Derive deadline as a QoS-dependent slack budget over the
            // task's nominal duration. Using `deletion` directly is wrong:
            // it equals the natural completion time, so any scheduler
            // achieves slack = 0 by definition (see java-validation-report
            // §B4). Tying the deadline to slackFactor(qos) lets stricter
            // classes (LS) act as the contention signal for the RL agent.
            double rawDuration = Math.max(0, deletion - Math.max(creation, scheduled));
            double deadline    = creation + rawDuration * SimulationConfig.qosToSlackFactor(qos);

            double qosWeight = SimulationConfig.qosToWeight(qos);

            return new TaskRecord(name, cpuMilli, memoryMib, numGpu, gpuMilli,
                    gpuSpec, qos, podPhase, creation, deletion, scheduled,
                    deadline, qosWeight);

        } catch (Exception e) {
            System.err.printf("[AlibabaTraceReader] Line %d: parse error (%s) — skipped%n",
                    lineNum, e.getMessage());
            return null;
        }
    }

    private static int parseIntSafe(String s, int fallback) {
        String trimmed = s.trim();
        if (trimmed.isEmpty() || "nan".equalsIgnoreCase(trimmed)) return fallback;
        return Integer.parseInt(trimmed);
    }

    private static double parseDoubleSafe(String s, double fallback) {
        String trimmed = s.trim();
        if (trimmed.isEmpty() || "nan".equalsIgnoreCase(trimmed)) return fallback;
        return Double.parseDouble(trimmed);
    }

    private AlibabaTraceReader() {} // utility class
}
