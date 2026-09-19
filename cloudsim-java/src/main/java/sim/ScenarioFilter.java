package sim;

import sim.AlibabaTraceReader.TaskRecord;

import java.util.*;
import java.util.stream.Collectors;

/**
 * T2.3 — Filters the Alibaba trace into three load scenarios for experiments.
 *
 * <ul>
 *   <li><b>LEGACY_LOW</b>   — first 25 % of tasks by arrival time</li>
 *   <li><b>LEGACY_HIGH</b>  — all schedulable tasks</li>
 *   <li><b>LEGACY_BURST</b> — tasks from time windows whose arrival rate exceeds
 *                             the 80th percentile</li>
 *   <li><b>NONE</b> — passthrough for a WM-1 trace, which already is one scenario</li>
 * </ul>
 *
 * <p>These three slices are kept, and named LEGACY_, because every Phase-1 and
 * §14/§15 result was produced with them: reproducing those numbers, and reporting
 * the before/after comparison, both require the old behaviour to stay available and
 * unambiguous. They are <i>not</i> the WM-1 scenarios of the same name — see
 * {@link Scenario} and PLAN-Workload-Model.md §1.
 *
 * In every scenario, tasks with {@code podPhase = Pending} are excluded because
 * they were never scheduled and have no meaningful execution duration.
 */
public final class ScenarioFilter {

    /**
     * {@code LEGACY_*} are the Phase-1 slices of a single trace file. {@code NONE}
     * (W2.2) is the passthrough used when the trace was selected by
     * {@code TRACE_PATTERN}: a WM-1 file already *is* one scenario, so slicing it again
     * would silently produce a different workload than the one whose offered load and
     * burstiness were calibrated.
     *
     * <p><b>Why the LEGACY_ prefix (W2.3).</b> WM-1 also has scenarios called LOW, HIGH
     * and BURST, and they are <i>not</i> these — the Phase-1 LOW is "the first 25 % of
     * rows", the WM-1 LOW is "offered load 0.30". Two different things wearing the same
     * name in the same codebase is exactly how a reader, or a future maintainer, ends up
     * comparing numbers that are not comparable. The prefix makes every log line and
     * every call site say which one it means.
     *
     * <p>The wire protocol is unchanged: {@link #fromLabel} still accepts "LOW", so
     * every existing script, saved result and Python caller keeps working.
     *
     * <p>{@code NONE} still drops {@code Pending} pods, so the "what the simulator
     * schedules" rule is the same on both paths.
     */
    public enum Scenario {
        LEGACY_LOW, LEGACY_HIGH, LEGACY_BURST, NONE;

        /**
         * Parse a scenario label, accepting both the historical spelling ("LOW") and
         * the explicit one ("LEGACY_LOW"), case-insensitively.
         *
         * @throws IllegalArgumentException with the accepted spellings listed
         */
        public static Scenario fromLabel(String label) {
            if (label == null) {
                throw new IllegalArgumentException("scenario label must not be null");
            }
            String v = label.trim().toUpperCase(Locale.ROOT);
            return switch (v) {
                case "LOW",   "LEGACY_LOW"   -> LEGACY_LOW;
                case "HIGH",  "LEGACY_HIGH"  -> LEGACY_HIGH;
                case "BURST", "LEGACY_BURST" -> LEGACY_BURST;
                case "NONE"                  -> NONE;
                default -> throw new IllegalArgumentException(
                        "unknown scenario '" + label + "'; the legacy filter accepts "
                      + "LOW | HIGH | BURST (or the LEGACY_ spelling) and NONE. "
                      + "Scenarios such as OVERLOAD or REPLAY exist only as generated "
                      + "WM-1 traces — set TRACE_PATTERN to select them.");
            };
        }
    }

    // ── Time-window width for burst detection (seconds) ────────────────────
    private static final double BURST_WINDOW_SEC = 3600.0;   // 1 hour

    // ── Percentile threshold for burst windows ─────────────────────────────
    private static final double BURST_PERCENTILE = 0.80;

    // ── Low-load fraction ──────────────────────────────────────────────────
    private static final double LOW_FRACTION = 0.25;

    // ── Public API ─────────────────────────────────────────────────────────

    /**
     * Filter tasks according to the chosen scenario.
     * Input list is NOT modified; a new sorted list is returned.
     *
     * @param allTasks full parsed trace (sorted by creationTime)
     * @param scenario desired load profile
     * @param seed     random seed (used for reproducibility in sampling)
     * @return filtered, sorted, unmodifiable list
     */
    public static List<TaskRecord> filter(List<TaskRecord> allTasks,
                                          Scenario scenario,
                                          long seed) {

        // Step 1 — exclude non-schedulable tasks
        List<TaskRecord> schedulable = allTasks.stream()
                .filter(t -> !"Pending".equals(t.podPhase()))
                .toList();

        List<TaskRecord> result = switch (scenario) {
            case LEGACY_LOW   -> filterLow(schedulable, seed);
            case LEGACY_HIGH  -> filterHigh(schedulable);
            case LEGACY_BURST -> filterBurst(schedulable);
            case NONE         -> filterHigh(schedulable);  // W2.2 — already a scenario
        };

        System.out.printf("[ScenarioFilter] %s: %d → %d tasks%n",
                scenario, allTasks.size(), result.size());

        return Collections.unmodifiableList(result);
    }

    // ── LOW: earliest 25 % of tasks by arrival time ────────────────────────

    private static List<TaskRecord> filterLow(List<TaskRecord> tasks, long seed) {
        if (tasks.isEmpty()) return List.of();

        int count = Math.max(1, (int) (tasks.size() * LOW_FRACTION));

        // Tasks are already sorted by creationTime — take the first chunk.
        // This naturally produces a low-utilisation period.
        return new ArrayList<>(tasks.subList(0, count));
    }

    // ── HIGH: all schedulable tasks ────────────────────────────────────────

    private static List<TaskRecord> filterHigh(List<TaskRecord> tasks) {
        return new ArrayList<>(tasks);
    }

    // ── BURST: time windows with arrival rate ≥ 80th percentile ────────────

    private static List<TaskRecord> filterBurst(List<TaskRecord> tasks) {
        if (tasks.isEmpty()) return List.of();

        // 1. Bucket tasks into fixed-width windows
        double minTime = tasks.getFirst().creationTime();
        double maxTime = tasks.getLast().creationTime();

        Map<Long, List<TaskRecord>> windows = new LinkedHashMap<>();
        for (TaskRecord t : tasks) {
            long bucket = (long) ((t.creationTime() - minTime) / BURST_WINDOW_SEC);
            windows.computeIfAbsent(bucket, k -> new ArrayList<>()).add(t);
        }

        // 2. Find the 80th-percentile threshold on window sizes
        int[] sizes = windows.values().stream()
                .mapToInt(List::size)
                .sorted()
                .toArray();

        int thresholdIdx = (int) (sizes.length * BURST_PERCENTILE);
        int threshold    = sizes[Math.min(thresholdIdx, sizes.length - 1)];

        // 3. Collect tasks from windows that meet or exceed the threshold
        List<TaskRecord> result = new ArrayList<>();
        for (List<TaskRecord> window : windows.values()) {
            if (window.size() >= threshold) {
                result.addAll(window);
            }
        }

        // Keep sorted by creationTime
        result.sort(Comparator.comparingDouble(TaskRecord::creationTime));
        return result;
    }

    private ScenarioFilter() {} // utility class
}
