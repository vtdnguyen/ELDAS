package sim;

import sim.AlibabaTraceReader.TaskRecord;

import java.util.*;
import java.util.stream.Collectors;

/**
 * T2.3 — Filters the Alibaba trace into three load scenarios for experiments.
 *
 * <ul>
 *   <li><b>LOW</b>  — light load: first 25 % of tasks by arrival time</li>
 *   <li><b>HIGH</b> — heavy load: all schedulable tasks</li>
 *   <li><b>BURST</b>— bursty load: tasks from time windows whose arrival rate
 *                     exceeds the 80th percentile</li>
 * </ul>
 *
 * In every scenario, tasks with {@code podPhase = Pending} are excluded because
 * they were never scheduled and have no meaningful execution duration.
 */
public final class ScenarioFilter {

    public enum Scenario { LOW, HIGH, BURST }

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
            case LOW   -> filterLow(schedulable, seed);
            case HIGH  -> filterHigh(schedulable);
            case BURST -> filterBurst(schedulable);
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
