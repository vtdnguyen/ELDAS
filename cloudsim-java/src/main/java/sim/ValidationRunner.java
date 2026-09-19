package sim;

import org.cloudsimplus.hosts.Host;

import sim.AlibabaTraceReader.TaskRecord;
import sim.DatacenterFactory.GpuState;
import sim.ScenarioFilter.Scenario;
import sim.SimulationManager.StepResult;

import java.lang.reflect.Field;
import java.util.*;

/**
 * Standalone validation runner — empirically demonstrates the bugs catalogued
 * in {@code docs/archive/phase1-dacn/docs/java-validation-report.md}.
 *
 * <p>Each {@code testBxx_…} method asserts something about the production code
 * and prints {@code [PASS] Bxx — description} (bug confirmed) or
 * {@code [FAIL] Bxx — …} (claim contradicted — investigate).
 *
 * <p>Run inside the cloudsim-java image (rebuild it first so this class
 * is included in {@code simulation.jar}):
 * <pre>
 *   docker compose build cloudsim-java
 *   docker compose run --rm --no-deps --entrypoint java cloudsim-java \
 *       -cp simulation.jar sim.ValidationRunner
 * </pre>
 *
 * <p>Exits with code 0 if every assertion passes, 1 otherwise.
 */
public final class ValidationRunner {

    private static int passes = 0;
    private static int fails  = 0;
    private static int skips  = 0;
    private static final List<String> failNotes = new ArrayList<>();
    private static final List<String> skipNotes = new ArrayList<>();

    public static void main(String[] args) {
        banner("ELDAS Java-side validation");

        testB7_HighFilterIsNoOp();
        testB9_DefaultConstructorSeedBug();
        testB4_SlaMathCannotTrigger();
        testB1_K8sAlwaysPicksLowestIndex();
        testB2_ResourcesNeverReleased();
        testB3_EnergyIntegratesLogicalTime();
        testB5_AvgCpuUtilFormula();
        testB6_ClampInconsistency();
        testB8_CustomPolicyNotUsedByDatacenter();
        testB10_ActionMaskWhenDone();
        testB11_IdleThresholdSuspendsUntouchedHosts();
        testB12_EnvOverrideFromSystemProps();
        testB13_StepAdvancesByExactlyOneTaskPerCall();
        testB14_ActionMaskReflectsCurrentTaskFeasibility();
        testB15_SuspendedHostIsStillSchedulable();
        testB16_SlaCostIsNonNegativeAndMatchesReward();
        testB17_HeterogeneousTopologyAndAffinity();
        testB18_StepCodecPacksLosslessly();
        testB19_ResetDoesNotLeakSteppingThreads();
        testB20_DeadlineHasAnAbsoluteFloor();
        testB21_TraceResolutionIsSafe();
        testB22_GeneratedTraceMatchesItsManifest();
        testB23_DroppedTasksAreCharged();

        banner("Summary");
        String topo = System.getenv("TOPOLOGY_CONFIG");
        System.out.printf("topology = %s%n",
                (topo != null && !topo.isBlank()) ? "HETEROGENEOUS (" + topo + ")"
                                                  : "homogeneous (default)");
        System.out.printf("PASS = %d (bug confirmed)%n", passes);
        System.out.printf("FAIL = %d (claim contradicted)%n", fails);
        System.out.printf("SKIP = %d (precondition N/A in this config)%n", skips);
        if (skips > 0) {
            System.out.println();
            for (String n : skipNotes) System.out.println("  - " + n);
        }
        if (fails > 0) {
            System.out.println();
            for (String n : failNotes) System.out.println("  - " + n);
        }
        System.exit(fails == 0 ? 0 : 1);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B7 — HIGH filter is no-op
    // ──────────────────────────────────────────────────────────────────────
    private static void testB7_HighFilterIsNoOp() {
        try {
            List<TaskRecord> all = AlibabaTraceReader.read(SimulationConfig.TRACE_FILE);
            List<TaskRecord> high = ScenarioFilter.filter(all, Scenario.LEGACY_HIGH, 42);
            long nonPending = all.stream().filter(t -> !"Pending".equals(t.podPhase())).count();
            assertEq("B7", "HIGH size == all non-Pending size", nonPending, (long) high.size());
        } catch (Exception e) {
            fail("B7", "cannot read trace: " + e.getMessage());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B9 (FIXED) — Default constructor now reads RANDOM_SEED env (default 42)
    //
    //  Pre-fix: default ctor passed DEFAULT_HOST.pesCount() (64) as the seed,
    //  silently making "no-seed" runs use a topology constant.
    //  Post-fix (T8.1): the default ctor reads RANDOM_SEED env, falling back
    //  to 42. This test confirms the fix.
    // ──────────────────────────────────────────────────────────────────────
    private static void testB9_DefaultConstructorSeedBug() {
        try {
            SimulationManager m = new SimulationManager();
            Field f = SimulationManager.class.getDeclaredField("seed");
            f.setAccessible(true);
            long seed = (long) f.get(m);
            long expected = Long.parseLong(
                    System.getenv().getOrDefault("RANDOM_SEED", "42"));
            assertEq("B9", "default ctor seed == RANDOM_SEED env (or 42)",
                    expected, seed);
        } catch (Exception e) {
            fail("B9", e.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B4 (FIXED) — SLA can trigger AND the formula responds to host load
    //
    //  Pre-fix: deadline == deletionTime, estimated == creation + duration
    //   ⇒ slack ≤ 0 for every task.
    //  Post-fix: deadline   = creation + duration × slackFactor(qos);
    //            estimated  = creation + duration × (1 + bgUtil_host).
    //
    //  Two assertions:
    //    B4a — running a real episode produces ≥ 1 SLA violation (proves
    //          the trigger condition can fire, contradicting the pre-fix
    //          algebraic proof that it cannot).
    //    B4b — mechanism check: for the SAME LS task, slack is strictly
    //          positive when the chosen host is heavily loaded, and
    //          non-positive when it is idle. This validates the formula
    //          directly, without depending on emergent behaviour of the
    //          full episode — which is workload-dependent and dominated
    //          by the trace's natural sparsity in LOW scenario.
    // ──────────────────────────────────────────────────────────────────────
    private static void testB4_SlaMathCannotTrigger() {
        try {
            int realRunViolations = runRealEpisodeRoundRobin();
            System.out.printf("       round-robin LOW violations = %d%n", realRunViolations);
            assertTrue("B4a", "SLA violation triggers fire in a real episode (≥ 1)",
                    realRunViolations >= 1);

            // B4b — direct formula check (independent of trace).
            // For an LS task of nominal duration 100s starting at t=0:
            //   slackFactor(LS) = 1.1  ⇒ deadline = 0 + 100 × 1.1 = 110
            //   on 50%-loaded host: congestionFactor = 1.5
            //     estimated = 100 × 1.5 = 150  ⇒ slack = 40 > 0  (violation)
            //   on idle host:        congestionFactor = 1.0
            //     estimated = 100 × 1.0 = 100  ⇒ slack = -10 ≤ 0 (no violation)
            double duration = 100;
            double deadline = duration * SimulationConfig.qosToSlackFactor("LS");
            double slackLoaded = duration * (1.0 + 0.5) - deadline;
            double slackEmpty  = duration * (1.0 + 0.0) - deadline;
            System.out.printf("       formula check: LS slack on 50%%-loaded = %.1f, on idle = %.1f%n",
                    slackLoaded, slackEmpty);
            assertTrue("B4b-loaded", "LS slack > 0 on 50 %-loaded host",  slackLoaded > 0);
            assertTrue("B4b-empty",  "LS slack ≤ 0 on idle host",         slackEmpty <= 0);
        } catch (Exception e) {
            fail("B4", e.toString());
        }
    }

    /** Drive a real LOW episode using a round-robin policy, returning the
     *  final violation count. Round-robin is a reasonable "typical" baseline. */
    private static int runRealEpisodeRoundRobin() throws Exception {
        SimulationManager m = new SimulationManager(
                SimulationConfig.DEFAULT_DC,
                SimulationConfig.TRACE_FILE,
                Scenario.LEGACY_LOW,
                42L);
        m.resetSimulation();
        int hostCount = m.getHostCount();
        int steps = Math.min(400, m.getTasks().size() - 1);
        for (int i = 0; i < steps; i++) {
            StepResult r = m.step(i % hostCount);
            if (r.done()) break;
        }
        Field f = SimulationManager.class.getDeclaredField("slaViolationCount");
        f.setAccessible(true);
        return f.getInt(m);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B1 (FIXED) — K8s reads live state, no longer always picks index 0
    //
    //  After fix, K8s sees the actual free PEs on each host (from
    //  SimulationManager.freePes). With host 0 partially loaded, its
    //  leastRequestedScore drops below idle hosts, and a subsequent
    //  selection picks a different host.
    // ──────────────────────────────────────────────────────────────────────
    private static void testB1_K8sAlwaysPicksLowestIndex() {
        try {
            SimulationManager m = new SimulationManager(
                    SimulationConfig.DEFAULT_DC,
                    SimulationConfig.TRACE_FILE,
                    Scenario.LEGACY_LOW,
                    42L);
            m.resetSimulation();

            VmAllocationPolicyK8sDefault k8s = new VmAllocationPolicyK8sDefault(m);
            List<Host> hosts = m.getHosts();

            // Step 1: empty cluster → K8s picks index 0 (deterministic tie-break).
            TaskRecord cpuTask = new TaskRecord(
                    "t1", 4000, 1024, 0, 0, "", "BE", "Running",
                    0, 100, 0, 100, 0.5);
            int idxEmpty = k8s.selectHostForTask(hosts, cpuTask);
            assertEq("B1a", "on empty cluster K8s picks index 0 (deterministic tie)", 0, idxEmpty);

            // Step 2: pump one task into host 0 to skew its utilisation,
            // then re-query K8s with the same synthetic task. Host 0 should
            // no longer be the best choice.
            m.step(0);
            int idxAfter = k8s.selectHostForTask(hosts, cpuTask);
            System.out.printf("       after pumping 1 task into host 0, K8s picks index %d%n", idxAfter);
            assertTrue("B1b", "K8s no longer picks host 0 once host 0 is loaded",
                    idxAfter != 0);
        } catch (Exception e) {
            fail("B1", e.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B2 (FIXED) — Resources DO release as tasks complete
    //
    //  Pre-fix: hostPeUsage grew monotonically, blowing past cluster capacity.
    //  Post-fix: completions queued in allocateTask are processed by
    //  advanceEnergy → releaseAllocation, so totals stay ≤ capacity AND
    //  individual hosts' usage decreases over time.
    // ──────────────────────────────────────────────────────────────────────
    private static void testB2_ResourcesNeverReleased() {
        try {
            SimulationManager m = new SimulationManager(
                    SimulationConfig.DEFAULT_DC,
                    SimulationConfig.TRACE_FILE,
                    Scenario.LEGACY_LOW,
                    42L);
            m.resetSimulation();

            int cluster = SimulationConfig.DEFAULT_DC.hostCount()
                    * SimulationConfig.DEFAULT_HOST.pesCount();
            int totalSteps = Math.min(500, m.getTasks().size() - 1);

            long maxObserved = 0;
            long peakStep    = -1;
            boolean sawDecrease = false;
            long prev = 0;

            for (int i = 0; i < totalSteps; i++) {
                StepResult r = m.step(0);
                long pe = sumPeUsage(m);
                if (pe > maxObserved) { maxObserved = pe; peakStep = i; }
                if (i > 0 && pe < prev) sawDecrease = true;
                prev = pe;
                if (r.done()) break;
            }

            System.out.printf(
                "       max PE-usage observed = %d at step %d (cluster capacity = %d)%n",
                maxObserved, peakStep, cluster);
            assertTrue("B2a", "total PE usage stays within cluster capacity",
                    maxObserved <= cluster);
            assertTrue("B2b", "at least one release was observed",
                    sawDecrease);
        } catch (Exception e) {
            fail("B2", e.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B3 (FIXED) — Energy is now bounded by physically meaningful envelopes
    //
    //  Energy is integrated piecewise across the simulated wall-clock
    //  (which equals the trace creationTime axis). For any completed
    //  episode E ∈ [E_idle, E_max] where the bounds use the cluster's
    //  total idle / peak power × elapsed wall-clock time.
    // ──────────────────────────────────────────────────────────────────────
    private static void testB3_EnergyIntegratesLogicalTime() {
        try {
            SimulationManager m = new SimulationManager(
                    SimulationConfig.DEFAULT_DC,
                    SimulationConfig.TRACE_FILE,
                    Scenario.LEGACY_LOW,
                    42L);
            m.resetSimulation();

            // Step through the full episode
            while (!m.isDone()) {
                StepResult r = m.step(0);
                if (r.done()) break;
            }

            int h = SimulationConfig.DEFAULT_DC.hostCount();
            int gpus = SimulationConfig.DEFAULT_HOST.gpuCount();
            var ps = SimulationConfig.DEFAULT_POWER;
            // T8.1 — Lower bound is now "all-suspended", not "all-idle":
            // a host that stays empty past IDLE_THRESHOLD draws only P_sus.
            double minW = h * ps.suspendedPowerWatt();
            double maxW = h * (ps.cpuMaxPowerWatt() + gpus * ps.gpuMaxPowerWatt());

            // End-of-sim simulated wall-clock = last snapshot timestamp
            double T = m.getSnapshots().getLast().timestamp();
            double kwh = m.getTotalEnergyKwh();
            double minLowerKwh = minW * T / 3_600_000.0;
            double maxUpperKwh = maxW * T / 3_600_000.0;

            System.out.printf(
                "       T_end=%.0f s, energy=%.2f kWh, suspended-lower=%.2f, max-upper=%.2f, wakeups=%d%n",
                T, kwh, minLowerKwh, maxUpperKwh, m.getTotalWakeups());
            assertTrue("B3a", "energy ≥ all-suspended baseline",
                    kwh + 1e-6 >= minLowerKwh);
            assertTrue("B3b", "energy ≤ cluster-peak upper bound",
                    kwh <= maxUpperKwh + 1e-6);
        } catch (Exception e) {
            fail("B3", e.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B5 (partially FIXED) — clusterCpuUtil is now bounded to [0, 1]
    //
    //  Pre-fix: snapshot averaged unclamped cumulative PE counts ÷ pesCount,
    //  so values exceeded 1.0. Post-fix the snapshot clamps per-host before
    //  averaging, matching the energy model's clamp.
    //  NOT YET FIXED: averaging snapshots by count instead of by time —
    //  see docs/archive/phase1-dacn/docs/java-validation-report.md mục 2.5.
    // ──────────────────────────────────────────────────────────────────────
    private static void testB5_AvgCpuUtilFormula() {
        try {
            SimulationManager m = new SimulationManager(
                    SimulationConfig.DEFAULT_DC,
                    SimulationConfig.TRACE_FILE,
                    Scenario.LEGACY_LOW,
                    42L);
            m.resetSimulation();

            double maxSeen = 0;
            int steps = Math.min(500, m.getTasks().size() - 1);
            for (int i = 0; i < steps; i++) {
                StepResult r = m.step(0);
                if (!m.getSnapshots().isEmpty()) {
                    double v = m.getSnapshots().getLast().clusterCpuUtil();
                    if (v > maxSeen) maxSeen = v;
                }
                if (r.done()) break;
            }
            System.out.printf("       max clusterCpuUtil observed = %.4f%n", maxSeen);
            assertTrue("B5", "clusterCpuUtil stays within [0, 1]",
                    maxSeen <= 1.0 + 1e-9);
        } catch (Exception e) {
            fail("B5", e.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B6 (FIXED) — Clamp consistency between energy and snapshot paths
    //
    //  Energy uses Math.min(1.0, usedPes/pesCount); the snapshot formula
    //  now does the same per-host before averaging. Both reach 1.0 only
    //  when the host is genuinely saturated.
    // ──────────────────────────────────────────────────────────────────────
    private static void testB6_ClampInconsistency() {
        try {
            SimulationManager m = new SimulationManager(
                    SimulationConfig.DEFAULT_DC,
                    SimulationConfig.TRACE_FILE,
                    Scenario.LEGACY_LOW,
                    42L);
            m.resetSimulation();

            for (int i = 0; i < 50; i++) {
                StepResult r = m.step(0);
                if (r.done()) break;
            }
            java.lang.reflect.Method bo =
                    SimulationManager.class.getDeclaredMethod("buildObservation");
            bo.setAccessible(true);
            double[] obs = (double[]) bo.invoke(m);
            double host0CpuObs = obs[0];
            double snapUtil    = m.getSnapshots().getLast().clusterCpuUtil();

            System.out.printf("       host0 obs util = %.4f, snapshot clusterCpuUtil = %.4f%n",
                    host0CpuObs, snapUtil);
            assertTrue("B6a", "observation util ∈ [0, 1]",
                    host0CpuObs <= 1.0 + 1e-9);
            assertTrue("B6b", "snapshot util ∈ [0, 1]",
                    snapUtil <= 1.0 + 1e-9);
        } catch (Exception e) {
            fail("B6", e.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B8 — DatacenterFactory default factory uses VmAllocationPolicySimple
    // ──────────────────────────────────────────────────────────────────────
    private static void testB8_CustomPolicyNotUsedByDatacenter() {
        try {
            org.cloudsimplus.core.CloudSimPlus sim = new org.cloudsimplus.core.CloudSimPlus();
            Map<Host, GpuState> reg = new IdentityHashMap<>();
            var dc = DatacenterFactory.create(sim, SimulationConfig.DEFAULT_DC, reg);
            var policy = dc.getVmAllocationPolicy();
            String cn = policy.getClass().getSimpleName();
            assertEq("B8", "default DC policy is VmAllocationPolicySimple, not K8s/Random",
                    "VmAllocationPolicySimple", cn);
        } catch (Exception e) {
            fail("B8", e.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B10 — getActionMask when episodeDone returns all-false
    // ──────────────────────────────────────────────────────────────────────
    private static void testB10_ActionMaskWhenDone() {
        try {
            // Force-set episodeDone to true via reflection
            SimulationManager m = new SimulationManager(
                    SimulationConfig.DEFAULT_DC,
                    SimulationConfig.TRACE_FILE,
                    Scenario.LEGACY_LOW,
                    42L);
            m.resetSimulation();
            Field f = SimulationManager.class.getDeclaredField("episodeDone");
            f.setAccessible(true);
            f.setBoolean(m, true);

            boolean[] mask = m.getActionMask();
            int trues = countTrue(mask);
            assertEq("B10", "mask is all-false when episodeDone", 0, trues);
        } catch (Exception e) {
            fail("B10", e.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B11 (T8.1) — Idle threshold suspends hosts that never get a task
    //
    //  Drive one task into host 0 (with FirstFit-style ordering via step(0)).
    //  After the episode flushes, hosts 1..N-1 — which received no tasks —
    //  should be SUSPENDED in the final snapshot. The total energy delta
    //  from the "all-idle" model should therefore be a substantial saving.
    // ──────────────────────────────────────────────────────────────────────
    private static void testB11_IdleThresholdSuspendsUntouchedHosts() {
        try {
            SimulationManager m = new SimulationManager(
                    SimulationConfig.DEFAULT_DC,
                    SimulationConfig.TRACE_FILE,
                    Scenario.LEGACY_LOW,
                    42L);
            m.resetSimulation();

            // Run the full episode, always packing onto host 0. This is a
            // synthetic BestFit: under the new state machine, hosts 1..9
            // get nothing → idle 30 s → suspend for the remaining sim time.
            while (!m.isDone()) {
                StepResult r = m.step(0);
                if (r.done()) break;
            }

            var last = m.getSnapshots().getLast();
            System.out.printf(
                "       end-of-sim: active=%d, idle=%d, suspended=%d, wakeups=%d, energy=%.2f kWh%n",
                last.activeHosts(), last.idleHosts(), last.suspendedHosts(),
                m.getTotalWakeups(), m.getTotalEnergyKwh());

            // The state machine is observable through (a) snapshots showing
            // hosts in SUSPENDED at some point and (b) wake-up events when
            // step(0) routes a task to host 0 after it idled out. Both
            // signals must be non-zero. We can't pin the final-snapshot
            // SUSPENDED count to >= 8 because flushTillEnd lands at the
            // last completion time — hosts that just released are still IDLE
            // (within the 30-s threshold) at that instant.
            int peakSuspended = m.getSnapshots().stream()
                    .mapToInt(MetricsExporter.Snapshot::suspendedHosts)
                    .max().orElse(0);
            assertTrue("B11a", "≥ 1 host SUSPENDED at some snapshot",
                    peakSuspended >= 1);
            assertTrue("B11b", "≥ 1 wake-up observed during episode",
                    m.getTotalWakeups() >= 1);
            // Sanity: state codes sum to host count.
            assertEq("B11c", "active+idle+suspended == host count",
                    SimulationConfig.DEFAULT_DC.hostCount(),
                    last.activeHosts() + last.idleHosts() + last.suspendedHosts());
        } catch (Exception e) {
            fail("B11", e.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B12 (T7.1) — fromEnv() honours JVM system properties (env fallback)
    //
    //  We don't mutate the JVM's env-var table from inside Java, so this
    //  test uses System.setProperty (the second-priority source in
    //  SimulationConfig.fromEnv) to verify the resolution chain works.
    // ──────────────────────────────────────────────────────────────────────
    private static void testB12_EnvOverrideFromSystemProps() {
        try {
            // G2.1 — This test's precondition is a HOMOGENEOUS, env/prop-driven
            // topology. When TOPOLOGY_CONFIG is set, the topology JSON is the
            // more specific source and deliberately overrides the scalar knobs
            // (NUM_HOSTS / VCPU_PER_HOST / GPU_PER_HOST), so asserting the
            // sys-props still win would flag intended behaviour as a bug.
            // Skip rather than emit a misleading FAIL.
            String topo = System.getenv("TOPOLOGY_CONFIG");
            if (topo != null && !topo.isBlank()) {
                skip("B12", "TOPOLOGY_CONFIG='" + topo + "' is set — the topology JSON "
                        + "intentionally overrides the scalar topology knobs, so the "
                        + "sys-prop precedence check does not apply");
                return;
            }

            // Restrict to props we'll mutate so we can roll back deterministically.
            String[] keys = {"eldas.num_hosts", "eldas.vcpu_per_host", "eldas.gpu_per_host"};
            String[] before = new String[keys.length];
            for (int i = 0; i < keys.length; i++) before[i] = System.getProperty(keys[i]);

            try {
                System.setProperty("eldas.num_hosts",     "7");
                System.setProperty("eldas.vcpu_per_host", "16");
                System.setProperty("eldas.gpu_per_host",  "2");

                var spec = SimulationConfig.fromEnv();
                assertEq("B12a", "num_hosts overridden by sys-prop",     7,  spec.hostCount());
                assertEq("B12b", "vcpu_per_host overridden by sys-prop", 16, spec.hostSpec().pesCount());
                assertEq("B12c", "gpu_per_host overridden by sys-prop",   2, spec.hostSpec().gpuCount());
            } finally {
                // Restore so subsequent tests in the same JVM run see clean state.
                for (int i = 0; i < keys.length; i++) {
                    if (before[i] == null) System.clearProperty(keys[i]);
                    else                   System.setProperty(keys[i], before[i]);
                }
            }
        } catch (Exception e) {
            fail("B12", e.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B13 (T8.8) — step() advances currentTaskIdx by exactly 1 per call
    //
    //  Confirms the RL stepping contract: each Py4J step() consumes exactly
    //  one pending task — no buffering, no batching, no skipped tasks.
    // ──────────────────────────────────────────────────────────────────────
    private static void testB13_StepAdvancesByExactlyOneTaskPerCall() {
        try {
            SimulationManager m = new SimulationManager(
                    SimulationConfig.DEFAULT_DC,
                    SimulationConfig.TRACE_FILE,
                    Scenario.LEGACY_LOW,
                    42L);
            m.resetSimulation();

            int before = m.getCurrentTaskIndex();
            assertEq("B13a", "reset leaves currentTaskIdx at 0", 0, before);

            // 10 successive steps must advance by exactly 10.
            int n = 10;
            for (int i = 0; i < n; i++) {
                int idxBefore = m.getCurrentTaskIndex();
                m.step(0);
                int idxAfter = m.getCurrentTaskIndex();
                if (idxAfter != idxBefore + 1) {
                    fail("B13b", String.format(
                        "step %d advanced by %d (expected exactly 1)",
                        i, idxAfter - idxBefore));
                    return;
                }
            }
            assertEq("B13b", "10 steps advance currentTaskIdx by 10",
                    n, m.getCurrentTaskIndex());
        } catch (Exception e) {
            fail("B13", e.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B14 (T8.8) — getActionMask() reflects current task's feasibility
    //
    //  Saturate host 0 (or any host) and confirm that mask[that host] becomes
    //  false ONLY when freePes/freeRam/freeGpus fall below the next task's
    //  requirements. This tests both: (a) mask uses live freePes/freeRam, and
    //  (b) the mask is computed against the CURRENT task, not a stale one.
    // ──────────────────────────────────────────────────────────────────────
    private static void testB14_ActionMaskReflectsCurrentTaskFeasibility() {
        try {
            SimulationManager m = new SimulationManager(
                    SimulationConfig.DEFAULT_DC,
                    SimulationConfig.TRACE_FILE,
                    Scenario.LEGACY_LOW,
                    42L);
            m.resetSimulation();

            // Initial mask before any allocation: at least one host must be
            // feasible for the first trace task, or the trace is broken.
            boolean[] m0 = m.getActionMask();
            int initial = countTrue(m0);
            System.out.printf("       initial mask trues = %d / %d%n",
                    initial, m0.length);
            assertTrue("B14a", "initial mask has ≥ 1 feasible host", initial >= 1);

            // Saturate host 0 by repeatedly placing tasks there. After enough
            // placements, freePes/freeRam/freeGpus on host 0 should hit zero,
            // and mask[0] for some future task should become false.
            //
            // We can't guarantee a specific task makes mask[0]=false (depends
            // on trace), but we CAN check: at SOME point during 200 successive
            // step(0) calls, mask[0] flips to false. If never, host 0 is
            // somehow infinitely-resourced — bug.
            boolean sawMaskFalse = false;
            for (int i = 0; i < 200 && !m.isDone(); i++) {
                m.step(0);
                boolean[] mk = m.getActionMask();
                if (mk.length > 0 && !mk[0]) { sawMaskFalse = true; break; }
            }
            assertTrue("B14b",
                    "mask[0] becomes false after enough placements on host 0",
                    sawMaskFalse);
        } catch (Exception e) {
            fail("B14", e.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B15 (T8.8) — A SUSPENDED host is still mask-feasible (can wake up)
    //
    //  Rationale: SUSPENDED means power-gated, not "rejecting work". When the
    //  scheduler picks a suspended host, the simulator pays wake-up cost and
    //  proceeds. So the mask, which is purely a resource feasibility check,
    //  must return true for suspended hosts that have free capacity.
    //
    //  Construction: do step(0) once → only host 0 has load; let energy
    //  accounting advance well past IDLE_THRESHOLD so hosts 1..9 transition
    //  to SUSPENDED. Then check that mask[1..9] for the NEXT task are still
    //  true (assuming the next task fits on an empty host).
    // ──────────────────────────────────────────────────────────────────────
    private static void testB15_SuspendedHostIsStillSchedulable() {
        try {
            SimulationManager m = new SimulationManager(
                    SimulationConfig.DEFAULT_DC,
                    SimulationConfig.TRACE_FILE,
                    Scenario.LEGACY_LOW,
                    42L);
            m.resetSimulation();

            // Skip ahead: step many tasks into host 0. The energy integration
            // crosses task arrival timestamps, which span seconds apart in
            // the trace — well past the 30-s idle threshold. So hosts 1..9
            // will have transitioned to SUSPENDED by the time we look.
            int suspendObserved = 0;
            for (int i = 0; i < 50 && !m.isDone(); i++) {
                m.step(0);
                if (!m.getSnapshots().isEmpty()) {
                    int s = m.getSnapshots().getLast().suspendedHosts();
                    if (s > suspendObserved) suspendObserved = s;
                }
            }
            assertTrue("B15a", "≥ 1 host enters SUSPENDED during the 50-step burst",
                    suspendObserved >= 1);

            // Read the post-burst snapshot's per-host state codes; find a
            // SUSPENDED host index and verify its mask bit is still set
            // when the next task is small enough to fit on an empty host.
            var last = m.getSnapshots().getLast();
            int suspendedIdx = -1;
            for (int i = 0; i < last.hostState().length; i++) {
                if (last.hostState()[i] == 0) { suspendedIdx = i; break; }
            }
            if (suspendedIdx < 0) {
                fail("B15b", "expected ≥ 1 SUSPENDED host but found none in last snapshot");
                return;
            }

            // The NEXT task's mask: SUSPENDED host should be true UNLESS the
            // task is too big for an empty host (then no host fits — different
            // failure). Probe by reading the mask and checking it's true.
            boolean[] mk = m.getActionMask();
            // Big tasks (e.g. cpu=88) can fail on EVERY host — skip ahead
            // to a task that fits at least somewhere. Up to 20 probes.
            int probes = 0;
            while (probes < 20 && !m.isDone() && countTrue(mk) == 0) {
                m.step(0); // burn through infeasible tasks
                mk = m.getActionMask();
                probes++;
                // Recompute suspendedIdx — the index we held might've woken.
                if (!m.getSnapshots().isEmpty()) {
                    int[] states = m.getSnapshots().getLast().hostState();
                    for (int i = 0; i < states.length; i++) {
                        if (states[i] == 0) { suspendedIdx = i; break; }
                    }
                }
            }

            System.out.printf(
                "       after burst: suspendedIdx=%d, mask[suspendedIdx]=%s, mask trues=%d%n",
                suspendedIdx, mk[suspendedIdx], countTrue(mk));
            assertTrue("B15b",
                    "mask[SUSPENDED host] is true (host can be woken)",
                    mk[suspendedIdx]);
        } catch (Exception e) {
            fail("B15", e.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B16 (G1.1) — CMDP constraint cost C_SLA is well-formed
    //
    //  Properties the PID-Lagrangian core depends on:
    //    (a) getSlaCost() ≥ 0 at all times (it is a one-sided violation cost);
    //    (b) it is monotonically non-decreasing over the episode (cumulative);
    //    (c) per-step StepResult.cost() = −min(0, R_sla) (non-negative SLA
    //        penalty magnitude), and Σ step costs == getSlaCost() (single
    //        definition reused everywhere — CLAUDE.md Lưu ý #5).
    // ──────────────────────────────────────────────────────────────────────
    private static void testB16_SlaCostIsNonNegativeAndMatchesReward() {
        try {
            SimulationManager m = new SimulationManager(
                    SimulationConfig.DEFAULT_DC,
                    SimulationConfig.TRACE_FILE,
                    Scenario.LEGACY_LOW,
                    42L);
            StepResult first = m.resetSimulation();

            assertTrue("B16a", "C_SLA starts at 0 after reset",
                    m.getSlaCost() == 0.0);
            assertTrue("B16a2", "initial reset reports zero step cost",
                    first.cost() == 0.0);

            double summedStepCost = 0.0;
            double prevCumulative = 0.0;
            boolean everNegative = false;
            boolean everDecreased = false;
            boolean mismatch = false;

            while (!m.isDone()) {
                StepResult r = m.step(0);  // pack everything onto host 0

                double stepCost = r.cost();
                double expectedStepCost = Math.max(0.0, -r.reward()[1]);
                if (stepCost < 0.0) everNegative = true;
                if (Math.abs(stepCost - expectedStepCost) > 1e-9) mismatch = true;
                summedStepCost += stepCost;

                double cumulative = m.getSlaCost();
                if (cumulative + 1e-9 < prevCumulative) everDecreased = true;
                prevCumulative = cumulative;

                if (r.done()) break;
            }

            double finalCost = m.getSlaCost();
            System.out.printf(
                "       C_SLA=%.4f, Σ step-cost=%.4f%n", finalCost, summedStepCost);

            assertTrue("B16b", "no negative per-step cost", !everNegative);
            assertTrue("B16c", "C_SLA never decreases (cumulative)", !everDecreased);
            assertTrue("B16d", "per-step cost == −min(0, R_sla)", !mismatch);
            assertTrue("B16e", "Σ step costs == getSlaCost()",
                    Math.abs(summedStepCost - finalCost) <= 1e-6);
            assertTrue("B16f", "final C_SLA ≥ 0", finalCost >= 0.0);
        } catch (Exception e) {
            fail("B16", e.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B19 (SYS) — reset() must not leak the previous episode's stepping thread.
    //
    //  GatewayEntryPoint.reset() builds a NEW SimulationManager each time, so
    //  the OLD manager's terminateExisting() never runs on its own behalf: its
    //  stepping thread stays blocked forever on an actionQueue nobody will write
    //  to, and — being a daemon — never blocks JVM exit, so nothing notices.
    //  Measured before the fix: +1 live "cloudsim-step" thread and ~1.4 MB RSS
    //  PER RESET (60 resets ⇒ 94→154 threads, 256→340 MB), because each stranded
    //  thread also pins its entire CloudSim graph against GC. Irrelevant for a
    //  smoke test; fatal for a budget sweep of hundreds of episodes × dozens of
    //  runs — exactly the long job this is all for.
    //
    //  Verifies that after N resets only ONE stepping thread is alive (the
    //  current episode's), not N. This is deterministic, not timing-dependent:
    //  terminateExisting() joins the old thread before reset returns.
    // ──────────────────────────────────────────────────────────────────────
    private static int countSteppingThreads() {
        int n = 0;
        for (Thread t : Thread.getAllStackTraces().keySet()) {
            if ("cloudsim-step".equals(t.getName()) && t.isAlive()) n++;
        }
        return n;
    }

    private static void testB19_ResetDoesNotLeakSteppingThreads() {
        GatewayEntryPoint ep = new GatewayEntryPoint();
        try {
            int before = countSteppingThreads();
            int resets = 6;
            for (int i = 0; i < resets; i++) {
                ep.reset("LOW", 42);
                ep.step(0);   // start the episode so the thread is genuinely busy
            }
            int after = countSteppingThreads();

            // Exactly one live stepping thread: the current episode's. Leaking
            // would give `before + resets`.
            assertEq("B19a", "only the live episode's stepping thread survives "
                            + resets + " resets (leak ⇒ " + (before + resets) + ")",
                    before + 1, after);

            ep.shutdown();
            // Give the interrupted thread a moment to unwind before counting.
            Thread.sleep(300);
            assertEq("B19b", "shutdown() reaps the last stepping thread too",
                    before, countSteppingThreads());
        } catch (Exception e) {
            fail("B19", e.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B18 (SYS.2) — StepCodec packs a step into ONE byte[] losslessly.
    //
    //  The codec exists because Py4J proxies double[]/boolean[] one element per
    //  round trip (~76 RPCs ≈ 21 ms per step at H=10, vs ~0.5 ms of actual
    //  simulation — measured by perf/profile_step.py). byte[] is the one array
    //  type Py4J passes by value, so the whole payload crosses in one call.
    //
    //  This is a CROSS-LANGUAGE contract: Python's state_builder.decode_packed
    //  reads these exact offsets. So the assertions below pin the *bytes* at
    //  explicit positions rather than round-tripping through a Java decoder —
    //  a Java-only round trip would agree with itself while drifting from the
    //  Python side. tests/test_packed_transport.py pins the mirror image, and
    //  perf/verify_packed_parity.py proves equality on the live stack.
    //
    //  Verifies: (a) version+done+taskIndex+H+obsLen header; (b) observation
    //  doubles are big-endian at offset 14; (c) reward/cost follow the
    //  observation; (d) the mask is one byte per host; (e) the UTF-8 task name
    //  is length-prefixed; (f) the blob is exactly the size the layout implies
    //  (no padding drift).
    // ──────────────────────────────────────────────────────────────────────
    private static void testB18_StepCodecPacksLosslessly() {
        try {
            int h = 3;
            int obsLen = 6 * h + 4;               // 22 — the 6H+4 contract (Lưu ý #14)
            double[] obs = new double[obsLen];
            for (int i = 0; i < obsLen; i++) obs[i] = i / 100.0;
            obs[0] = 1.0;                          // a value with a known bit pattern

            var result = new SimulationManager.StepResult(
                    obs, new double[]{-3.25, -7.5}, 7.5, true, 13, "pod-xyz", 4);
            boolean[] mask = {false, true, true};

            byte[] blob = StepCodec.encode(result, mask);
            var buf = java.nio.ByteBuffer.wrap(blob).order(java.nio.ByteOrder.BIG_ENDIAN);

            assertEq("B18a1", "header: wire version", (int) StepCodec.VERSION, (int) buf.get());
            assertEq("B18a2", "header: done flag", 1, (int) buf.get());
            assertEq("B18a3", "header: taskIndex", 13, buf.getInt());
            assertEq("B18a4", "header: numHosts", h, buf.getInt());
            assertEq("B18a5", "header: obsLen = 6H+4", obsLen, buf.getInt());
            // W3.1 — droppedTasks joined the header in wire v2, which is why the
            // observation moved from byte 14 to byte 18 and VERSION went to 2.
            assertEq("B18a6", "header: droppedTasks (W3.1)", 4, buf.getInt());

            // (b) Observation starts at byte 18 and is big-endian: 1.0 must be
            // 3F F0 00 00 00 00 00 00, which is what Python reads as '>f8'.
            assertEq("B18b1", "observation begins at offset 18", 18, buf.position());
            byte[] first = java.util.Arrays.copyOfRange(blob, 18, 26);
            assertTrue("B18b2", "observation[0]=1.0 encoded big-endian (3F F0 ...)",
                    java.util.Arrays.equals(first, new byte[]{
                            (byte) 0x3F, (byte) 0xF0, 0, 0, 0, 0, 0, 0}));

            boolean obsOk = true;
            for (int i = 0; i < obsLen; i++) {
                if (buf.getDouble() != obs[i]) { obsOk = false; break; }
            }
            assertTrue("B18b3", "all " + obsLen + " observation doubles survive exactly", obsOk);

            // (c) reward + cost follow immediately.
            assertEq("B18c1", "reward[0] = R_energy", -3.25, buf.getDouble());
            assertEq("B18c2", "reward[1] = R_sla",    -7.5,  buf.getDouble());
            assertEq("B18c3", "cost = C_SLA ≥ 0",      7.5,  buf.getDouble());

            // (d) one byte per host, in host order.
            boolean maskOk = true;
            for (boolean m : mask) {
                if (buf.get() != (byte) (m ? 1 : 0)) { maskOk = false; break; }
            }
            assertTrue("B18d", "action mask is one byte per host, in order", maskOk);

            // (e) length-prefixed UTF-8 name.
            int nameLen = buf.getInt();
            byte[] nameBytes = new byte[nameLen];
            buf.get(nameBytes);
            assertEq("B18e", "taskName round-trips as length-prefixed UTF-8",
                    "pod-xyz", new String(nameBytes, java.nio.charset.StandardCharsets.UTF_8));

            // (f) nothing left over: the blob is exactly the documented size.
            int expectedSize = 18 + 8 * obsLen + 16 + 8 + h + 4 + "pod-xyz".length();
            assertEq("B18f1", "blob size matches the layout exactly", expectedSize, blob.length);
            assertEq("B18f2", "decoder consumes the whole blob (no trailing bytes)",
                    0, buf.remaining());
        } catch (Exception e) {
            fail("B18", e.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B17 (G2.1 / G2.2) — Heterogeneous topology + GPU affinity masking
    //
    //  Builds a 3-SKU cluster (3× GPU-heavy 4-GPU, 3× balanced 2-GPU,
    //  4× CPU-only 0-GPU = 10 hosts) directly as a DatacenterSpec (no JSON /
    //  gson at test time — TopologyConfig is tested by parsing separately).
    //  Verifies:
    //    (a) total host count = 3+3+4 = 10;
    //    (b) per-host GPU inventory matches its SKU (0..2→4, 3..5→2, 6..9→0);
    //    (c) a GPU task (num_gpu>0) is feasible on GPU hosts, INFEASIBLE on
    //        every CPU-only host (affinity — Lưu ý #11);
    //    (d) a fractional-GPU task (num_gpu=0 but gpu_milli>0) is ALSO masked
    //        off CPU-only hosts (affinity keys on either GPU signal);
    //    (e) a pure-CPU task is feasible on ALL hosts (affinity never over-masks);
    //    (f) the idle→suspend energy gradient still fires under heterogeneity
    //        (Lưu ý #12 — hetero must NOT break the P1.8 state machine).
    // ──────────────────────────────────────────────────────────────────────
    private static void testB17_HeterogeneousTopologyAndAffinity() {
        try {
            var dp = SimulationConfig.DEFAULT_POWER;
            var dh = SimulationConfig.DEFAULT_HOST;

            // Helper specs: keep vcpu/ram uniform, vary GPU count + CPU power.
            SimulationConfig.HostSpec gpuHeavyHost = new SimulationConfig.HostSpec(
                    64, dh.mips(), 256L * 1024, dh.bwMbps(), dh.storageMb(), 4, dh.gpuMemoryMb());
            SimulationConfig.HostSpec balancedHost = new SimulationConfig.HostSpec(
                    64, dh.mips(), 256L * 1024, dh.bwMbps(), dh.storageMb(), 2, dh.gpuMemoryMb());
            SimulationConfig.HostSpec cpuOnlyHost = new SimulationConfig.HostSpec(
                    64, dh.mips(), 256L * 1024, dh.bwMbps(), dh.storageMb(), 0, dh.gpuMemoryMb());

            SimulationConfig.PowerSpec gpuHeavyPow = new SimulationConfig.PowerSpec(
                    500, 200, dp.gpuMaxPowerWatt(), dp.gpuIdlePowerWatt(),
                    dp.idleThresholdSec(), dp.suspendedPowerWatt(), dp.wakeEnergyKwh(), dp.wakeLatencySec());
            SimulationConfig.PowerSpec balancedPow = new SimulationConfig.PowerSpec(
                    350, 120, dp.gpuMaxPowerWatt(), dp.gpuIdlePowerWatt(),
                    dp.idleThresholdSec(), dp.suspendedPowerWatt(), dp.wakeEnergyKwh(), dp.wakeLatencySec());
            SimulationConfig.PowerSpec cpuOnlyPow = new SimulationConfig.PowerSpec(
                    200, 60, dp.gpuMaxPowerWatt(), dp.gpuIdlePowerWatt(),
                    dp.idleThresholdSec(), dp.suspendedPowerWatt(), dp.wakeEnergyKwh(), dp.wakeLatencySec());

            List<SimulationConfig.HostSku> skus = List.of(
                    new SimulationConfig.HostSku("gpu-heavy", 3, gpuHeavyHost, gpuHeavyPow),
                    new SimulationConfig.HostSku("balanced",  3, balancedHost, balancedPow),
                    new SimulationConfig.HostSku("cpu-only",  4, cpuOnlyHost,  cpuOnlyPow));

            var hetSpec = new SimulationConfig.DatacenterSpec(
                    10, gpuHeavyHost, gpuHeavyPow, 1.0, skus);

            SimulationManager m = new SimulationManager(
                    hetSpec, SimulationConfig.TRACE_FILE, Scenario.LEGACY_LOW, 42L);
            m.resetSimulation();
            List<Host> hosts = m.getHosts();

            // (a) total host count.
            assertEq("B17a", "hetero cluster has 3+3+4 = 10 hosts", 10, hosts.size());

            // (b) per-host GPU inventory.
            int[] expectedGpu = {4, 4, 4, 2, 2, 2, 0, 0, 0, 0};
            boolean gpuInvOk = true;
            for (int i = 0; i < hosts.size(); i++) {
                if (m.gpuTotal(hosts.get(i)) != expectedGpu[i]) gpuInvOk = false;
            }
            assertTrue("B17b", "per-host GPU inventory matches SKU (4,4,4,2,2,2,0,0,0,0)", gpuInvOk);

            // (c) GPU task: feasible on GPU hosts (0..5), infeasible on CPU-only (6..9).
            TaskRecord gpuTask = new TaskRecord(
                    "gpu-t", 4000, 1024, 2, 2000, "", "Guaranteed", "Running",
                    0, 100, 0, 100, 2.0);
            boolean gpuFeasibleOnGpuHosts = true, gpuMaskedOnCpuHosts = true;
            for (int i = 0; i < hosts.size(); i++) {
                boolean can = m.canHost(hosts.get(i), gpuTask);
                if (i <= 5 && !can) gpuFeasibleOnGpuHosts = false;
                if (i >= 6 &&  can) gpuMaskedOnCpuHosts   = false;
            }
            assertTrue("B17c1", "GPU task feasible on all GPU hosts (0..5)", gpuFeasibleOnGpuHosts);
            assertTrue("B17c2", "GPU task masked off every CPU-only host (6..9)", gpuMaskedOnCpuHosts);

            // (d) Fractional-GPU task (num_gpu=0, gpu_milli>0) — still GPU-only.
            TaskRecord fracTask = new TaskRecord(
                    "frac-t", 2000, 512, 0, 500, "", "Burstable", "Running",
                    0, 100, 0, 100, 1.0);
            boolean fracMaskedOnCpu = true, fracFeasibleOnGpu = true;
            for (int i = 0; i < hosts.size(); i++) {
                boolean can = m.canHost(hosts.get(i), fracTask);
                if (i >= 6 &&  can) fracMaskedOnCpu   = false;
                if (i <= 5 && !can) fracFeasibleOnGpu = false;
            }
            assertTrue("B17d1", "fractional-GPU task masked off CPU-only hosts", fracMaskedOnCpu);
            assertTrue("B17d2", "fractional-GPU task feasible on GPU hosts",     fracFeasibleOnGpu);

            // (e) Pure-CPU task: feasible everywhere (affinity must not over-mask).
            TaskRecord cpuTask = new TaskRecord(
                    "cpu-t", 4000, 1024, 0, 0, "", "BE", "Running",
                    0, 100, 0, 100, 0.5);
            int cpuFeasible = 0;
            for (Host h : hosts) if (m.canHost(h, cpuTask)) cpuFeasible++;
            assertEq("B17e", "pure-CPU task feasible on all 10 hosts", 10, cpuFeasible);

            // (f) idle→suspend gradient still works under heterogeneity.
            while (!m.isDone()) {
                StepResult r = m.step(0);   // pack onto host 0 → hosts 1..9 idle out
                if (r.done()) break;
            }
            int peakSuspended = m.getSnapshots().stream()
                    .mapToInt(MetricsExporter.Snapshot::suspendedHosts)
                    .max().orElse(0);
            System.out.printf("       hetero energy=%.2f kWh, peakSuspended=%d, wakeups=%d%n",
                    m.getTotalEnergyKwh(), peakSuspended, m.getTotalWakeups());
            assertTrue("B17f", "idle→suspend still fires under heterogeneity (≥1 suspended)",
                    peakSuspended >= 1);
        } catch (Exception e) {
            fail("B17", e.toString());
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Helpers
    // ══════════════════════════════════════════════════════════════════════

    private static long sumPeUsage(SimulationManager m) throws Exception {
        Field f = SimulationManager.class.getDeclaredField("hostPeUsage");
        f.setAccessible(true);
        @SuppressWarnings("unchecked")
        Map<Host, Integer> map = (Map<Host, Integer>) f.get(m);
        return map.values().stream().mapToLong(Integer::longValue).sum();
    }

    private static int countTrue(boolean[] arr) {
        int n = 0; for (boolean b : arr) if (b) n++;
        return n;
    }

    private static void assertEq(String tag, String desc, Object expected, Object actual) {
        if (Objects.equals(expected, actual)) pass(tag, desc + " — got " + actual);
        else                                  fail(tag, desc + " — expected=" + expected + " actual=" + actual);
    }

    private static void assertTrue(String tag, String desc, boolean cond) {
        if (cond) pass(tag, desc);
        else      fail(tag, desc);
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B20 (W1.5) — the deadline carries an absolute floor, not only a
    //  bounded-slowdown allowance.
    //
    //  Before the floor, 24.6 % of tasks tolerated under 60 s of lateness and
    //  the median LS budget was 237 s, so the fixed 5 s wake latency was a large
    //  share of the whole budget: suspending a host to save energy scored as an
    //  SLA violation regardless of load, entangling the two objectives through an
    //  artefact of job length.
    //
    //  This test also emits a checksum over the whole trace so the Python mirrors
    //  (workload/deadline.py, eval/qos.py) can be verified to compute bit-identical
    //  deadlines — a cross-language contract, like StepCodec (CLAUDE.md Lưu ý #17).
    // ──────────────────────────────────────────────────────────────────────
    private static void testB20_DeadlineHasAnAbsoluteFloor() {
        try {
            List<TaskRecord> all = AlibabaTraceReader.read(SimulationConfig.TRACE_FILE);
            List<TaskRecord> sched = ScenarioFilter.filter(all, Scenario.LEGACY_HIGH, 42);

            // Floors must be ordered: a stricter class tolerates less lateness.
            double ls   = SimulationConfig.qosToSlackFloorSec("LS");
            double guar = SimulationConfig.qosToSlackFloorSec("Guaranteed");
            double burst= SimulationConfig.qosToSlackFloorSec("Burstable");
            double be   = SimulationConfig.qosToSlackFloorSec("BE");
            assertTrue("B20a", "slack floors ordered LS < Guaranteed < Burstable < BE",
                    ls < guar && guar < burst && burst < be);
            assertTrue("B20b", "every class tolerates at least 100 s",
                    ls >= 100.0 && guar >= 100.0 && burst >= 100.0 && be >= 100.0);
            assertTrue("B20c", "LS floor exceeds the wake latency by a wide margin",
                    ls > 10.0 * SimulationConfig.DEFAULT_POWER.wakeLatencySec());

            int under60 = 0;
            double minBudget = Double.MAX_VALUE;
            double checksum = 0.0;
            for (TaskRecord t : sched) {
                double budget = t.deadline() - t.creationTime() - t.duration();
                if (budget < 60.0) under60++;
                minBudget = Math.min(minBudget, budget);
                checksum += t.deadline();
            }

            System.out.printf("       floors: LS=%.0f Guaranteed=%.0f Burstable=%.0f BE=%.0f s%n",
                    ls, guar, burst, be);
            System.out.printf("       n=%d, min budget=%.1f s, budget<60s: %d%n",
                    sched.size(), minBudget, under60);
            System.out.printf("       deadline checksum = %.6f%n", checksum);

            assertTrue("B20d", "no task tolerates less than 60 s of lateness",
                    under60 == 0);
            assertTrue("B20e", "minimum absolute budget is at least the LS floor",
                    minBudget >= ls - 1e-9);
        } catch (Exception e) {
            fail("B20", e.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B21 (W2.1/W2.2) — trace resolution is backward compatible and fails loudly.
    //
    //  Every property here guards against the same class of accident: a run that
    //  completes normally against the wrong workload. That produces a full set of
    //  plausible numbers and no error, which is the most expensive failure mode
    //  this project has (PLAN-Workload-Model.md risks R5/R6).
    // ──────────────────────────────────────────────────────────────────────
    private static void testB21_TraceResolutionIsSafe() {
        try {
            boolean patterned = SimulationConfig.usesTracePattern();

            if (!patterned) {
                // Phase-1 path: unchanged for every scenario and seed.
                assertTrue("B21a", "no TRACE_PATTERN ⇒ resolution returns TRACE_FILE",
                        SimulationConfig.resolveTracePath("HIGH", 42)
                                .equals(SimulationConfig.TRACE_FILE)
                     && SimulationConfig.resolveTracePath("LOW", 7)
                                .equals(SimulationConfig.TRACE_FILE));
            } else {
                String p = SimulationConfig.resolveTracePath("HIGH", 42);
                assertTrue("B21a", "TRACE_PATTERN resolves to a readable file — " + p,
                        java.nio.file.Files.isReadable(java.nio.file.Path.of(p)));
            }

            // The next two use the explicit-pattern overload rather than poking at env
            // or system properties: resolve() reads the environment first, so a
            // property-based override is a no-op whenever TRACE_PATTERN is actually
            // set — which is exactly when these guards matter most.

            // A pattern without {scenario} maps every scenario onto one file, which
            // would make LOW, HIGH and BURST the same experiment while still running.
            boolean threw = false;
            try {
                SimulationConfig.resolveTracePath("/data/wm1/homo/seed{seed}.csv",
                                                  "HIGH", 42);
            } catch (IllegalStateException expected) {
                threw = true;
            }
            assertTrue("B21b", "pattern without {scenario} is rejected", threw);

            // A missing file must not silently fall back to the legacy trace.
            threw = false;
            try {
                SimulationConfig.resolveTracePath("/nonexistent/{scenario}/seed{seed}.csv",
                                                  "HIGH", 42);
            } catch (IllegalStateException expected) {
                threw = true;
            }
            assertTrue("B21c", "unresolvable pattern raises instead of falling back",
                    threw);

            // An unset pattern is the Phase-1 path regardless of ambient environment.
            assertTrue("B21f", "null/blank pattern resolves to TRACE_FILE",
                    SimulationConfig.resolveTracePath(null, "HIGH", 42)
                            .equals(SimulationConfig.TRACE_FILE)
                 && SimulationConfig.resolveTracePath("  ", "LOW", 7)
                            .equals(SimulationConfig.TRACE_FILE));

            // Scenario.NONE must keep exactly the tasks HIGH keeps: a WM-1 file is
            // already one scenario, so the only filtering left is dropping Pending.
            List<TaskRecord> all = AlibabaTraceReader.read(SimulationConfig.TRACE_FILE);
            List<TaskRecord> none = ScenarioFilter.filter(all, Scenario.NONE, 42);
            List<TaskRecord> high = ScenarioFilter.filter(all, Scenario.LEGACY_HIGH, 42);
            assertTrue("B21d", "Scenario.NONE is a passthrough (same tasks as HIGH)",
                    none.size() == high.size() && none.equals(high));
            assertTrue("B21e", "Scenario.NONE still drops Pending pods",
                    none.stream().noneMatch(t -> "Pending".equals(t.podPhase())));

            System.out.printf("       TRACE_FILE=%s%n", SimulationConfig.TRACE_FILE);
            System.out.printf("       TRACE_PATTERN=%s%n",
                    patterned ? SimulationConfig.tracePattern() : "(unset — Phase-1 path)");
        } catch (Exception e) {
            fail("B21", e.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B22 (W2.5) — a generated trace carries the offered load its manifest claims.
    //
    //  WM-1 calibrates the load in Python; the simulator re-derives it in Java from
    //  the CSV it actually reads. If the two ever disagree — a units slip, a column
    //  mis-parsed, the wrong file resolved — every downstream number is measured
    //  against a workload nobody characterised. Recomputing rho here from the parsed
    //  TaskRecords closes that loop across the language boundary.
    //
    //  rho is the DOMINANT-RESOURCE load, max(rho_cpu, rho_gpu) (PLAN §3.1): the two
    //  arms bind on different resources, so checking only CPU would pass a hetero
    //  trace whose GPU load is completely wrong.
    //
    //  Skipped when /data/wm1 is not mounted, so the legacy configuration stays green.
    // ──────────────────────────────────────────────────────────────────────
    private static void testB22_GeneratedTraceMatchesItsManifest() {
        final double TOL = 0.02;                       // PLAN W1.4 acceptance
        try {
            String root = System.getenv().getOrDefault("WM1_ROOT", "/data/wm1");
            java.nio.file.Path base = java.nio.file.Path.of(root);
            if (!java.nio.file.Files.isDirectory(base)) {
                skip("B22", "no WM-1 traces at " + root
                        + " — run scripts/gen-workloads.sh to enable this check");
                return;
            }

            com.google.gson.Gson gson = new com.google.gson.Gson();
            int checked = 0;
            double worstErr = 0.0;
            String worstWhat = "";

            try (var arms = java.nio.file.Files.list(base)) {
                for (java.nio.file.Path arm : arms.filter(java.nio.file.Files::isDirectory)
                                                  .sorted().toList()) {
                    java.nio.file.Path mf = arm.resolve("wm1-manifest.json");
                    if (!java.nio.file.Files.isReadable(mf)) continue;

                    com.google.gson.JsonObject man = gson.fromJson(
                            java.nio.file.Files.readString(mf),
                            com.google.gson.JsonObject.class);
                    double horizon = man.get("horizon_sec").getAsDouble();
                    com.google.gson.JsonObject cap = man.getAsJsonObject("capacity");
                    int totalPes  = cap.get("total_pes").getAsInt();
                    int totalGpus = cap.get("total_gpus").getAsInt();

                    for (com.google.gson.JsonElement el : man.getAsJsonArray("traces")) {
                        com.google.gson.JsonObject tr = el.getAsJsonObject();
                        // REPLAY has no calibrated target; it is whatever the real
                        // window happened to be, so there is nothing to check it against.
                        if (tr.get("rho_target").isJsonNull()) continue;

                        java.nio.file.Path csv = arm.resolve(tr.get("path").getAsString());
                        if (!java.nio.file.Files.isReadable(csv)) {
                            fail("B22", "manifest lists a missing trace: " + csv);
                            return;
                        }

                        List<TaskRecord> tasks = AlibabaTraceReader.read(csv.toString());
                        if (tasks.size() != tr.get("n_task").getAsInt()) {
                            fail("B22", csv + ": manifest says " + tr.get("n_task").getAsInt()
                                    + " tasks, file has " + tasks.size());
                            return;
                        }

                        double wCpu = 0.0, wGpu = 0.0;
                        for (TaskRecord t : tasks) {
                            wCpu += t.pesNeeded() * t.duration();
                            wGpu += t.numGpu()    * t.duration();
                        }
                        double rho = Math.max(wCpu / (totalPes * horizon),
                                totalGpus > 0 ? wGpu / (totalGpus * horizon) : 0.0);
                        double target = tr.get("rho_target").getAsDouble();
                        double err = Math.abs(rho - target) / target;
                        if (err > worstErr) {
                            worstErr = err;
                            worstWhat = tr.get("arm").getAsString() + "/"
                                      + tr.get("scenario").getAsString() + "/seed"
                                      + tr.get("seed").getAsInt();
                        }
                        checked++;
                    }
                }
            }

            if (checked == 0) {
                skip("B22", "found " + root + " but no calibrated traces in it");
                return;
            }

            System.out.printf("       %d generated traces checked; worst rho error "
                            + "%.3f%% at %s%n", checked, 100 * worstErr, worstWhat);
            assertTrue("B22a", "every generated trace carries its manifest's task count "
                    + "and rho within " + (int) (100 * TOL) + "%", worstErr <= TOL);
        } catch (Exception e) {
            fail("B22", e.toString());
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    //  B23 (W3.1 / W3.3, PLAN §3.9) — an unplaceable task is CHARGED, not free.
    //
    //  Before W3 the `target == null` branch returned reward {0,0} and touched
    //  neither C_SLA nor any counter. Two consequences, both fatal for Phase-2
    //  numbers: dropping a task was strictly cheaper than scheduling it (the
    //  constraint could be satisfied by shedding load rather than placing it
    //  well), and the shortfall was invisible — a run that silently dropped a
    //  third of the trace reported a *better* energy figure, because the tasks
    //  it never ran drew no power.
    //
    //  (a) drives the same trace on a normal cluster and on a deliberately tiny
    //      one, so the only difference is placeability;
    //  (b) pins the charge to κ·(T − creation) task by task — the formula, not
    //      just its sign;
    //  (c) checks the charge dominates what the task would have paid if placed,
    //      which is the property that closes the loophole;
    //  (d) confirms Σ step-cost == getSlaCost() still holds (B16's invariant must
    //      survive the new cost channel);
    //  (e-h) W3.3 acceptance on the calibrated WM-1 traces — see the block comment
    //      there for why "HIGH drops nothing" turned out to be unreachable and what
    //      replaced it. Skipped when /data/wm1 is not mounted.
    // ──────────────────────────────────────────────────────────────────────

    /** Run a whole episode with step(0) (⇒ first-feasible fallback); returns the manager. */
    private static SimulationManager runEpisode(SimulationConfig.DatacenterSpec dc,
                                                String trace, Scenario sc, long seed,
                                                double[] outSummedStepCost,
                                                double[] outWorstUnderCharge) {
        SimulationManager m = new SimulationManager(dc, trace, sc, seed);
        m.resetSimulation();
        double summed = 0.0;
        double worstUnder = Double.POSITIVE_INFINITY;   // min(dropCharge − κ·duration)
        int prevDropped = 0;
        while (!m.isDone()) {
            int idxBefore = m.getCurrentTaskIndex();
            var task = m.getTasks().get(idxBefore);
            StepResult r = m.step(0);
            summed += r.cost();

            if (r.droppedTasks() > prevDropped) {
                // This step is the one that dropped `task`. Re-derive the charge
                // independently of the code under test.
                double expected = task.qosWeight()
                        * Math.max(0.0, m.getEpisodeHorizonSec() - task.creationTime());
                if (Math.abs(r.cost() - expected) > 1e-6) {
                    fail("B23b", String.format(
                            "drop charge for %s is %.3f, expected kappa*(T-creation)=%.3f",
                            task.name(), r.cost(), expected));
                }
                worstUnder = Math.min(worstUnder,
                        r.cost() - task.qosWeight() * task.duration());
            }
            prevDropped = r.droppedTasks();
            if (r.done()) break;
        }
        if (outSummedStepCost != null) outSummedStepCost[0] = summed;
        if (outWorstUnderCharge != null) outWorstUnderCharge[0] = worstUnder;
        return m;
    }

    private static void testB23_DroppedTasksAreCharged() {
        try {
            // ── (a) same trace, two cluster sizes ────────────────────────────
            double[] sumBig = new double[1];
            SimulationManager big = runEpisode(SimulationConfig.DEFAULT_DC,
                    SimulationConfig.TRACE_FILE, Scenario.LEGACY_LOW, 42L, sumBig, null);
            double costBig = big.getSlaCost();
            int droppedBig = big.getDroppedTasks();
            big.shutdown();

            // One host with a single PE and 1 GB of RAM: almost nothing fits.
            SimulationConfig.HostSpec tinyHost = new SimulationConfig.HostSpec(
                    1, SimulationConfig.DEFAULT_HOST.mips(), 1024L,
                    SimulationConfig.DEFAULT_HOST.bwMbps(),
                    SimulationConfig.DEFAULT_HOST.storageMb(), 0,
                    SimulationConfig.DEFAULT_HOST.gpuMemoryMb());
            SimulationConfig.DatacenterSpec tinyDc = new SimulationConfig.DatacenterSpec(
                    1, tinyHost, SimulationConfig.DEFAULT_POWER, 1.0);

            double[] sumTiny = new double[1];
            double[] worstUnder = new double[1];
            SimulationManager tiny = runEpisode(tinyDc,
                    SimulationConfig.TRACE_FILE, Scenario.LEGACY_LOW, 42L,
                    sumTiny, worstUnder);
            double costTiny = tiny.getSlaCost();
            int droppedTiny = tiny.getDroppedTasks();
            int nTasks = tiny.getTasks().size();
            tiny.shutdown();

            System.out.printf("       10-host: dropped=%d, C_SLA=%.1f | "
                            + "1-host-1-PE: dropped=%d/%d, C_SLA=%.1f%n",
                    droppedBig, costBig, droppedTiny, nTasks, costTiny);

            assertTrue("B23a1", "an undersized cluster actually drops tasks "
                    + "(otherwise the rest of B23 proves nothing)", droppedTiny > 0);
            assertTrue("B23a2",
                    "dropping is NOT cheaper than scheduling: C_SLA rises from "
                  + String.format("%.1f to %.1f when tasks stop fitting", costBig, costTiny),
                    costTiny > costBig);

            // ── (c) the charge dominates any placement charge ────────────────
            // Every drop was charged at least kappa*duration, and a placed task can
            // never be charged that much (see SimulationManager.dropCost).
            assertTrue("B23c", "every drop charge covers at least kappa*duration "
                            + "(margin " + String.format("%.1f", worstUnder[0]) + ")",
                    worstUnder[0] >= -1e-6);

            // ── (d) B16's accounting invariant survives the new cost channel ──
            assertTrue("B23d", "sum of step costs == getSlaCost() with drops present",
                    Math.abs(sumTiny[0] - costTiny) <= 1e-6);

            // ── (e-h) W3.3 acceptance on the calibrated WM-1 traces ──────────
            //
            // The plan originally asked for `dropped == 0` on HIGH. Measurement says that
            // is unreachable and says why: the simulator has NO ADMISSION QUEUE — a task
            // is placed at its arrival instant or never — while rho = 0.85 is a *time
            // average*. Under MMPP-2 arrivals the instantaneous demand exceeds 640 PEs
            // regardless of how well the tasks are packed. All five classical baselines
            // drop 1.8-7.1 % of HIGH across seeds 42-46 (best-fit lowest, K8s highest),
            // and none reaches zero. So the criterion below is what the model can actually
            // support, and it is stronger than a bare bound in one respect: it separates
            // the two causes.
            //
            //   LOW  (rho = 0.30)  -> 0 drops, every policy, every seed. This is the
            //                        "the calibration is achievable" check HIGH was meant
            //                        to be; a non-zero here IS a workload bug.
            //   HIGH (rho = 0.85)  -> a small bounded share, reported, never silent.
            //   BURST              -> strictly MORE than HIGH although it carries the SAME
            //                        job multiset at the SAME rho (the split-RNG paired
            //                        design). That isolates arrival burstiness as the
            //                        cause: same work, same load, different concentration.
            //   OVERLOAD (1.25)    -> more still, and flagged if it passes 20 % (risk R10).
            String root = System.getenv().getOrDefault("WM1_ROOT", "/data/wm1");
            java.nio.file.Path low  = java.nio.file.Path.of(root, "homo", "LOW", "seed42.csv");
            java.nio.file.Path high = java.nio.file.Path.of(root, "homo", "HIGH", "seed42.csv");
            java.nio.file.Path burst = java.nio.file.Path.of(root, "homo", "BURST", "seed42.csv");
            java.nio.file.Path over = java.nio.file.Path.of(root, "homo", "OVERLOAD", "seed42.csv");
            if (!java.nio.file.Files.isReadable(low) || !java.nio.file.Files.isReadable(high)
                    || !java.nio.file.Files.isReadable(burst)
                    || !java.nio.file.Files.isReadable(over)) {
                skip("B23e", "no WM-1 homo traces under " + root
                        + " — run scripts/gen-workloads.sh to enable the W3.3 acceptance");
                return;
            }

            // Scenario.NONE: a WM-1 file IS the scenario and must not be re-sliced.
            int[] dropped = new int[4];
            int[] total = new int[4];
            java.nio.file.Path[] files = {low, high, burst, over};
            String[] labels = {"LOW", "HIGH", "BURST", "OVERLOAD"};
            for (int i = 0; i < files.length; i++) {
                SimulationManager m = runEpisode(SimulationConfig.DEFAULT_DC,
                        files[i].toString(), Scenario.NONE, 42L, null, null);
                dropped[i] = m.getDroppedTasks();
                total[i] = m.getTasks().size();
                m.shutdown();
            }

            StringBuilder line = new StringBuilder("       WM-1 homo (first-fit driver):");
            for (int i = 0; i < files.length; i++) {
                line.append(String.format(" %s %d/%d (%.1f%%)", labels[i], dropped[i],
                        total[i], 100.0 * dropped[i] / Math.max(1, total[i])));
            }
            System.out.println(line);

            assertEq("B23e", "homo/LOW (rho=0.30) places every task — a drop at moderate "
                    + "load would mean the workload itself is not placeable", 0, dropped[0]);

            double highRate = (double) dropped[1] / Math.max(1, total[1]);
            assertTrue("B23f", String.format(
                    "homo/HIGH (rho=0.85) drops a bounded, reported share (%.1f%% <= 10%%) "
                  + "— there is no admission queue, so a burst above 640 PEs must go "
                  + "somewhere", 100 * highRate),
                    highRate <= 0.10);

            assertTrue("B23g", String.format(
                    "burstiness alone drives drops: BURST %d > HIGH %d at the SAME rho and "
                  + "the SAME job multiset", dropped[2], dropped[1]),
                    dropped[2] > dropped[1]);

            assertTrue("B23h", "homo/OVERLOAD (rho=1.25) drops more than HIGH and reports it",
                    dropped[3] > dropped[1]);

            // Risk R10: OVERLOAD is only interpretable while most of it still runs.
            double dropRate = (double) dropped[3] / Math.max(1, total[3]);
            if (dropRate > 0.20) {
                System.out.printf("       [R10] OVERLOAD drop rate %.1f%% exceeds 20%% — "
                        + "consider lowering rho toward 1.00 (PLAN risk register)%n",
                        100 * dropRate);
            }

            // ── (i) the heterogeneous arm has a POLICY-INDEPENDENT drop floor ──
            //
            // Measured across all five classical baselines on hetero/HIGH/seed42: 118, 119,
            // 119, 119, 120 drops — a spread of 2 out of ~1 377 tasks. That flatness is the
            // point. With only 18 GPU cards, a GPU task arriving while all 18 are busy has
            // nowhere to go, so the drop count is a property of the *workload against the
            // cluster*, not of the packing decision. Placement still moves energy and
            // tardiness on this arm; it does not move drops.
            //
            // Worth a check rather than only a note: if a future change made hetero/HIGH
            // suddenly placeable, the calibration would have silently drifted.
            java.nio.file.Path hetHigh =
                    java.nio.file.Path.of(root, "hetero", "HIGH", "seed42.csv");
            java.nio.file.Path topo = java.nio.file.Path.of("/config/topology-hetero.json");
            if (!java.nio.file.Files.isReadable(hetHigh)
                    || !java.nio.file.Files.isReadable(topo)) {
                skip("B23i", "hetero trace or /config/topology-hetero.json not mounted");
                return;
            }
            SimulationConfig.DatacenterSpec hetDc =
                    TopologyConfig.load(topo.toString(), SimulationConfig.DEFAULT_DC);
            SimulationManager mHet = runEpisode(hetDc, hetHigh.toString(),
                    Scenario.NONE, 42L, null, null);
            int dHet = mHet.getDroppedTasks();
            int nHet = mHet.getTasks().size();
            mHet.shutdown();

            System.out.printf("       WM-1 hetero/HIGH: dropped=%d/%d (%.1f%%) — "
                            + "GPU-bound, ~identical under every baseline%n",
                    dHet, nHet, 100.0 * dHet / Math.max(1, nHet));
            assertTrue("B23i", String.format(
                    "hetero/HIGH drops more than homo/HIGH at the same nominal rho "
                  + "(%.1f%% vs %.1f%%): 18 cards, not packing quality, is the binding "
                  + "constraint", 100.0 * dHet / Math.max(1, nHet), 100 * highRate),
                    (double) dHet / Math.max(1, nHet) > highRate);
        } catch (Exception e) {
            fail("B23", e.toString());
        }
    }

    private static void pass(String tag, String msg) {
        passes++;
        System.out.println("[PASS] " + tag + " — " + msg);
    }

    private static void fail(String tag, String msg) {
        fails++;
        String line = "[FAIL] " + tag + " — " + msg;
        System.out.println(line);
        failNotes.add(line);
    }

    /** Precondition for this check does not hold in the current configuration. */
    private static void skip(String tag, String msg) {
        skips++;
        String line = "[SKIP] " + tag + " — " + msg;
        System.out.println(line);
        skipNotes.add(line);
    }

    private static void banner(String t) {
        System.out.println();
        System.out.println("============================================================");
        System.out.println("  " + t);
        System.out.println("============================================================");
    }

    private ValidationRunner() {}
}
