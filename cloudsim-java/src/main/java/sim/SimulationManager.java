package sim;

import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.hosts.Host;

import sim.AlibabaTraceReader.TaskRecord;
import sim.DatacenterFactory.GpuState;
import sim.ScenarioFilter.Scenario;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.SynchronousQueue;

/**
 * T2.5 / T8.1 — Central lifecycle manager for the CloudSim Plus simulation,
 * extended with a host state machine (ACTIVE / IDLE / SUSPENDED) so that
 * allocation choices have a measurable energy gradient.
 *
 * <h3>Energy gradient (T8.1)</h3>
 * <p>With the original always-on linear power model
 * ({@code P(U) = P_idle + U·(P_max − P_idle)}), the total energy is
 * approximately {@code N·P_idle·T + (P_max−P_idle)·∫ΣU dt} — the second
 * integral depends only on the <i>total</i> demand, not on the placement
 * decision. Hence K8s ≈ Random to two decimal places, and PPO has no
 * energy gradient. The cheap fix is a host shutdown policy:
 * <ul>
 *   <li><b>ACTIVE</b>: any allocation present → draws
 *       {@code P_idle + span·U + GPU_power}.</li>
 *   <li><b>IDLE</b>: zero allocations, but transition was less than
 *       {@code IDLE_THRESHOLD} seconds ago → still draws the full
 *       {@code P_idle + GPU_idle} bill.</li>
 *   <li><b>SUSPENDED</b>: zero allocations for ≥ {@code IDLE_THRESHOLD}
 *       seconds → draws only {@code P_suspended} (≈ 10 W). Waking it
 *       costs {@code WAKE_ENERGY_KWH} (one-shot) and adds
 *       {@code WAKE_LATENCY_SEC} to the arriving task's completion time.</li>
 * </ul>
 * Pack-friendly policies (BestFit) now leave more hosts SUSPENDED, saving
 * {@code N_off · P_idle · T} — the gradient PPO needs.
 *
 * <h3>Threading model</h3>
 * Simulation runs on a dedicated daemon thread. Python (via Py4J) calls
 * {@link #resetSimulation()} and {@link #step(int)} from the Py4J gateway
 * thread. Two {@link SynchronousQueue}s synchronise the two threads.
 */
public class SimulationManager {

    // ── Host state machine ────────────────────────────────────────────────

    /** Per-host state for energy accounting and Prometheus reporting. */
    public enum HostState {
        SUSPENDED(0),   // power-gated; draws only P_suspended
        IDLE     (1),   // no allocations but recently active; full idle bill
        ACTIVE   (2);   // ≥ 1 allocation; draws P_idle + span·U + GPU power

        public final int code;
        HostState(int code) { this.code = code; }
    }

    // ── Configuration ──────────────────────────────────────────────────────

    /**
     * Active datacenter spec. NOT final — when {@link #readEnvOnReset} is
     * {@code true}, this is re-read from environment variables at every
     * {@link #resetSimulation()} so Phase 1.7 sweeps can vary topology
     * without rebuilding the JVM.
     */
    private SimulationConfig.DatacenterSpec dcSpec;
    private final boolean   readEnvOnReset;
    private final String    traceFile;
    private final Scenario  scenario;
    private final long      seed;

    // ── CloudSim objects (rebuilt on each reset) ───────────────────────────

    private CloudSimPlus      simulation;
    private Datacenter        datacenter;
    private DatacenterBroker  broker;
    private List<Host>        hosts;
    private Map<Host, GpuState> gpuRegistry;

    // ── Trace & stepping state ─────────────────────────────────────────────

    private List<TaskRecord>  tasks;          // filtered, sorted by creationTime
    private int               currentTaskIdx;
    private boolean           episodeDone;

    // ── Manual resource tracking (CloudSim simulation is not started, so
    //    host.getCpuPercentUtilization() / getRam().getAvailableResource()
    //    always return 0/full. We maintain our own accounting instead.) ────────

    private final Map<Host, Integer> hostPeUsage  = new HashMap<>();
    private final Map<Host, Long>    hostRamUsage = new HashMap<>();

    /**
     * Sim-time at which each host's allocation count last dropped to zero
     * (or 0.0 if the host has never carried load, i.e. at simulation start).
     * Used to decide if a host has crossed the IDLE → SUSPENDED threshold.
     * Removed from the map when the host transitions back to ACTIVE.
     */
    private final Map<Host, Double>  hostZeroLoadSince = new HashMap<>();

    /**
     * Pending task completions ordered by end-time. Each entry holds the
     * resources to give back to its host when simulated time reaches
     * {@link TaskCompletion#endTime()}.
     */
    private final PriorityQueue<TaskCompletion> globalCompletions =
            new PriorityQueue<>(Comparator.comparingDouble(TaskCompletion::endTime));

    private record TaskCompletion(
            double endTime, Host host, int pes, long ramMib, int gpus) {}

    // ── Energy tracking ────────────────────────────────────────────────────

    private double cumulativeCpuEnergyWs;  // Watt-seconds
    private double cumulativeGpuEnergyWs;
    private double cumulativeWakeEnergyWs; // T8.1 — one-shot wake-up bonuses
    private double lastEnergyTimestamp;
    private int    totalWakeups;           // T8.1 — counter for state-machine telemetry

    // ── Metric history ─────────────────────────────────────────────────────

    private final List<MetricsExporter.Snapshot> snapshots = new ArrayList<>();
    private int slaViolationCount;

    // ── Thread synchronisation (for RL stepping via Py4J) ──────────────────

    private final SynchronousQueue<StepResult> stepResultQueue = new SynchronousQueue<>();
    private final SynchronousQueue<Integer>    actionQueue     = new SynchronousQueue<>();
    private Thread simThread;

    // ── Step result returned to the RL agent ───────────────────────────────

    public record StepResult(
        double[] observation,
        double[] reward,       // [R_energy, R_sla]
        boolean  done,
        int      taskIndex,
        String   taskName
    ) {}

    // ── Constructors ───────────────────────────────────────────────────────

    /**
     * Pinned-spec constructor — uses the supplied {@link SimulationConfig.DatacenterSpec}
     * and does NOT re-read env on reset. Suitable for deterministic unit tests.
     */
    public SimulationManager(SimulationConfig.DatacenterSpec dcSpec,
                             String traceFile,
                             Scenario scenario,
                             long seed) {
        this.dcSpec         = dcSpec;
        this.readEnvOnReset = false;
        this.traceFile      = traceFile;
        this.scenario       = scenario;
        this.seed           = seed;
    }

    /**
     * Env-driven constructor — reads {@link SimulationConfig#fromEnv()} on
     * every {@link #resetSimulation()} so Phase 1.7 topology sweeps work
     * without a container rebuild.
     */
    public SimulationManager(String traceFile, Scenario scenario, long seed) {
        this.dcSpec         = SimulationConfig.fromEnv();
        this.readEnvOnReset = true;
        this.traceFile      = traceFile;
        this.scenario       = scenario;
        this.seed           = seed;
    }

    /** Convenience constructor with all defaults (HIGH scenario). */
    public SimulationManager() {
        this(SimulationConfig.TRACE_FILE, Scenario.HIGH,
             Long.parseLong(System.getenv().getOrDefault("RANDOM_SEED", "42")));
    }

    // ════════════════════════════════════════════════════════════════════════
    //  PUBLIC API — called by Python (Py4J) or by Java tests
    // ════════════════════════════════════════════════════════════════════════

    /**
     * (Re)initialise the simulation and return the first observation.
     * Blocks until the simulation is ready for the first action.
     */
    public StepResult resetSimulation() {
        terminateExisting();
        buildSimulation();

        // Start the stepping loop on a daemon thread
        simThread = new Thread(this::steppingLoop, "cloudsim-step");
        simThread.setDaemon(true);
        simThread.start();

        try {
            return stepResultQueue.take();   // blocks until first obs ready
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted during reset", e);
        }
    }

    /**
     * Provide the RL agent's action and receive the next observation.
     *
     * @param hostIndex index into {@link #getHosts()} (0-based)
     * @return next observation, reward, and done flag
     */
    public StepResult step(int hostIndex) {
        if (episodeDone) {
            throw new IllegalStateException("Episode is done — call resetSimulation()");
        }
        try {
            actionQueue.put(hostIndex);
            return stepResultQueue.take();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted during step", e);
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  ACCESSORS
    // ════════════════════════════════════════════════════════════════════════

    public List<Host>       getHosts()       { return hosts; }
    public int              getHostCount()   { return hosts.size(); }
    public List<TaskRecord> getTasks()       { return tasks; }
    public int              getCurrentTaskIndex() { return currentTaskIdx; }
    public boolean          isDone()         { return episodeDone; }

    public Map<Host, GpuState> getGpuRegistry() { return gpuRegistry; }

    public List<MetricsExporter.Snapshot> getSnapshots() { return snapshots; }

    /** Total energy consumed so far (kWh) — CPU + GPU + wake-up bonuses. */
    public double getTotalEnergyKwh() {
        return (cumulativeCpuEnergyWs + cumulativeGpuEnergyWs
              + cumulativeWakeEnergyWs) / 3_600_000.0;
    }

    /** Cumulative number of SUSPENDED → ACTIVE wake-up events. */
    public int getTotalWakeups() { return totalWakeups; }

    /** Cumulative one-shot wake-up energy in kWh (subset of getTotalEnergyKwh). */
    public double getWakeEnergyKwh() {
        return cumulativeWakeEnergyWs / 3_600_000.0;
    }

    /** Currently-active datacenter spec (varies across resets in env mode). */
    public SimulationConfig.DatacenterSpec getDatacenterSpec() { return dcSpec; }

    /** Number of action-mask-valid hosts for the current task. */
    public boolean[] getActionMask() {
        if (episodeDone || currentTaskIdx >= tasks.size()) {
            return new boolean[hosts.size()];
        }
        TaskRecord task = tasks.get(currentTaskIdx);
        boolean[] mask = new boolean[hosts.size()];
        for (int i = 0; i < hosts.size(); i++) {
            mask[i] = canHost(hosts.get(i), task);
        }
        return mask;
    }

    // ════════════════════════════════════════════════════════════════════════
    //  SIMULATION BUILD
    // ════════════════════════════════════════════════════════════════════════

    private void buildSimulation() {
        // T7.1 — Re-read env vars on every reset so topology sweeps work
        // without a JVM rebuild. The pinned-spec constructor opts out.
        if (readEnvOnReset) {
            dcSpec = SimulationConfig.fromEnv();
        }

        // 1. CloudSim engine
        simulation = new CloudSimPlus();

        // 2. Datacenter + GPU registry
        // IdentityHashMap uses object-reference equality (==), ensuring each
        // Host is stored as a distinct key regardless of its getId() value.
        gpuRegistry = new java.util.IdentityHashMap<>();
        datacenter  = DatacenterFactory.create(simulation, dcSpec, gpuRegistry);

        // 3. Hosts list — use the datacenter's authoritative ordered list,
        //    not gpuRegistry.keySet() (IdentityHashMap has no stable order).
        hosts = new ArrayList<>(datacenter.getHostList());

        // 4. Broker
        broker = new DatacenterBrokerSimple(simulation);

        // 5. Load trace
        try {
            List<TaskRecord> allTasks = AlibabaTraceReader.read(traceFile);
            tasks = ScenarioFilter.filter(allTasks, scenario, seed);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load trace: " + traceFile, e);
        }

        // 6. Reset counters and per-host state
        currentTaskIdx          = 0;
        episodeDone             = false;
        cumulativeCpuEnergyWs   = 0;
        cumulativeGpuEnergyWs   = 0;
        cumulativeWakeEnergyWs  = 0;
        lastEnergyTimestamp     = 0;
        totalWakeups            = 0;
        slaViolationCount       = 0;
        hostPeUsage.clear();
        hostRamUsage.clear();
        hostZeroLoadSince.clear();
        globalCompletions.clear();
        snapshots.clear();

        // T8.1 — Every host starts IDLE at t=0 (counter armed at 0). If no
        // task arrives within IDLE_THRESHOLD, the host auto-suspends; the
        // first arrival on a suspended host pays the wake-up cost. Modelled
        // as a warm boot — we do not start in SUSPENDED, because the
        // assumption "cluster was just brought up" matches how scenarios
        // are launched in our trace.
        for (Host h : hosts) {
            hostZeroLoadSince.put(h, 0.0);
        }

        System.out.printf("[SimulationManager] Built simulation: %d hosts, %d tasks (%s) "
                        + "| idle→suspend %.1fs, wake %.4f kWh+%.1fs%n",
                hosts.size(), tasks.size(), scenario,
                dcSpec.powerSpec().idleThresholdSec(),
                dcSpec.powerSpec().wakeEnergyKwh(),
                dcSpec.powerSpec().wakeLatencySec());

        // T5.3 — Reset per-episode gauges and seed the initial backlog.
        // Safe no-op when monitoring is disabled.
        MetricsRegistry.setContext(scenario.name(), MetricsRegistry.schedulerTag());
        MetricsRegistry.onEpisodeStart();
        MetricsRegistry.setPendingTasks(tasks.size());
    }

    // ════════════════════════════════════════════════════════════════════════
    //  STEPPING LOOP  (runs on simThread)
    // ════════════════════════════════════════════════════════════════════════

    private void steppingLoop() {
        try {
            // Emit initial observation (no reward yet)
            stepResultQueue.put(new StepResult(
                    buildObservation(),
                    new double[]{0.0, 0.0},
                    false,
                    currentTaskIdx,
                    tasks.isEmpty() ? "" : tasks.get(currentTaskIdx).name()
            ));

            while (currentTaskIdx < tasks.size()) {
                // Wait for action from Python
                int hostIdx = actionQueue.take();

                // Validate
                if (hostIdx < 0 || hostIdx >= hosts.size()) {
                    System.err.printf("[SimulationManager] Invalid host index %d, "
                                   + "clamping to 0%n", hostIdx);
                    hostIdx = 0;
                }

                TaskRecord task = tasks.get(currentTaskIdx);
                Host target = hosts.get(hostIdx);

                // 1. Advance simulated clock up to this task's arrival,
                //    integrating power piecewise and releasing any allocations
                //    that finished in the meantime. State seen by step 2 is
                //    therefore the cluster state immediately BEFORE the new
                //    arrival — not contaminated by it.
                advanceEnergy(task.creationTime());

                // 1b. Defensive feasibility check. MaskablePPO and our
                //     baseline policies respect the action mask, so this
                //     fallback is a no-op in the normal path.
                if (!canHost(target, task)) {
                    target = null;
                    for (Host h : hosts) {
                        if (canHost(h, task)) { target = h; break; }
                    }
                }

                // 2. Apply action (if any feasible host found). Schedules
                //    a future release at creationTime + duration (+ wake latency
                //    if the host was suspended).
                double[] reward;
                if (target != null) {
                    boolean wokeUp = allocateTask(task, target);
                    reward = computeReward(task, target, wokeUp);
                } else {
                    System.err.printf(
                        "[SimulationManager] Task %s dropped: no feasible host "
                      + "(needs cpu=%d, ram=%d, gpu=%d)%n",
                        task.name(), task.pesNeeded(), task.memoryMib(), task.numGpu());
                    reward = new double[]{0.0, 0.0};
                }

                // 3. Advance bookkeeping
                currentTaskIdx++;
                episodeDone = (currentTaskIdx >= tasks.size());

                // 4. If the episode just ended, flush remaining energy
                //    accounting up to the last scheduled completion so
                //    that the final snapshot's makespan and totals are
                //    physically meaningful.
                if (episodeDone) {
                    flushTillEnd();
                }

                // 5. Snapshot uses the CURRENT lastEnergyTimestamp as its
                //    timestamp — this is the new task's creationTime, or
                //    the last-completion time when episode just ended.
                recordSnapshot(lastEnergyTimestamp);

                // Emit result
                stepResultQueue.put(new StepResult(
                        buildObservation(),
                        reward,
                        episodeDone,
                        currentTaskIdx,
                        episodeDone ? "" : tasks.get(currentTaskIdx).name()
                ));
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  TASK ALLOCATION
    // ════════════════════════════════════════════════════════════════════════

    /**
     * Apply one allocation and schedule its future release.
     *
     * @return {@code true} if the host was in {@link HostState#SUSPENDED}
     *         immediately before this allocation — in which case the caller
     *         (computeReward / energy counters) must include the wake-up
     *         cost. Returning the flag instead of stashing it in a field
     *         keeps the state machine localised.
     */
    private boolean allocateTask(TaskRecord task, Host host) {
        // Detect wake-up BEFORE we mutate state.
        HostState prevState = stateOf(host, lastEnergyTimestamp);
        boolean wokeUp = prevState == HostState.SUSPENDED;

        // T8.1 — One-shot wake-up energy. Bookkeep on the CPU energy line
        // because it represents whole-host power-on overhead (PSU inrush +
        // boot). The reward path uses getWakeEnergyKwh() separately so
        // counters never double-count.
        if (wokeUp) {
            double wakeWs = dcSpec.powerSpec().wakeEnergyKwh() * 3_600_000.0;
            cumulativeWakeEnergyWs += wakeWs;
            totalWakeups++;
            MetricsRegistry.incWakeup();
        }

        // Manual resource tracking.
        // broker.submitVm() internally calls simulation.schedule() which requires
        // a started CloudSim clock — omitting it prevents silent JVM crashes.
        hostPeUsage.merge(host, task.pesNeeded(), Integer::sum);
        hostRamUsage.merge(host, (long) task.memoryMib(), Long::sum);

        if (task.numGpu() > 0) {
            GpuState gpuState = gpuRegistry.get(host);
            if (gpuState != null) {
                gpuState.allocate(task.numGpu());
            }
        }

        // T8.1 — Host now has load; clear its zero-load timestamp so future
        // state checks don't mistakenly classify it as IDLE/SUSPENDED.
        hostZeroLoadSince.remove(host);

        // Schedule the future release. With wake-up, the task starts
        // δ_wake seconds later, so its completion slips by the same amount.
        double wakeLatency = wokeUp ? dcSpec.powerSpec().wakeLatencySec() : 0.0;
        double endTime = task.creationTime() + wakeLatency + task.duration();
        globalCompletions.offer(new TaskCompletion(
                endTime, host, task.pesNeeded(),
                (long) task.memoryMib(), task.numGpu()));

        // T5.3 — Cumulative placement counter (Prometheus monotonic).
        MetricsRegistry.incTasksScheduled();

        return wokeUp;
    }

    /**
     * Apply one completion event: give resources back to its host.
     * When this drops the host to zero allocations, start the IDLE
     * countdown by stamping {@link #hostZeroLoadSince}.
     */
    private void releaseAllocation(TaskCompletion c) {
        hostPeUsage.computeIfPresent(c.host(),
                (h, v) -> Math.max(0, v - c.pes()));
        hostRamUsage.computeIfPresent(c.host(),
                (h, v) -> Math.max(0L, v - c.ramMib()));
        if (c.gpus() > 0) {
            GpuState g = gpuRegistry.get(c.host());
            if (g != null) g.release(c.gpus());
        }

        // T8.1 — Did this release bring the host to zero load? If yes,
        // stamp the timestamp so the suspend countdown can start.
        if (hasNoLoad(c.host()) && !hostZeroLoadSince.containsKey(c.host())) {
            hostZeroLoadSince.put(c.host(), lastEnergyTimestamp);
        }
    }

    /** True iff the host carries no CPU, RAM, or GPU allocations. */
    private boolean hasNoLoad(Host h) {
        int pes = hostPeUsage.getOrDefault(h, 0);
        if (pes > 0) return false;
        GpuState g = gpuRegistry.get(h);
        if (g != null && g.used() > 0) return false;
        return true;
    }

    /**
     * Classify a host's current state for energy accounting / telemetry.
     *
     * @param now sim-time at which the state is being observed
     */
    private HostState stateOf(Host h, double now) {
        if (!hasNoLoad(h)) return HostState.ACTIVE;
        Double zeroSince = hostZeroLoadSince.get(h);
        if (zeroSince == null) {
            // Defensive: a host with no load AND no stamp must have just
            // released; treat as IDLE starting now.
            return HostState.IDLE;
        }
        double threshold = dcSpec.powerSpec().idleThresholdSec();
        return (now - zeroSince) >= threshold ? HostState.SUSPENDED : HostState.IDLE;
    }

    // ════════════════════════════════════════════════════════════════════════
    //  RESOURCE STATE ACCESSORS  (used by baseline allocation policies)
    // ════════════════════════════════════════════════════════════════════════

    /** Number of CPU PEs currently free on {@code host}. */
    public int freePes(Host host) {
        return dcSpec.hostSpec().pesCount() - hostPeUsage.getOrDefault(host, 0);
    }

    /** RAM (MiB) currently free on {@code host}. */
    public long freeRam(Host host) {
        return dcSpec.hostSpec().ramMb() - hostRamUsage.getOrDefault(host, 0L);
    }

    /** GPUs currently free on {@code host} (or 0 if not in the registry). */
    public int freeGpus(Host host) {
        GpuState gpu = gpuRegistry.get(host);
        return (gpu != null) ? gpu.available() : 0;
    }

    /** Per-host capacity normaliser for K8s-style scoring. */
    public SimulationConfig.HostSpec hostSpec() {
        return dcSpec.hostSpec();
    }

    /** Check if a host can accept a task (CPU PEs + RAM + GPU). */
    public boolean canHost(Host host, TaskRecord task) {
        return freePes(host)  >= task.pesNeeded()
            && freeRam(host)  >= task.memoryMib()
            && freeGpus(host) >= task.numGpu();
    }

    // ════════════════════════════════════════════════════════════════════════
    //  REWARD COMPUTATION
    // ════════════════════════════════════════════════════════════════════════

    /**
     * Compute the two-component reward vector:
     *   R_energy = −ΔE  (negative energy delta in Watt-seconds)
     *   R_sla    = −λ × max(0, estimated_completion − deadline)
     *
     * <p>If the allocation woke a SUSPENDED host, both components carry
     * its cost: R_energy gains the wake-up bonus, R_sla integrates the
     * wake-up latency into the completion estimate.
     */
    private double[] computeReward(TaskRecord task, Host host, boolean wokeUp) {
        SimulationConfig.PowerSpec ps = dcSpec.powerSpec();
        int pesPerHost = dcSpec.hostSpec().pesCount();

        // ── R_energy: incremental energy from this allocation ──
        int usedPes = hostPeUsage.getOrDefault(host, 0);
        double cpuUtil = Math.min(1.0, (double) usedPes / pesPerHost);
        double cpuPower = ps.cpuIdlePowerWatt()
                + (ps.cpuMaxPowerWatt() - ps.cpuIdlePowerWatt()) * cpuUtil;
        GpuState gpu = gpuRegistry.get(host);
        double gpuPower = (gpu != null)
                ? DatacenterFactory.gpuPowerWatt(gpu, ps)
                : 0.0;
        double deltaEnergy = (cpuPower + gpuPower) * dcSpec.schedulingIntervalSec();
        if (wokeUp) {
            // Wake-up bonus in Watt-seconds, so it lives on the same scale
            // as deltaEnergy — agent feels the hit directly.
            deltaEnergy += ps.wakeEnergyKwh() * 3_600_000.0;
        }
        double rEnergy = -deltaEnergy;

        // ── R_sla: penalty for estimated deadline miss ──
        //
        // Contention proxy: how loaded was the host BEFORE this task arrived?
        // bgUsedPes = current usage minus what we just added.
        int bgUsedPes = Math.max(0, usedPes - task.pesNeeded());
        double bgUtil = Math.min(1.0, (double) bgUsedPes / pesPerHost);
        double congestionFactor = 1.0 + bgUtil;

        double wakeLatency = wokeUp ? ps.wakeLatencySec() : 0.0;
        double estimatedCompletion =
                task.creationTime() + wakeLatency + task.duration() * congestionFactor;
        double slaSlack = estimatedCompletion - task.deadline();
        double rSla    = -task.slaLambda() * Math.max(0.0, slaSlack);

        if (slaSlack > 1.0) {
            slaViolationCount++;
            // T5.3 — Live SLA violation counter, broken down by QoS class.
            MetricsRegistry.incSlaViolation(task.qos());
        }

        return new double[]{rEnergy, rSla};
    }

    // ════════════════════════════════════════════════════════════════════════
    //  ENERGY ACCOUNTING
    // ════════════════════════════════════════════════════════════════════════

    /**
     * Advance the simulated clock from {@link #lastEnergyTimestamp} up to
     * {@code toTime}, integrating power consumption piecewise.
     *
     * <p>The integration is broken at three kinds of state-change events:
     * <ol>
     *   <li>{@code toTime} (the caller's target — usually next task arrival)</li>
     *   <li>The next pending task completion (releases resources)</li>
     *   <li>The next IDLE → SUSPENDED transition for any host (changes its
     *       per-second power draw from {@code P_idle+GPU_idle} down to
     *       {@code P_suspended})</li>
     * </ol>
     * Sub-intervals are integrated with the state that holds during them,
     * which is why the state-machine maths is correct without per-host
     * scheduled-event entries in {@code globalCompletions}.
     *
     * <p>Idempotent for {@code toTime <= lastEnergyTimestamp}.
     */
    private void advanceEnergy(double toTime) {
        if (toTime <= lastEnergyTimestamp) return;

        while (true) {
            // Earliest pending completion time, or +∞ if none queued.
            double nextCompletion = globalCompletions.isEmpty()
                    ? Double.POSITIVE_INFINITY
                    : globalCompletions.peek().endTime();

            // T8.1 — Earliest IDLE → SUSPENDED transition. Scan all hosts
            // with a zero-load stamp; the smallest stamp+threshold strictly
            // greater than lastEnergyTimestamp wins.
            double nextSuspend = nextSuspendTransition();

            double tStop = Math.min(toTime, Math.min(nextCompletion, nextSuspend));
            double dt    = tStop - lastEnergyTimestamp;

            if (dt > 0) {
                integratePowerOverInterval(lastEnergyTimestamp, tStop);
            }
            lastEnergyTimestamp = tStop;

            // Process all completions occurring at or before tStop.
            // Done AFTER integration so released resources stop drawing
            // power FROM tStop ONWARDS — not before.
            while (!globalCompletions.isEmpty()
                    && globalCompletions.peek().endTime() <= lastEnergyTimestamp) {
                releaseAllocation(globalCompletions.poll());
            }

            if (lastEnergyTimestamp >= toTime) break;
        }
    }

    /**
     * Earliest pending IDLE → SUSPENDED transition strictly after
     * {@link #lastEnergyTimestamp}, or {@link Double#POSITIVE_INFINITY}
     * when no host is currently in the IDLE window.
     */
    private double nextSuspendTransition() {
        double threshold = dcSpec.powerSpec().idleThresholdSec();
        double best = Double.POSITIVE_INFINITY;
        for (Map.Entry<Host, Double> e : hostZeroLoadSince.entrySet()) {
            double transition = e.getValue() + threshold;
            // Strict >; equality means the transition already happened.
            if (transition > lastEnergyTimestamp && transition < best) {
                best = transition;
            }
        }
        return best;
    }

    /**
     * Integrate cluster power consumption over {@code [t0, t1]}. Each host
     * is classified per its state at {@code t0} (the state is constant on
     * this sub-interval thanks to {@link #advanceEnergy} breaking on every
     * transition):
     * <ul>
     *   <li>ACTIVE    → {@code P_idle + span·U + GPU_active_power}</li>
     *   <li>IDLE      → {@code P_idle + GPU_idle_power}</li>
     *   <li>SUSPENDED → {@code P_suspended} only</li>
     * </ul>
     */
    private void integratePowerOverInterval(double t0, double t1) {
        double dt = t1 - t0;
        if (dt <= 0) return;

        SimulationConfig.PowerSpec ps = dcSpec.powerSpec();
        double cpuIdle = ps.cpuIdlePowerWatt();
        double cpuSpan = ps.cpuMaxPowerWatt() - cpuIdle;
        double pSus    = ps.suspendedPowerWatt();
        int    pesPer  = dcSpec.hostSpec().pesCount();

        for (Host host : hosts) {
            HostState st = stateOf(host, t0);

            switch (st) {
                case ACTIVE -> {
                    int usedPes = hostPeUsage.getOrDefault(host, 0);
                    double util = Math.min(1.0, (double) usedPes / pesPer);
                    cumulativeCpuEnergyWs += (cpuIdle + cpuSpan * util) * dt;
                    GpuState gpu = gpuRegistry.get(host);
                    if (gpu != null) {
                        cumulativeGpuEnergyWs +=
                                DatacenterFactory.gpuPowerWatt(gpu, ps) * dt;
                    }
                }
                case IDLE -> {
                    cumulativeCpuEnergyWs += cpuIdle * dt;
                    GpuState gpu = gpuRegistry.get(host);
                    if (gpu != null) {
                        // gpu.used() == 0 here, so this is the all-cards-idle bill.
                        cumulativeGpuEnergyWs +=
                                DatacenterFactory.gpuPowerWatt(gpu, ps) * dt;
                    }
                }
                case SUSPENDED -> {
                    // Whole-host suspended budget — replaces CPU & GPU idle bills.
                    cumulativeCpuEnergyWs += pSus * dt;
                }
            }
        }
    }

    /**
     * After the last task has been allocated, advance the clock to the
     * latest scheduled completion and process every remaining release.
     * Without this, energy/utilisation for the long tail of running tasks
     * (those still consuming after the last arrival) would not be counted.
     */
    private void flushTillEnd() {
        if (globalCompletions.isEmpty()) return;
        double latestEnd = lastEnergyTimestamp;
        for (TaskCompletion c : globalCompletions) {
            if (c.endTime() > latestEnd) latestEnd = c.endTime();
        }
        advanceEnergy(latestEnd);
    }

    // ════════════════════════════════════════════════════════════════════════
    //  OBSERVATION BUILDER
    // ════════════════════════════════════════════════════════════════════════

    /**
     * Build a flat observation vector for the RL agent.
     *
     * <p>Layout (H = number of hosts):
     * <pre>
     *   [0..H-1]      host CPU utilisation       (0.0–1.0)
     *   [H..2H-1]     host memory utilisation    (0.0–1.0)
     *   [2H..3H-1]    host GPU utilisation       (0.0–1.0)
     *   [3H..6H-1]    host state (one-hot 3-way; T8.9):
     *                   [3H+3i]   = 1 if SUSPENDED (state code 0)
     *                   [3H+3i+1] = 1 if IDLE      (state code 1)
     *                   [3H+3i+2] = 1 if ACTIVE    (state code 2)
     *   [6H..6H+3]    current task: [cpu_norm, mem_norm, gpu_norm, qos_lambda]
     * </pre>
     * Total length = <b>6H + 4</b>.
     *
     * <p>The host-state one-hot encoding is the signal PPO-min needs to learn
     * that picking a SUSPENDED host costs extra energy and adds latency. Without
     * it, the agent has no way to differentiate {U=0, IDLE} from {U=0, SUSPENDED}
     * — both look like "free host" through CPU/MEM/GPU util alone.
     */
    private double[] buildObservation() {
        int h = hosts.size();
        double[] obs = new double[6 * h + 4];

        for (int i = 0; i < h; i++) {
            Host host = hosts.get(i);
            int  usedPes = hostPeUsage.getOrDefault(host, 0);
            long usedRam = hostRamUsage.getOrDefault(host, 0L);
            obs[i]         = Math.min(1.0, (double) usedPes / dcSpec.hostSpec().pesCount());
            obs[h + i]     = Math.min(1.0, (double) usedRam / dcSpec.hostSpec().ramMb());
            GpuState gpu   = gpuRegistry.get(host);
            obs[2 * h + i] = (gpu != null) ? gpu.utilization() : 0.0;

            // One-hot state encoding. stateOf reads the same maps used by
            // the energy integrator, so the agent sees a coherent picture.
            HostState st = stateOf(host, lastEnergyTimestamp);
            obs[3 * h + 3 * i + st.code] = 1.0;
        }

        // Current task features (normalised to [0, 1] based on host capacity)
        if (!episodeDone && currentTaskIdx < tasks.size()) {
            TaskRecord t = tasks.get(currentTaskIdx);
            SimulationConfig.HostSpec hs = dcSpec.hostSpec();
            obs[6 * h]     = (double) t.pesNeeded()  / hs.pesCount();
            obs[6 * h + 1] = (double) t.memoryMib()  / hs.ramMb();
            obs[6 * h + 2] = (double) t.numGpu()     / Math.max(1, hs.gpuCount());
            obs[6 * h + 3] = t.slaLambda() / 3.0;  // normalise: max λ = 3.0
        }

        return obs;
    }

    // ════════════════════════════════════════════════════════════════════════
    //  METRICS RECORDING
    // ════════════════════════════════════════════════════════════════════════

    private void recordSnapshot(double timestamp) {
        double cpuKwh  = cumulativeCpuEnergyWs  / 3_600_000.0;
        double gpuKwh  = cumulativeGpuEnergyWs  / 3_600_000.0;
        double wakeKwh = cumulativeWakeEnergyWs / 3_600_000.0;
        // Wake-up energy is accounted on the CPU energy line (whole-host
        // boot overhead) for the totalEnergyKwh aggregate; the dedicated
        // counter is exported separately for diagnostics.
        double totalKwh = cpuKwh + gpuKwh + wakeKwh;

        int pesPerHost = dcSpec.hostSpec().pesCount();
        int nHosts = hosts.size();

        double[] hostCpuUtil = new double[nHosts];
        int[]    hostStates  = new int[nHosts];
        int active = 0, idle = 0, suspended = 0;

        for (int i = 0; i < nHosts; i++) {
            Host h = hosts.get(i);
            int  usedPes = hostPeUsage.getOrDefault(h, 0);
            double util = Math.min(1.0, (double) usedPes / pesPerHost);
            hostCpuUtil[i] = util;
            HostState st = stateOf(h, timestamp);
            hostStates[i] = st.code;
            switch (st) {
                case ACTIVE    -> active++;
                case IDLE      -> idle++;
                case SUSPENDED -> suspended++;
            }
        }

        double avgCpuUtil = Arrays.stream(hostCpuUtil).average().orElse(0.0);
        double avgGpuUtil = MetricsExporter.averageGpuUtilization(gpuRegistry);
        double avgSlack   = 0;  // simplified — detailed tracking in Phase 2

        snapshots.add(new MetricsExporter.Snapshot(
                timestamp,
                cpuKwh + wakeKwh,   // bundle wake bonus into CPU line for the cumulative column
                gpuKwh, totalKwh,
                currentTaskIdx, slaViolationCount, avgSlack,
                avgCpuUtil, avgGpuUtil,
                active, idle, suspended,
                totalWakeups, wakeKwh,
                hostCpuUtil, hostStates));

        // T5.3 / T8.2 — Push per-host gauges + episode-scoped totals.
        try {
            pushHostGaugesToPrometheus(hostCpuUtil, hostStates);
            MetricsRegistry.setTotalEnergyKwh(totalKwh);
            MetricsRegistry.setSimClock(timestamp);
            MetricsRegistry.setPendingTasks(
                    Math.max(0, tasks.size() - currentTaskIdx));
            MetricsRegistry.setHostStateCounts(active, idle, suspended);
        } catch (Throwable t) {
            System.err.printf(
                "[SimulationManager] Prometheus push failed: %s (swallowed)%n", t);
        }
    }

    /**
     * Push current per-host utilisation + instantaneous power draw + state
     * to the metrics façade. Index-based host_id matches the observation
     * layout, so Prometheus labels align 1:1 with the agent's view.
     */
    private void pushHostGaugesToPrometheus(double[] hostCpuUtil, int[] hostStates) {
        if (!MetricsRegistry.isEnabled()) return; // skip the loop entirely
        SimulationConfig.HostSpec  hs = dcSpec.hostSpec();
        SimulationConfig.PowerSpec ps = dcSpec.powerSpec();
        double cpuIdle = ps.cpuIdlePowerWatt();
        double cpuSpan = ps.cpuMaxPowerWatt() - cpuIdle;
        double pSus    = ps.suspendedPowerWatt();

        for (int i = 0; i < hosts.size(); i++) {
            Host h = hosts.get(i);
            long usedRam = hostRamUsage.getOrDefault(h, 0L);
            double cpu = hostCpuUtil[i];
            double mem = Math.min(1.0, (double) usedRam / hs.ramMb());

            GpuState gpu = gpuRegistry.get(h);
            double gpuUtil = (gpu != null) ? gpu.utilization() : 0.0;

            // Power draw matches integratePowerOverInterval() exactly so that
            // Grafana power panels and the metrics.csv energy column tell
            // the same story.
            double power;
            switch (hostStates[i]) {
                case 2 -> power = cpuIdle + cpuSpan * cpu
                                  + ((gpu != null) ? DatacenterFactory.gpuPowerWatt(gpu, ps) : 0.0);
                case 1 -> power = cpuIdle
                                  + ((gpu != null) ? DatacenterFactory.gpuPowerWatt(gpu, ps) : 0.0);
                default -> power = pSus;
            }

            MetricsRegistry.recordHost(i, cpu, mem, gpuUtil, power);
            MetricsRegistry.setHostState(i, hostStates[i]);
        }
    }

    /**
     * Export metrics after the episode ends.
     * Call this from the gateway or test harness.
     */
    public void exportMetrics(String outputDir) {
        try {
            MetricsExporter.writeCsv(outputDir + "/metrics.csv", snapshots);
            MetricsExporter.Summary summary = MetricsExporter.buildSummary(snapshots);
            MetricsExporter.writeJson(outputDir + "/summary.json", summary);
        } catch (IOException e) {
            System.err.printf("[SimulationManager] Failed to export metrics: %s%n",
                    e.getMessage());
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  LIFECYCLE
    // ════════════════════════════════════════════════════════════════════════

    /** Terminate any running simulation thread. */
    private void terminateExisting() {
        if (simThread != null && simThread.isAlive()) {
            simThread.interrupt();
            try {
                simThread.join(2000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        // Drain queues to prevent stale data
        stepResultQueue.poll();
        actionQueue.poll();
    }

    /** Gracefully shut down (call when the JVM is exiting). */
    public void shutdown() {
        terminateExisting();
        if (simulation != null) {
            simulation.terminate();
        }
        System.out.println("[SimulationManager] Shutdown complete.");
    }
}
