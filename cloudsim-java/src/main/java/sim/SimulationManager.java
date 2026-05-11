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
 * T2.5 — Central lifecycle manager for the CloudSim Plus simulation.
 *
 * Responsibilities:
 * <ol>
 *   <li>Build / tear-down the simulation (datacenter, broker, trace)</li>
 *   <li>RL stepping: pause at each task arrival, wait for action, resume</li>
 *   <li>Track GPU state and compute energy / SLA metrics per step</li>
 *   <li>Thread-safe reset for multiple RL episodes</li>
 * </ol>
 *
 * <b>Threading model:</b> The simulation runs on a dedicated daemon thread.
 * Python (via Py4J in T3.1) calls {@link #resetSimulation()} and
 * {@link #step(int)} from the Py4J gateway thread.  Two
 * {@link SynchronousQueue}s synchronise the two threads:
 * <pre>
 *   simThread  ──►  stepResultQueue  ──►  pythonThread
 *   simThread  ◄──  actionQueue      ◄──  pythonThread
 * </pre>
 */
public class SimulationManager {

    // ── Configuration ──────────────────────────────────────────────────────

    private final SimulationConfig.DatacenterSpec dcSpec;
    private final String   traceFile;
    private final Scenario scenario;
    private final long     seed;

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

    // ── Energy tracking ────────────────────────────────────────────────────

    private double cumulativeCpuEnergyWs;  // Watt-seconds
    private double cumulativeGpuEnergyWs;
    private double lastEnergyTimestamp;

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

    // ── Constructor ────────────────────────────────────────────────────────

    public SimulationManager(SimulationConfig.DatacenterSpec dcSpec,
                             String traceFile,
                             Scenario scenario,
                             long seed) {
        this.dcSpec    = dcSpec;
        this.traceFile = traceFile;
        this.scenario  = scenario;
        this.seed      = seed;
    }

    /** Convenience constructor with all defaults. */
    public SimulationManager() {
        this(SimulationConfig.DEFAULT_DC,
             SimulationConfig.TRACE_FILE,
             Scenario.HIGH,
             SimulationConfig.DEFAULT_DC.hostSpec().pesCount()); // just reuse seed from env
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

    /** Total energy consumed so far (kWh). */
    public double getTotalEnergyKwh() {
        return (cumulativeCpuEnergyWs + cumulativeGpuEnergyWs) / 3_600_000.0;
    }

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

        // 6. Reset counters
        currentTaskIdx         = 0;
        episodeDone            = false;
        cumulativeCpuEnergyWs  = 0;
        cumulativeGpuEnergyWs  = 0;
        lastEnergyTimestamp    = 0;
        slaViolationCount      = 0;
        hostPeUsage.clear();
        hostRamUsage.clear();
        snapshots.clear();

        System.out.printf("[SimulationManager] Built simulation: %d hosts, %d tasks (%s)%n",
                hosts.size(), tasks.size(), scenario);
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

                // Apply action: allocate current task to chosen host
                TaskRecord task = tasks.get(currentTaskIdx);
                Host target = hosts.get(hostIdx);
                allocateTask(task, target);

                // Advance energy accounting to this task's creation time
                advanceEnergy(task.creationTime());

                // Compute reward
                double[] reward = computeReward(task, target);

                // Move to next task
                currentTaskIdx++;
                episodeDone = (currentTaskIdx >= tasks.size());

                // Record snapshot
                recordSnapshot(task.creationTime());

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

    private void allocateTask(TaskRecord task, Host host) {
        // Manual resource tracking.
        // broker.submitVm() internally calls simulation.schedule() which requires
        // a started CloudSim clock — omitting it prevents silent JVM crashes.
        // Phase 2 will start the simulation clock and restore CloudSim execution.
        hostPeUsage.merge(host, task.pesNeeded(), Integer::sum);
        hostRamUsage.merge(host, (long) task.memoryMib(), Long::sum);

        if (task.numGpu() > 0) {
            GpuState gpuState = gpuRegistry.get(host);
            if (gpuState != null) {
                gpuState.allocate(task.numGpu());
            }
        }
    }

    /** Check if a host can accept a task (CPU PEs + RAM + GPU). */
    private boolean canHost(Host host, TaskRecord task) {
        int  usedPes = hostPeUsage.getOrDefault(host, 0);
        long usedRam = hostRamUsage.getOrDefault(host, 0L);
        int  freePes = dcSpec.hostSpec().pesCount() - usedPes;
        long freeRam = dcSpec.hostSpec().ramMb()    - usedRam;
        GpuState gpu = gpuRegistry.get(host);
        int freeGpus = (gpu != null) ? gpu.available() : 0;

        return freePes >= task.pesNeeded()
            && freeRam >= task.memoryMib()
            && freeGpus >= task.numGpu();
    }

    // ════════════════════════════════════════════════════════════════════════
    //  REWARD COMPUTATION
    // ════════════════════════════════════════════════════════════════════════

    /**
     * Compute the two-component reward vector:
     *   R_energy = −ΔE  (negative energy delta in Watt-seconds)
     *   R_sla    = −λ × max(0, estimated_completion − deadline)
     */
    private double[] computeReward(TaskRecord task, Host host) {
        // ── R_energy: incremental energy from this allocation ──
        // Use manual PE tracking (allocateTask already updated hostPeUsage).
        int usedPes = hostPeUsage.getOrDefault(host, 0);
        double cpuUtil = Math.min(1.0, (double) usedPes / dcSpec.hostSpec().pesCount());
        double cpuPower = dcSpec.powerSpec().cpuIdlePowerWatt()
                + (dcSpec.powerSpec().cpuMaxPowerWatt() - dcSpec.powerSpec().cpuIdlePowerWatt())
                  * cpuUtil;
        GpuState gpu = gpuRegistry.get(host);
        double gpuPower = (gpu != null)
                ? DatacenterFactory.gpuPowerWatt(gpu, dcSpec.powerSpec())
                : 0.0;
        // Energy delta for one scheduling interval
        double deltaEnergy = (cpuPower + gpuPower) * dcSpec.schedulingIntervalSec();
        double rEnergy = -deltaEnergy;

        // ── R_sla: penalty for estimated deadline miss ──
        double estimatedCompletion = task.creationTime() + task.duration();
        double slaSlack = estimatedCompletion - task.deadline();
        double rSla = -task.slaLambda() * Math.max(0.0, slaSlack);
        if (slaSlack > 0) {
            slaViolationCount++;
        }

        return new double[]{rEnergy, rSla};
    }

    // ════════════════════════════════════════════════════════════════════════
    //  ENERGY ACCOUNTING
    // ════════════════════════════════════════════════════════════════════════

    private void advanceEnergy(double toTime) {
        double dt = toTime - lastEnergyTimestamp;
        if (dt <= 0) return;

        for (Host host : hosts) {
            int usedPes = hostPeUsage.getOrDefault(host, 0);
            double cpuUtil = Math.min(1.0, (double) usedPes / dcSpec.hostSpec().pesCount());
            double cpuPower = dcSpec.powerSpec().cpuIdlePowerWatt()
                    + (dcSpec.powerSpec().cpuMaxPowerWatt() - dcSpec.powerSpec().cpuIdlePowerWatt())
                      * cpuUtil;
            cumulativeCpuEnergyWs += cpuPower * dt;

            GpuState gpu = gpuRegistry.get(host);
            if (gpu != null) {
                cumulativeGpuEnergyWs += DatacenterFactory.gpuPowerWatt(gpu, dcSpec.powerSpec()) * dt;
            }
        }
        lastEnergyTimestamp = toTime;
    }

    // ════════════════════════════════════════════════════════════════════════
    //  OBSERVATION BUILDER
    // ════════════════════════════════════════════════════════════════════════

    /**
     * Build a flat observation vector for the RL agent.
     *
     * Layout (H = number of hosts):
     *   [0..H-1]     host CPU utilisation       (0.0–1.0)
     *   [H..2H-1]    host memory utilisation     (0.0–1.0)
     *   [2H..3H-1]   host GPU utilisation        (0.0–1.0)
     *   [3H..3H+3]   current task: [cpu_norm, mem_norm, gpu_norm, qos_lambda]
     *
     * Total length = 3H + 4
     */
    private double[] buildObservation() {
        int h = hosts.size();
        double[] obs = new double[3 * h + 4];

        for (int i = 0; i < h; i++) {
            Host host = hosts.get(i);
            int  usedPes = hostPeUsage.getOrDefault(host, 0);
            long usedRam = hostRamUsage.getOrDefault(host, 0L);
            obs[i]         = Math.min(1.0, (double) usedPes / dcSpec.hostSpec().pesCount());
            obs[h + i]     = Math.min(1.0, (double) usedRam / dcSpec.hostSpec().ramMb());
            GpuState gpu   = gpuRegistry.get(host);
            obs[2 * h + i] = (gpu != null) ? gpu.utilization() : 0.0;
        }

        // Current task features (normalised to [0, 1] based on host capacity)
        if (!episodeDone && currentTaskIdx < tasks.size()) {
            TaskRecord t = tasks.get(currentTaskIdx);
            SimulationConfig.HostSpec hs = dcSpec.hostSpec();
            obs[3 * h]     = (double) t.pesNeeded()  / hs.pesCount();
            obs[3 * h + 1] = (double) t.memoryMib()  / hs.ramMb();
            obs[3 * h + 2] = (double) t.numGpu()     / Math.max(1, hs.gpuCount());
            obs[3 * h + 3] = t.slaLambda() / 3.0;  // normalise: max λ = 3.0
        }

        return obs;
    }

    // ════════════════════════════════════════════════════════════════════════
    //  METRICS RECORDING
    // ════════════════════════════════════════════════════════════════════════

    private void recordSnapshot(double timestamp) {
        double cpuKwh = cumulativeCpuEnergyWs / 3_600_000.0;
        double gpuKwh = cumulativeGpuEnergyWs / 3_600_000.0;

        double avgCpuUtil = hosts.isEmpty() ? 0.0 : hostPeUsage.values().stream()
                .mapToInt(Integer::intValue).average().orElse(0.0)
                / dcSpec.hostSpec().pesCount();
        double avgGpuUtil = MetricsExporter.averageGpuUtilization(gpuRegistry);

        // Average SLA slack across all scheduled tasks so far
        double avgSlack = 0;  // simplified — detailed tracking in Phase 2

        snapshots.add(new MetricsExporter.Snapshot(
                timestamp, cpuKwh, gpuKwh, cpuKwh + gpuKwh,
                currentTaskIdx, slaViolationCount, avgSlack,
                avgCpuUtil, avgGpuUtil));
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
