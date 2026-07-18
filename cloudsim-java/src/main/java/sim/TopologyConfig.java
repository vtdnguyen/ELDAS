package sim;

import com.google.gson.Gson;

import sim.SimulationConfig.DatacenterSpec;
import sim.SimulationConfig.HostSku;
import sim.SimulationConfig.HostSpec;
import sim.SimulationConfig.PowerSpec;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * G2.1 — Loads a heterogeneous datacenter topology from a JSON file
 * (e.g. {@code config/topology-hetero.json}) into a
 * {@link SimulationConfig.DatacenterSpec} with a per-SKU host list.
 *
 * <h3>Schema</h3>
 * <pre>
 * {
 *   "name": "hetero-3sku",
 *   "schedulingIntervalSec": 1.0,
 *   "skus": [
 *     { "name": "gpu-heavy", "count": 3,
 *       "vcpu": 64, "mips": 10000, "ramGb": 256, "gpu": 4, "gpuMemoryMb": 32768,
 *       "cpuMaxWatt": 500, "cpuIdleWatt": 200, "gpuMaxWatt": 300, "gpuIdleWatt": 30,
 *       "idleThresholdSec": 30, "suspendedPowerWatt": 10,
 *       "wakeEnergyKwh": 0.0005, "wakeLatencySec": 5 },
 *     ...
 *   ]
 * }
 * </pre>
 *
 * <p>Every numeric field is optional: a missing field inherits the value from
 * the supplied homogeneous {@code fallback} spec (which itself comes from the
 * env-var defaults). Only {@code name} and {@code count} truly matter per SKU.
 * This keeps the JSON terse — a CPU-only SKU only needs to override
 * {@code gpu:0} and its power figures.
 *
 * <p><b>Design:</b> Gson deserialises into boxed-primitive DTOs so we can
 * distinguish "absent" ({@code null}) from "explicitly zero". The DTOs are
 * then folded into immutable {@link HostSku} records applying the fallbacks.
 */
public final class TopologyConfig {

    private TopologyConfig() {} // utility class

    // ── Gson DTOs (mirror the JSON exactly; boxed to detect absence) ──────

    /** Top-level JSON object. */
    private static final class TopoDto {
        String       name;
        Double       schedulingIntervalSec;
        List<SkuDto> skus;
    }

    /** One SKU entry in the {@code skus} array. */
    private static final class SkuDto {
        String  name;
        Integer count;
        // Host hardware
        Integer vcpu;
        Long    mips;
        Integer ramGb;
        Integer gpu;
        Long    gpuMemoryMb;
        // Power model
        Double  cpuMaxWatt;
        Double  cpuIdleWatt;
        Double  gpuMaxWatt;
        Double  gpuIdleWatt;
        Double  idleThresholdSec;
        Double  suspendedPowerWatt;
        Double  wakeEnergyKwh;
        Double  wakeLatencySec;
    }

    // ── Public API ────────────────────────────────────────────────────────

    /**
     * Parse {@code jsonPath} into a heterogeneous {@link DatacenterSpec}.
     *
     * @param jsonPath path to the topology JSON
     * @param fallback homogeneous spec supplying defaults for absent fields
     *                 and the reference host/power specs
     * @return heterogeneous spec (hostCount = Σ SKU counts, skus populated)
     * @throws IOException              if the file cannot be read
     * @throws IllegalArgumentException if the JSON has no usable SKU
     */
    public static DatacenterSpec load(String jsonPath, DatacenterSpec fallback)
            throws IOException {
        Path path = Path.of(jsonPath);
        TopoDto dto;
        try (Reader r = Files.newBufferedReader(path)) {
            dto = new Gson().fromJson(r, TopoDto.class);
        }
        if (dto == null || dto.skus == null || dto.skus.isEmpty()) {
            throw new IllegalArgumentException(
                    "topology JSON has no 'skus' array: " + jsonPath);
        }

        HostSpec  dh = fallback.hostSpec();
        PowerSpec dp = fallback.powerSpec();

        List<HostSku> skus = new ArrayList<>(dto.skus.size());
        int totalHosts = 0;

        for (int i = 0; i < dto.skus.size(); i++) {
            SkuDto s = dto.skus.get(i);
            int count = orDefault(s.count, 0);
            if (count <= 0) {
                throw new IllegalArgumentException(
                        "SKU " + i + " ('" + s.name + "') has count <= 0");
            }
            String skuName = (s.name != null && !s.name.isBlank())
                    ? s.name.trim() : ("sku-" + i);

            HostSpec hs = new HostSpec(
                    Math.max(1, orDefault(s.vcpu, dh.pesCount())),
                    Math.max(100L, orDefault(s.mips, dh.mips())),
                    ((long) Math.max(1, orDefault(s.ramGb, (int) (dh.ramMb() / 1024)))) * 1024L,
                    dh.bwMbps(),
                    dh.storageMb(),
                    Math.max(0, orDefault(s.gpu, dh.gpuCount())),
                    orDefault(s.gpuMemoryMb, dh.gpuMemoryMb()));

            PowerSpec ps = new PowerSpec(
                    orDefault(s.cpuMaxWatt,         dp.cpuMaxPowerWatt()),
                    orDefault(s.cpuIdleWatt,        dp.cpuIdlePowerWatt()),
                    orDefault(s.gpuMaxWatt,         dp.gpuMaxPowerWatt()),
                    orDefault(s.gpuIdleWatt,        dp.gpuIdlePowerWatt()),
                    orDefault(s.idleThresholdSec,   dp.idleThresholdSec()),
                    orDefault(s.suspendedPowerWatt, dp.suspendedPowerWatt()),
                    orDefault(s.wakeEnergyKwh,      dp.wakeEnergyKwh()),
                    orDefault(s.wakeLatencySec,     dp.wakeLatencySec()));

            skus.add(new HostSku(skuName, count, hs, ps));
            totalHosts += count;
        }

        double interval = orDefault(dto.schedulingIntervalSec,
                fallback.schedulingIntervalSec());

        // Reference host/power = first SKU (used only for top-level printouts
        // and task-feature normalisers; per-host specs live in `skus`).
        HostSpec  refHost  = skus.get(0).hostSpec();
        PowerSpec refPower = skus.get(0).powerSpec();

        return new DatacenterSpec(totalHosts, refHost, refPower, interval, skus);
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    private static int    orDefault(Integer v, int dflt)    { return v == null ? dflt : v; }
    private static long   orDefault(Long v, long dflt)       { return v == null ? dflt : v; }
    private static double orDefault(Double v, double dflt)   { return v == null ? dflt : v; }
}
