package sim;

import sim.SimulationManager.StepResult;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/**
 * SYS.2 — Pack a whole {@link StepResult} (+ action mask) into ONE {@code byte[]}.
 *
 * <p><b>Why this exists (measured, not assumed).</b> Py4J passes Java arrays
 * <i>by reference</i>: every element read from a {@code double[]} costs a
 * separate network round trip. Profiling the real stack
 * ({@code rl-agent/src/perf/profile_step.py}) showed a step at 10 hosts cost
 * ≈21.5 ms, of which the JVM simulation was only ≈0.5 ms — the other ≈21 ms was
 * ~76 per-element round trips at ≈0.28 ms each to drag a 64-element observation
 * and a 10-element mask across the bridge.
 *
 * <p>{@code byte[]} is the one array type Py4J transfers <b>by value</b>, in a
 * single call. Encoding the step payload into one blob therefore collapses
 * ~76 round trips into 1, and the cost stops scaling with the host count —
 * which is what made a 50-host cluster (SYS.3) disproportionately slow.
 *
 * <p>This is a pure <b>transport</b> change: the numbers carried are bit-for-bit
 * the same {@code double}s the reference path returns ({@code ValidationRunner}
 * B18 asserts exact equality against {@link GatewayEntryPoint#step(int)}), so no
 * simulation result, energy figure or SLA cost can shift because of it.
 *
 * <p><b>Wire format</b> (big-endian, matching {@code np.dtype('>f8')}):
 * <pre>
 *   offset  type        field
 *   0       int8        version (= {@link #VERSION})
 *   1       int8        done (0/1)
 *   2       int32       taskIndex
 *   6       int32       numHosts (H)
 *   10      int32       obsLen (= 6H+4)
 *   14      float64[]   observation  (obsLen values)
 *   ...     float64[2]  reward [R_energy, R_sla]
 *   ...     float64     cost (C_SLA for this step, ≥ 0)
 *   ...     uint8[H]    action mask (1 = host feasible)
 *   ...     int32       taskName length in UTF-8 bytes
 *   ...     uint8[]     taskName (UTF-8)
 * </pre>
 * Big-endian is deliberate: it is the JVM's natural order, so the Java side
 * needs no byte swapping and Python reads it with an explicit {@code '>f8'}
 * dtype rather than depending on the host's endianness.
 */
public final class StepCodec {

    /** Wire-format version. Bump on any layout change; Python asserts on it. */
    public static final byte VERSION = 1;

    private StepCodec() {}

    /**
     * Encode a step payload into a single self-describing blob.
     *
     * @param r    the step result to encode
     * @param mask action mask for the resulting state (length H)
     * @return a byte[] Py4J will hand to Python as {@code bytes} in one call
     */
    public static byte[] encode(StepResult r, boolean[] mask) {
        final double[] obs = r.observation();
        final double[] reward = r.reward();
        final String name = r.taskName() == null ? "" : r.taskName();
        final byte[] nameBytes = name.getBytes(StandardCharsets.UTF_8);
        final int h = mask.length;

        final int size = 1 + 1 + 4 + 4 + 4
                + 8 * obs.length
                + 8 * 2
                + 8
                + h
                + 4 + nameBytes.length;

        ByteBuffer buf = ByteBuffer.allocate(size).order(ByteOrder.BIG_ENDIAN);
        buf.put(VERSION);
        buf.put((byte) (r.done() ? 1 : 0));
        buf.putInt(r.taskIndex());
        buf.putInt(h);
        buf.putInt(obs.length);
        for (double v : obs) {
            buf.putDouble(v);
        }
        // reward is always [R_energy, R_sla]; encode defensively so a malformed
        // reward array cannot silently shift every following field.
        buf.putDouble(reward.length > 0 ? reward[0] : 0.0);
        buf.putDouble(reward.length > 1 ? reward[1] : 0.0);
        buf.putDouble(r.cost());
        for (boolean m : mask) {
            buf.put((byte) (m ? 1 : 0));
        }
        buf.putInt(nameBytes.length);
        buf.put(nameBytes);

        return buf.array();
    }
}
