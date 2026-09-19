"""SYS.2 — tests for the packed single-RPC transport (StepCodec ↔ decode_packed).

The blobs here are built with explicit ``struct.pack`` calls rather than by
round-tripping a Python encoder. That is deliberate: the format is a
**cross-language contract** with ``sim/StepCodec.java``, and a Python-only
round-trip would happily agree with itself while disagreeing with the JVM. These
tests pin the actual bytes, so a layout change on either side fails here.

End-to-end parity against the live gateway is proved separately by
``src/perf/verify_packed_parity.py`` (exact equality of every observation,
reward, cost, mask and episode total).
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import state_builder


NUM_HOSTS = 3
OBS_LEN = state_builder.PER_HOST_DIM * NUM_HOSTS + state_builder.TASK_FEATURE_DIM  # 22


def build_blob(
    *,
    version: int = state_builder.PACKED_VERSION,
    done: int = 0,
    task_index: int = 7,
    num_hosts: int = NUM_HOSTS,
    obs: list[float] | None = None,
    reward: tuple[float, float] = (-1.5, -2.5),
    cost: float = 2.5,
    mask: list[int] | None = None,
    task_name: str = "task-42",
    dropped_tasks: int = 0,
) -> bytes:
    """Hand-assemble a blob byte-for-byte per the documented StepCodec layout."""
    obs = [0.5] * OBS_LEN if obs is None else obs
    mask = [1, 0, 1] if mask is None else mask
    name_bytes = task_name.encode("utf-8")

    out = struct.pack(">bbiiii", version, done, task_index, num_hosts, len(obs),
                      dropped_tasks)
    out += struct.pack(f">{len(obs)}d", *obs)
    out += struct.pack(">2d", *reward)
    out += struct.pack(">d", cost)
    out += bytes(mask)
    out += struct.pack(">i", len(name_bytes))
    out += name_bytes
    return out


class TestDecodeHappyPath:
    def test_decodes_every_field(self):
        obs = [i / 100.0 for i in range(OBS_LEN)]
        blob = build_blob(obs=obs, done=1, task_index=13, reward=(-3.25, -7.5),
                          cost=7.5, mask=[0, 1, 1], task_name="pod-xyz",
                          dropped_tasks=4)

        pk = state_builder.decode_packed(blob, NUM_HOSTS)

        np.testing.assert_allclose(pk.observation, np.array(obs, dtype=np.float32))
        np.testing.assert_array_equal(pk.reward, np.array([-3.25, -7.5]))
        assert pk.cost == 7.5
        assert pk.done is True
        assert pk.task_index == 13
        assert pk.task_name == "pod-xyz"
        assert pk.dropped_tasks == 4
        np.testing.assert_array_equal(pk.action_mask, np.array([False, True, True]))

    def test_observation_shape_matches_env_contract(self):
        pk = state_builder.decode_packed(build_blob(), NUM_HOSTS)
        assert pk.observation.shape == (6 * NUM_HOSTS + 4,)
        assert pk.observation.dtype == np.float32

    def test_mask_is_bool_not_uint8(self):
        # MaskablePPO indexes with the mask; a uint8 array would silently act as
        # integer indices rather than a boolean selection.
        pk = state_builder.decode_packed(build_blob(mask=[1, 0, 1]), NUM_HOSTS)
        assert pk.action_mask.dtype == np.bool_

    def test_observation_is_clipped_to_unit_range(self):
        # The observation space is Box(0,1); the legacy path clips, so the packed
        # path must clip identically or the two transports would disagree.
        obs = [5.0] + [-3.0] + [0.5] * (OBS_LEN - 2)
        pk = state_builder.decode_packed(build_blob(obs=obs), NUM_HOSTS)
        assert pk.observation[0] == 1.0
        assert pk.observation[1] == 0.0

    def test_observation_is_writable(self):
        # np.frombuffer views immutable bytes; a read-only observation would
        # break any downstream in-place normalisation.
        pk = state_builder.decode_packed(build_blob(), NUM_HOSTS)
        pk.observation[0] = 0.25  # must not raise
        assert pk.observation[0] == 0.25

    def test_empty_task_name(self):
        pk = state_builder.decode_packed(build_blob(task_name=""), NUM_HOSTS)
        assert pk.task_name == ""

    def test_non_ascii_task_name_round_trips(self):
        pk = state_builder.decode_packed(build_blob(task_name="tác-vụ-λ"), NUM_HOSTS)
        assert pk.task_name == "tác-vụ-λ"

    def test_dropped_tasks_defaults_to_zero(self):
        # The common case: nothing was unplaceable, so the field must read 0
        # rather than picking up whatever bytes follow it.
        assert state_builder.decode_packed(build_blob(), NUM_HOSTS).dropped_tasks == 0

    def test_dropped_tasks_is_big_endian_at_offset_14(self):
        # W3.1 pushed the observation from byte 14 to 18. Pinning the offset here
        # is the Python half of ValidationRunner B18a6: if Java ever writes the
        # field elsewhere, the observation shifts and every value is wrong while
        # still decoding cleanly.
        blob = build_blob(dropped_tasks=258)          # 0x00000102
        assert blob[14:18] == bytes([0, 0, 1, 2])
        assert state_builder.decode_packed(blob, NUM_HOSTS).dropped_tasks == 258

    def test_negative_rewards_and_zero_cost(self):
        # R_energy = −ΔE ≤ 0 and C_SLA ≥ 0 are the physical signs (G1.1).
        pk = state_builder.decode_packed(
            build_blob(reward=(-123.75, 0.0), cost=0.0), NUM_HOSTS)
        assert pk.reward[0] == -123.75
        assert pk.cost == 0.0


class TestDecodeRejectsBadBlobs:
    """A malformed blob must fail loudly — a silent misread corrupts training."""

    def test_version_mismatch_is_rejected(self):
        blob = build_blob(version=state_builder.PACKED_VERSION + 1)
        with pytest.raises(ValueError, match="wire-format mismatch"):
            state_builder.decode_packed(blob, NUM_HOSTS)

    def test_host_count_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="Host count mismatch"):
            state_builder.decode_packed(build_blob(), NUM_HOSTS + 1)

    def test_observation_length_mismatch_is_rejected(self):
        # H agrees but obsLen does not ⇒ the two sides disagree on the layout.
        blob = build_blob(obs=[0.5] * (OBS_LEN - 1))
        with pytest.raises(ValueError, match="Observation length mismatch"):
            state_builder.decode_packed(blob, NUM_HOSTS)

    def test_truncated_header_is_rejected(self):
        with pytest.raises(ValueError, match="truncated"):
            state_builder.decode_packed(b"\x01\x00", NUM_HOSTS)

    def test_truncated_task_name_is_rejected(self):
        blob = build_blob(task_name="abcdef")[:-3]
        with pytest.raises(ValueError, match="truncated"):
            state_builder.decode_packed(blob, NUM_HOSTS)


class TestBigEndianContract:
    """The JVM writes big-endian; decoding must not depend on host byte order."""

    def test_decode_is_explicitly_big_endian(self):
        # 1.0 as big-endian float64 is 3F F0 00 00 00 00 00 00. If the decoder
        # used native ('=f8') order it would read this as a denormal on x86.
        obs = [1.0] + [0.0] * (OBS_LEN - 1)
        blob = build_blob(obs=obs)
        assert blob[18:26] == bytes([0x3F, 0xF0, 0, 0, 0, 0, 0, 0])
        pk = state_builder.decode_packed(blob, NUM_HOSTS)
        assert pk.observation[0] == 1.0

    def test_task_index_is_big_endian(self):
        blob = build_blob(task_index=1)
        assert blob[2:6] == bytes([0, 0, 0, 1])
        assert state_builder.decode_packed(blob, NUM_HOSTS).task_index == 1
