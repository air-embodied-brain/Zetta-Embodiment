"""Payload encoding/decoding and size budget tests.

Assertion focus: inline/ref threshold, 8 MiB ceiling, image encode/decode.
"""

from __future__ import annotations

import numpy as np
import pytest

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError
from rollout_runtime.api.ids import EpisodeId, RequestId, SessionId
from rollout_runtime.api.messages import Observation, StepResult
from rollout_runtime.api.payload_ref import (
    InlineBytes,
    ObjectRefId,
    PayloadCodec,
    payload_nbytes,
)
from rollout_runtime.core import obs_schema
from rollout_runtime.core import payload as payload_module

SESSION = SessionId("sess-payload")


@pytest.fixture(autouse=True)
def _reset_stats() -> None:
    """Reset the payload counters before each test case."""
    payload_module.stats().reset()


def _image(height: int = 12, width: int = 9, channels: int = 3) -> np.ndarray:
    rng = np.random.default_rng(20260806)
    return rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)


# ------------------------------------------------------------------ Image codec


def test_png_round_trip_is_exact() -> None:
    """PNG is lossless: uint8 HWC must match pixel-for-pixel."""
    image = _image()
    ref = payload_module.encode_image(image)
    assert isinstance(ref, InlineBytes)
    assert ref.codec is PayloadCodec.PNG
    assert ref.shape == image.shape
    assert ref.dtype == "uint8"
    assert np.array_equal(payload_module.decode_payload(ref), image)


def test_png_bytes_are_deterministic() -> None:
    """Encoding the same array twice must produce byte-identical output.

    Legacy parity checks compare images by hash step by step, so encoding must
    not depend on version-specific behavior of third-party libraries.
    """
    image = _image()
    assert (
        payload_module.encode_image(image).data
        == payload_module.encode_image(image).data
    )


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_png_supports_common_channel_counts(channels: int) -> None:
    """Grayscale / grayscale+alpha / RGB / RGBA must all be supported.

    Args:
        channels: Channel count.
    """
    image = _image(channels=channels)
    ref = payload_module.encode_image(image)
    assert np.array_equal(payload_module.decode_payload(ref), image)


def test_png_accepts_2d_and_normalizes_to_hwc() -> None:
    """A 2D grayscale image gets padded to ``[H, W, 1]``."""
    grey = _image(channels=1)[:, :, 0]
    ref = payload_module.encode_image(grey)
    assert ref.shape == (*grey.shape, 1)
    assert np.array_equal(payload_module.decode_payload(ref)[:, :, 0], grey)


def test_png_rejects_non_uint8() -> None:
    """PNG only accepts uint8."""
    with pytest.raises(RuntimeApiError) as excinfo:
        payload_module.encode_image(np.zeros((4, 4, 3), dtype=np.float32))
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT


def test_png_rejects_unsupported_channel_count() -> None:
    """5 channels is not a valid PNG color type."""
    with pytest.raises(RuntimeApiError):
        payload_module.encode_image(np.zeros((4, 4, 5), dtype=np.uint8))


def test_png_decode_rejects_non_png_bytes() -> None:
    """Non-PNG byte streams must raise an error instead of crashing inside zlib."""
    bogus = InlineBytes(
        codec=PayloadCodec.PNG, shape=(1, 1, 3), dtype="uint8", data=b"not-a-png"
    )
    with pytest.raises(RuntimeApiError):
        payload_module.decode_image(bogus)


def test_png_decode_detects_shape_mismatch() -> None:
    """A declared shape that doesn't match the PNG header must raise an error."""
    ref = payload_module.encode_image(_image())
    tampered = InlineBytes(
        codec=ref.codec, shape=(1, 1, 3), dtype=ref.dtype, data=ref.data
    )
    with pytest.raises(RuntimeApiError):
        payload_module.decode_image(tampered)


def test_png_decoder_handles_all_filter_types() -> None:
    """External encoders (Pillow) may use non-zero filters; decoding must handle them."""
    pillow = pytest.importorskip("PIL.Image")
    import io

    image = _image(height=17, width=13)
    buffer = io.BytesIO()
    pillow.fromarray(image).save(buffer, format="PNG")
    ref = InlineBytes(
        codec=PayloadCodec.PNG,
        shape=image.shape,
        dtype="uint8",
        data=buffer.getvalue(),
    )
    assert np.array_equal(payload_module.decode_image(ref), image)


# ------------------------------------------------------------------ Array codec


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "uint8", "bool"])
def test_raw_round_trip_preserves_dtype_and_shape(dtype: str) -> None:
    """Raw encoding preserves dtype and shape.

    Args:
        dtype: numpy dtype name.
    """
    array = (np.arange(24).reshape(2, 3, 4) % 2).astype(dtype)
    ref = payload_module.encode_array(array)
    assert ref.codec is PayloadCodec.RAW
    assert ref.dtype == dtype
    restored = payload_module.decode_payload(ref)
    assert restored.dtype == np.dtype(dtype)
    assert np.array_equal(restored, array)


def test_raw_handles_non_contiguous_input() -> None:
    """A non-contiguous view must be materialized first, so encoding doesn't produce misaligned bytes."""
    array = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
    assert not array.flags["C_CONTIGUOUS"]
    restored = payload_module.decode_payload(payload_module.encode_array(array))
    assert np.array_equal(restored, array)


def test_raw_decode_detects_size_mismatch() -> None:
    """A byte count that doesn't match shape/dtype must raise an error."""
    ref = payload_module.encode_array(np.zeros((3, 4), dtype=np.float32))
    tampered = InlineBytes(
        codec=ref.codec, shape=(3, 5), dtype=ref.dtype, data=ref.data
    )
    with pytest.raises(RuntimeApiError) as excinfo:
        payload_module.decode_array(tampered)
    assert excinfo.value.info.code is ErrorCode.INVALID_ARGUMENT


def test_decode_rejects_codec_mismatch() -> None:
    """Using the wrong decoder must raise an explicit error."""
    png = payload_module.encode_image(_image())
    with pytest.raises(RuntimeApiError):
        payload_module.decode_array(png)
    raw = payload_module.encode_array(np.zeros(4, dtype=np.float32))
    with pytest.raises(RuntimeApiError):
        payload_module.decode_image(raw)


def test_encode_payload_picks_png_for_images_and_raw_otherwise() -> None:
    """Automatic selection: uint8 HWC -> PNG; float32 action chunk -> raw."""
    assert payload_module.encode_payload(_image()).codec is PayloadCodec.PNG
    actions = np.zeros((4, 7), dtype=np.float32)
    assert payload_module.encode_payload(actions).codec is PayloadCodec.RAW
    wide = np.zeros((4, 4, 9), dtype=np.uint8)
    assert payload_module.encode_payload(wide).codec is PayloadCodec.RAW


def test_object_ref_is_not_produced_in_v1() -> None:
    """v1 always inlines; decoding an ``ObjectRefId`` must be explicitly rejected."""
    ref = ObjectRefId(id="obj-1", shape=(2, 2), dtype="uint8")
    assert payload_nbytes(ref) == 0
    with pytest.raises(RuntimeApiError, match="object-store"):
        payload_module.decode_payload(ref)


# ------------------------------------------------------------------ Threshold and budget


def test_inline_threshold_only_counts_oversize_but_still_inlines() -> None:
    """A payload over 256 KiB still inlines, but must count toward the
    oversize warning."""
    small = payload_module.encode_array(np.zeros(1024, dtype=np.uint8))
    assert payload_nbytes(small) < payload_module.INLINE_THRESHOLD_BYTES
    assert payload_module.stats().oversize_count == 0

    big = payload_module.encode_array(
        np.zeros(payload_module.INLINE_THRESHOLD_BYTES + 1, dtype=np.uint8)
    )
    assert isinstance(big, InlineBytes)
    assert payload_nbytes(big) > payload_module.INLINE_THRESHOLD_BYTES
    stats = payload_module.stats()
    assert stats.oversize_count == 1
    assert stats.oversize_bytes > payload_module.INLINE_THRESHOLD_BYTES
    assert stats.encoded_count == 2


def test_payload_budget_accepts_within_limit() -> None:
    """Returns the tallied byte count when within budget."""
    refs = [
        payload_module.encode_array(np.zeros(1000, dtype=np.uint8)) for _ in range(3)
    ]
    assert payload_module.check_payload_budget(refs) == 3000


def test_payload_budget_rejects_oversized_request() -> None:
    """A single request exceeding 8 MiB -> ``INVALID_ARGUMENT``, to avoid
    overwhelming the Channel."""
    limit = payload_module.REQUEST_PAYLOAD_LIMIT_BYTES
    ref = payload_module.encode_array(np.zeros(limit + 1, dtype=np.uint8))
    with pytest.raises(RuntimeApiError) as excinfo:
        payload_module.check_payload_budget(ref)
    info = excinfo.value.info
    assert info.code is ErrorCode.INVALID_ARGUMENT
    assert info.detail["payload_limit"] == limit
    assert info.detail["payload_bytes"] == limit + 1
    assert payload_module.REQUEST_PAYLOAD_LIMIT_BYTES == 8 * 1024 * 1024


def test_payload_budget_walks_nested_messages() -> None:
    """The budget tally must traverse nested dataclass / list / dict structures."""
    observation = Observation(
        session_id=SESSION,
        episode_id=EpisodeId(1),
        step_index=0,
        main_image=payload_module.encode_image(_image(64, 64)),
        extra_view_images=[payload_module.encode_image(_image(32, 32))],
        state=[0.0] * 8,
    )
    step = StepResult(
        request_id=RequestId("req-1"), session_id=SESSION, observation=observation
    )
    total = payload_module.check_payload_budget(step)
    expected = payload_nbytes(observation.main_image) + payload_nbytes(
        observation.extra_view_images[0]
    )
    assert total == expected
    with pytest.raises(RuntimeApiError):
        payload_module.check_payload_budget(step, limit=expected - 1)


def test_stats_track_encode_and_decode_volume() -> None:
    """The counters must cover both encode and decode bytes (used for A/B
    comparison against legacy HTTP transfer volume)."""
    ref = payload_module.encode_array(np.zeros(512, dtype=np.uint8))
    payload_module.decode_payload(ref)
    stats = payload_module.stats()
    assert stats.encoded_count == 1
    assert stats.encoded_bytes == 512
    assert stats.decoded_count == 1
    assert stats.decoded_bytes == 512


# ------------------------------------------------------------------ obs schema


def _observation(state_dim: int = 4, wrist: bool = True, seed: int = 1) -> Observation:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    return Observation(
        session_id=SESSION,
        episode_id=EpisodeId(1),
        step_index=0,
        main_image=payload_module.encode_image(pixels),
        wrist_image=payload_module.encode_image(pixels) if wrist else None,
        state=[float(index) for index in range(state_dim)],
        instruction="pick",
    )


def test_obs_schema_digest_ignores_pixel_values() -> None:
    """The schema digest only looks at structure, not values -- otherwise
    every step would fall into a new bucket."""
    first = _observation(seed=1)
    second = _observation(seed=2)
    assert first.main_image != second.main_image
    assert obs_schema.obs_schema_digest(first) == obs_schema.obs_schema_digest(second)


def test_obs_schema_digest_reacts_to_structure_changes() -> None:
    """Field presence and dimensionality changes must change the digest
    (``_merge_obs_batches`` requires matching structure)."""
    base = obs_schema.obs_schema_digest(_observation())
    assert base != obs_schema.obs_schema_digest(_observation(state_dim=7))
    assert base != obs_schema.obs_schema_digest(_observation(wrist=False))


def test_env_output_keys_match_rlinf_schema() -> None:
    """The 5-key schema must align with rlinf's ``EnvOutput.prepare_observations``."""
    assert obs_schema.ENV_OUTPUT_KEYS == (
        "main_images",
        "wrist_images",
        "extra_view_images",
        "states",
        "task_descriptions",
    )
    assert set(obs_schema.OBS_FIELD_TO_ENV_OUTPUT_KEY.values()) == set(
        obs_schema.ENV_OUTPUT_KEYS
    )


def test_observations_to_env_output_stacks_batch() -> None:
    """Batch conversion produces the 5-key batch dict."""
    batch = obs_schema.observations_to_env_output([_observation(), _observation()])
    assert set(batch) == set(obs_schema.ENV_OUTPUT_KEYS)
    assert batch["main_images"].shape == (2, 8, 8, 3)
    assert batch["states"].shape == (2, 4)
    assert batch["states"].dtype == np.float32
    assert batch["task_descriptions"] == ["pick", "pick"]
    assert batch["extra_view_images"] is None


def test_observations_to_env_output_rejects_mixed_schema() -> None:
    """A batch with mismatched structure must be rejected, not silently misaligned."""
    with pytest.raises(ValueError, match="schema mismatch"):
        obs_schema.observations_to_env_output(
            [_observation(), _observation(state_dim=9)]
        )
    with pytest.raises(ValueError, match="must not be empty"):
        obs_schema.observations_to_env_output([])
