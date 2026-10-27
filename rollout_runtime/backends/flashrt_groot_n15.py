# Copyright (c) 2026 Zetta Contributors
"""CUDA-Graph accelerator for GR00T N1.5 (Eagle2.5 backbone + DiT action head).

FlashRT (https://github.com/flashrt-project/FlashRT) ships GR00T N1.6/N1.7
frontends only; N1.5 (``GR00T_N1_5``: SigLIP2 + 12-layer Qwen3 + 4-step flow
matching DiT head) has none. This module is the N1.5 counterpart of FlashRT's
approach -- one static-shape CUDA Graph per request shape that replays the
whole ``backbone -> vlln/vl_self_attention -> 4x DiT denoise`` forward -- built
on the stock ``Gr00tPolicy`` so pre/post-processing (Eagle processor,
normalisation, unnormalisation) stays bit-identical to the reference path.

What changes versus ``Gr00tPolicy.get_action``'s model call:

* Attention stays on flash-attn. The vision tower's FA2 path is capturable as
  is, but transformers' FA2 path probes Qwen3's ``position_ids`` for packed
  sequences (a host sync), so while a graph is being captured the Qwen3
  configs point at ``_static_flash_attention`` instead. Everything outside
  ``_capture`` keeps the stock FA2 path, so the eager fallback is the
  reference computation.
* The Eagle image-token scatter (``input_embeds[selected] = ...``, a
  data-dependent boolean index that cannot be captured) becomes a
  cumsum/gather/where, so the prompt tokens may differ between replays.
* The initial noise is a static buffer refilled before each replay, so the
  request-level seed contract of ``groot_core._seed_process`` still holds.
* One graph per ``(num_text_tokens, num_image_tiles)``: instructions of the
  same token length share a graph. At most ``max_graphs`` are kept; a new
  shape evicts the least recently used graph and is captured in its place.
  Batches larger than one (and failed captures) take the eager reference call.

Only ``torch`` and ``transformers`` are imported (lazily), so the module is
importable without GPU dependencies.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

__all__ = ["Gr00tN15GraphRunner", "install_graph_runner"]

_LOG = logging.getLogger(__name__)

_STATIC_FA2 = "flashrt_groot_static_fa2"
"""``AttentionInterface`` key of ``_static_flash_attention``."""


def _static_flash_attention(
    module: Any, query: Any, key: Any, value: Any, attention_mask: Any, **kwargs: Any
) -> Any:
    """transformers' FA2 attention without its host-synchronising probes, for capture only.

    Stock FA2 unpads when given a mask (``.item()``) and checks ``position_ids``
    for packed sequences (``bool(tensor)``); neither can be captured. The
    captured Qwen3 call is unpadded, batch 1, with monotonic positions, so both
    are no-ops there and FA2 takes its plain ``flash_attn_func`` branch. The
    mask passed in is then only the causal mask, which ``module.is_causal``
    already encodes.
    """
    from transformers.integrations.flash_attention import flash_attention_forward

    kwargs.pop("position_ids", None)
    return flash_attention_forward(module, query, key, value, None, **kwargs)


class _CachedProjection:
    """``to_k``/``to_v`` of a DiT cross-attention block, memoised within one ``_forward``.

    The encoder features are identical across the denoise steps, so K/V are
    projected on the first step only. Outside ``_forward`` (``state.active`` false,
    e.g. the eager fallback) it is a plain pass-through.
    """

    def __init__(self, linear: Any, state: Any) -> None:
        self.linear = linear
        self.state = state
        self.value: Any = None

    def __call__(self, x: Any) -> Any:
        if not self.state.active:
            return self.linear(x)
        if self.value is None:
            self.value = self.linear(x)
        return self.value


class _KvCacheState:
    def __init__(self) -> None:
        self.active = False
        self.projections: list[_CachedProjection] = []


class _GraphEntry:
    """Static buffers + captured graph for one ``(tokens, tiles)`` shape."""

    def __init__(self) -> None:
        self.graph: Any = None
        self.input_ids: Any = None
        self.pixel_values: Any = None
        self.state: Any = None
        self.embodiment_id: Any = None
        self.noise: Any = None
        self.out: Any = None


class Gr00tN15GraphRunner:
    """Replays a captured forward of ``GR00T_N1_5`` instead of eager ``get_action``.

    Args:
        model: Loaded ``GR00T_N1_5`` (eval, bf16, on CUDA).
        max_graphs: Maximum number of cached shape-specialised graphs.
    """

    def __init__(self, model: Any, *, max_graphs: int = 16) -> None:
        import torch

        if max_graphs < 1:
            raise ValueError(f"max_graphs must be >= 1, got {max_graphs}")
        self.model = model
        self.max_graphs = max_graphs
        self._entries: OrderedDict[tuple[int, int], _GraphEntry] = OrderedDict()
        self._torch = torch
        self.captures = 0
        self.replays = 0
        self.eager_fallbacks = 0
        self._fa2_configs = self._register_static_fa2()
        eagle = model.backbone.eagle_model
        self._image_token = int(eagle.image_token_index)
        self._stream = torch.cuda.Stream()
        self._pool = torch.cuda.graph_pool_handle()
        self._embodiment_id: Any = None
        self._kv = _KvCacheState()
        self._install_kv_cache()

    def _install_kv_cache(self) -> None:
        head = self.model.action_head
        cross_dim = head.model.config.cross_attention_dim
        for block in head.model.transformer_blocks:
            attn = block.attn1
            if attn.to_k.in_features != cross_dim or block.cross_attention_dim is None:
                continue
            for name in ("to_k", "to_v"):
                wrapper = _CachedProjection(getattr(attn, name), self._kv)
                self._kv.projections.append(wrapper)
                # Bypass nn.Module.__setattr__ so the original Linear stays registered
                # (state_dict/parameters unchanged); attribute lookup hits __dict__ first.
                object.__setattr__(attn, name, wrapper)

    # ------------------------------------------------------------------ setup

    def _register_static_fa2(self) -> list[Any]:
        """Register ``_static_flash_attention``; return the Qwen3 configs that use it during capture."""
        from transformers.modeling_utils import AttentionInterface

        AttentionInterface.register(_STATIC_FA2, _static_flash_attention)
        lm = self.model.backbone.eagle_model.language_model
        configs = {id(m.config): m.config for m in lm.modules() if hasattr(m, "config")}.values()
        return [cfg for cfg in configs if getattr(cfg, "_attn_implementation", None) == "flash_attention_2"]

    @contextlib.contextmanager
    def _static_attention(self) -> Iterator[None]:
        for cfg in self._fa2_configs:
            cfg._attn_implementation = _STATIC_FA2
        try:
            yield
        finally:
            for cfg in self._fa2_configs:
                cfg._attn_implementation = "flash_attention_2"

    # --------------------------------------------------------------- forward

    def _forward(self, input_ids: Any, pixel_values: Any, state: Any, embodiment_id: Any, noise: Any) -> Any:
        """Same math as ``GR00T_N1_5.get_action`` with a static noise input."""
        torch = self._torch
        model = self.model
        eagle = model.backbone.eagle_model
        lm = eagle.language_model
        self._embodiment_id = embodiment_id

        embeds = lm.get_input_embeddings()(input_ids)  # [1, N, C]
        batch, length, width = embeds.shape
        vit = eagle.extract_feature(pixel_values).reshape(-1, width)
        flat_ids = input_ids.reshape(batch * length)
        selected = flat_ids == self._image_token
        index = (selected.cumsum(0) - 1).clamp_(0, vit.shape[0] - 1)
        flat = torch.where(selected[:, None], vit[index].to(embeds.dtype), embeds.reshape(batch * length, width))
        # Eagle runs the (already truncated) Qwen3 without an attention mask.
        hidden = lm.model(inputs_embeds=flat.reshape(batch, length, width)).last_hidden_state
        features = model.backbone.eagle_linear(hidden)

        head = model.action_head
        features = head.vl_self_attention(head.vlln(features))
        state_features = head.state_encoder(state, embodiment_id)
        future_tokens = head.future_tokens.weight.unsqueeze(0).expand(features.shape[0], -1, -1)
        pos_ids = torch.arange(head.action_horizon, dtype=torch.long, device=features.device)
        pos_embs = head.position_embedding(pos_ids).unsqueeze(0) if head.config.add_pos_embed else None

        for projection in self._kv.projections:
            projection.value = None
        self._kv.active = True
        try:
            return self._denoise(head, features, state_features, future_tokens, pos_embs, noise)
        finally:
            self._kv.active = False

    def _denoise(self, head: Any, features: Any, state_features: Any, future_tokens: Any, pos_embs: Any, noise: Any) -> Any:
        torch = self._torch
        embodiment_id = self._embodiment_id
        actions = noise
        steps = head.num_inference_timesteps
        dt = 1.0 / steps
        for step in range(steps):
            bucket = int(step / float(steps) * head.num_timestep_buckets)
            timesteps = torch.full((features.shape[0],), bucket, device=features.device)
            action_features = head.action_encoder(actions, timesteps, embodiment_id)
            if pos_embs is not None:
                action_features = action_features + pos_embs
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
            out = head.model(hidden_states=sa_embs, encoder_hidden_states=features, timestep=timesteps)
            pred = head.action_decoder(out, embodiment_id)
            actions = actions + dt * pred[:, -head.action_horizon :]
        return actions

    # ---------------------------------------------------------------- capture

    def _capture(self, key: tuple[int, int], input_ids: Any, pixel_values: Any, state: Any, embodiment_id: Any) -> _GraphEntry:
        torch = self._torch
        head = self.model.action_head
        entry = _GraphEntry()
        entry.input_ids = input_ids.clone()
        entry.pixel_values = pixel_values.clone()
        entry.state = state.clone()
        entry.embodiment_id = embodiment_id.clone()
        entry.noise = torch.zeros(
            (input_ids.shape[0], head.action_horizon, head.action_dim),
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )

        def run() -> Any:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False), self._static_attention():
                return self._forward(entry.input_ids, entry.pixel_values, entry.state, entry.embodiment_id, entry.noise)

        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(self._stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            entry.out = run()
        entry.graph = graph
        self.captures += 1
        _LOG.info("captured GR00T N1.5 CUDA graph for (tokens=%d, tiles=%d)", *key)
        return entry

    # ----------------------------------------------------------------- public

    def get_action(self, normalized_input: dict[str, Any]) -> Any:
        """Return normalized actions ``[B, horizon, action_dim]`` (float32)."""
        torch = self._torch
        model = self.model
        with torch.inference_mode():
            backbone_inputs, action_inputs = model.prepare_input(normalized_input)
            input_ids = backbone_inputs["eagle_input_ids"]
            pixel_values = backbone_inputs["eagle_pixel_values"]
            state = action_inputs["state"]
            embodiment_id = action_inputs["embodiment_id"]
            if input_ids.shape[0] != 1:
                return self._eager(normalized_input)
            key = (int(input_ids.shape[1]), int(pixel_values.shape[0]))
            entry = self._entries.get(key)
            if entry is None:
                if len(self._entries) >= self.max_graphs:
                    self._entries.popitem(last=False)
                try:
                    entry = self._capture(key, input_ids, pixel_values, state, embodiment_id)
                except Exception:  # noqa: BLE001 - capture failure must not fail the request
                    _LOG.warning("GR00T N1.5 graph capture failed; eager fallback", exc_info=True)
                    return self._eager(normalized_input)
                self._entries[key] = entry
            else:
                self._entries.move_to_end(key)
            entry.input_ids.copy_(input_ids)
            entry.pixel_values.copy_(pixel_values)
            entry.state.copy_(state)
            entry.embodiment_id.copy_(embodiment_id)
            entry.noise.normal_()
            entry.graph.replay()
            self.replays += 1
            return entry.out.float().clone()

    def _eager(self, normalized_input: dict[str, Any]) -> Any:
        self.eager_fallbacks += 1
        torch = self._torch
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return self.model.get_action(normalized_input)["action_pred"].float()


def install_graph_runner(
    policy: Any, *, max_graphs: int = 16
) -> Gr00tN15GraphRunner:
    """Route ``Gr00tPolicy``'s model call through a ``Gr00tN15GraphRunner``.

    Args:
        policy: A loaded ``gr00t.model.policy.Gr00tPolicy``.
        max_graphs: Maximum number of cached shape-specialised graphs.

    Returns:
        The installed runner (also reachable as ``policy.graph_runner``).

    Raises:
        RuntimeError: ``policy.get_action`` never calls
            ``_get_action_from_normalized_input`` (e.g. forks that route through
            ``_get_action_and_diagnostics_from_normalized_input``), so the
            runner would silently never run.
    """
    get_action = inspect.unwrap(type(policy).get_action)
    if "_get_action_from_normalized_input" not in get_action.__code__.co_names:
        raise RuntimeError(
            f"{type(policy).__module__}.{type(policy).__qualname__}.get_action does not call "
            "_get_action_from_normalized_input, so the GR00T N1.5 graph runner would never run; "
            "use the robocasa-benchmark/Isaac-GR00T Gr00tPolicy or the `groot` backend"
        )
    runner = Gr00tN15GraphRunner(policy.model, max_graphs=max_graphs)
    policy._get_action_from_normalized_input = runner.get_action
    policy.graph_runner = runner
    return runner
