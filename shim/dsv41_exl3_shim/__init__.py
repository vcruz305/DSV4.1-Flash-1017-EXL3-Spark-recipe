"""Minimal vLLM shim: let DeepSeek-V4.1 accept EXL3 routed experts.

Why this exists
---------------
The spark-ds41f r6w engine builds ``DeepseekV4Model`` with a hard gate

    if quant_config is None or quant_config.get_name() not in (
        "deepseek_v41_fp8", "mxfp4_csf",
    ):
        raise ValueError("V4.1 requires its native block32 quantization config")

and everything else in the tree is keyed off that same name (``weight_block_size``
decides the block-FP8 shared-expert padding, ``is_scale_e8m0`` selects the E8M0
scale layout).  vllm-exl3's ``Exl3Config`` is a different quant method
(``"exl3"``) that owns the routed experts but delegates every dense/attention
weight to a declared source-format config.

So: keep one config object, give it the accepted *name*, and let it return
``Exl3MoEMethod`` for ``RoutedExperts`` (main stack and the ``mtp.*`` DSpark
experts) while every ``LinearBase``/embedding keeps the engine's own native
block-32 FP8 kernels through the ``non_routed_quantization`` delegate.

The class is registered under its own method name, ``exl3_dsv41``, so the
Kimi-K3 ``model_config.quantization != "exl3"`` hook in
``vllm/model_executor/models/config.py`` never fires and vLLM's own native
``exl3`` (B12X Kimi container) stays untouched.

No vLLM source file is modified by this shim.

Memory probe (``DSV41_MEM_PROBE=1``)
-----------------------------------
Phase 2 accounting hook.  Wraps ``DefaultModelLoader.load_model`` and prints a
one-line-per-stage memory waterfall (host MemAvailable / AnonPages / Cached /
VmRSS, torch cuda allocated+reserved) plus, after the load returns, the
parameter+buffer bytes grouped by module path with the layer index collapsed to
``*`` and the largest single tensors.  It only reads state; it never allocates.

DSML tool-call stop (``DSV41_DSML_STOP=1``)
-------------------------------------------
(In the DSML tags in these docstrings, ``|`` stands for U+FF5C, the fullwidth bar.)
Adds the stop string ``</|DSML| invoke>\n</`` (kept in the output) to chat
requests that carry tools, so a reply ends at the first closing tag after a
closed invoke.  See ``_install_dsml_stop``.
"""

from __future__ import annotations

import os

__all__ = ["register"]


def _proc_meminfo() -> dict[str, float]:
    want = {"MemAvailable", "AnonPages", "Cached", "MemFree", "SwapFree", "SwapTotal"}
    out: dict[str, float] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                name, _, rest = line.partition(":")
                if name in want:
                    out[name] = int(rest.strip().split()[0]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return out


def _mem_line(tag: str) -> str:
    mi = _proc_meminfo()
    parts = [f"MEMPROBE {tag}"]
    for k in ("MemAvailable", "AnonPages", "Cached", "MemFree"):
        if k in mi:
            parts.append(f"{k}={mi[k]:.2f}GiB")
    if "SwapTotal" in mi and "SwapFree" in mi:
        parts.append(f"swap_used={(mi['SwapTotal'] - mi['SwapFree']) * 1024:.0f}MiB")
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    parts.append(f"VmRSS={int(line.split()[1]) / 1048576.0:.2f}GiB")
                    break
    except OSError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            parts.append(f"cuda_alloc={torch.cuda.memory_allocated() / 2**30:.2f}GiB")
            parts.append(f"cuda_reserved={torch.cuda.memory_reserved() / 2**30:.2f}GiB")
    except Exception:
        pass
    return " ".join(parts)


def _module_breakdown(model) -> list[str]:
    import re

    by_mod: dict[str, int] = {}
    by_tensor: list[tuple[int, str]] = []
    seen: set[int] = set()
    total = 0

    def _visit(name: str, t) -> None:
        nonlocal total
        try:
            nbytes = int(t.numel()) * int(t.element_size())
            ptr = t.untyped_storage().data_ptr()
        except Exception:
            return
        if ptr in seen:
            return
        seen.add(ptr)
        total += nbytes
        by_tensor.append((nbytes, name))
        mod = name.rsplit(".", 1)[0] if "." in name else name
        mod = re.sub(r"\.\d+\.", ".*.", "." + mod + ".")[1:-1]
        mod = re.sub(r"^(\w+)\.\*\.", r"\1.*.", mod)
        by_mod[mod] = by_mod.get(mod, 0) + nbytes

    try:
        for n, p in model.named_parameters():
            _visit(n, p.data if hasattr(p, "data") else p)
        for n, b in model.named_buffers():
            _visit(n, b)
    except Exception as exc:  # pragma: no cover - diagnostic only
        return [f"MEMPROBE breakdown failed: {exc!r}"]

    out = [f"MEMPROBE TENSOR_TOTAL unique_bytes={total / 2**30:.3f}GiB tensors={len(seen)}"]
    for mod, nb in sorted(by_mod.items(), key=lambda kv: -kv[1])[:18]:
        out.append(f"MEMPROBE   module {nb / 2**30:8.3f}GiB  {mod}")
    for nb, name in sorted(by_tensor, reverse=True)[:8]:
        out.append(f"MEMPROBE   tensor {nb / 2**30:8.3f}GiB  {name}")
    return out


def _cuda_snapshot_top(n: int = 12) -> list[str]:
    """Group the caching allocator by allocating Python frame.

    ``MEMPROBE POST_LOAD`` gives the non-arena CUDA total but not its owner:
    the gap between the shard-derived weight bytes and ``cuda_alloc`` is
    anonymous, i.e. not reachable from ``named_parameters()``.  This walks the
    allocator's segment map (only *allocated* blocks) and prints the largest
    aggregate stacks, which names the residual without guessing.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        snap = torch.cuda.memory_snapshot()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return [f"MEMPROBE SNAPSHOT failed: {exc!r}"]
    agg: dict[str, int] = {}
    for seg in snap:
        blocks = seg.get("blocks") or []
        for b in blocks:
            if b.get("state") != "active_allocated":
                continue
            nbytes = int(b.get("size", 0))
            frames = b.get("frames") or []
            label = "?"
            for f in frames:
                fn = f.get("filename") or ""
                if "/torch/" in fn or not fn:
                    continue
                label = f"{fn.rsplit('/', 1)[-1]}:{f.get('line')} {f.get('name')}"
                break
            agg[label] = agg.get(label, 0) + nbytes
    out = [
        f"MEMPROBE SNAPSHOT allocated={sum(agg.values()) / 2**30:.3f}GiB "
        f"stacks={len(agg)}"
    ]
    for label, nb in sorted(agg.items(), key=lambda kv: -kv[1])[:n]:
        out.append(f"MEMPROBE   alloc {nb / 2**20:9.1f}MiB  {label}")
    return out



def _install_mem_probe() -> None:
    if os.environ.get("DSV41_MEM_PROBE", "0") != "1":
        return
    try:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    except Exception as exc:  # pragma: no cover
        print(f"MEMPROBE install failed: {exc!r}", flush=True)
        return
    if getattr(DefaultModelLoader.load_model, "_dsv41_probe", False):
        return
    orig = DefaultModelLoader.load_model

    def load_model(self, *args, **kwargs):
        import gc

        print(_mem_line("PRE_LOAD"), flush=True)
        model = orig(self, *args, **kwargs)
        gc.collect()
        print(_mem_line("POST_LOAD"), flush=True)
        for line in _module_breakdown(model):
            print(line, flush=True)
        for line in _cuda_snapshot_top():
            print(line, flush=True)
        return model

    load_model._dsv41_probe = True
    DefaultModelLoader.load_model = load_model


def _install_ats_alias() -> None:
    """Alias ``embed.weight`` (and optionally prefetch expert aliases) after load.

    Runs inside ``DefaultModelLoader.load_model`` so the CUDA copy of the
    embedding is released *before* the engine sizes the KV cache.
    """
    if os.environ.get("VLLM_EXL3_ATS_ALIAS", "0") in ("0", "", "false"):
        return
    try:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    except Exception as exc:  # pragma: no cover
        print(f"ATS alias install failed: {exc!r}", flush=True)
        return
    if getattr(DefaultModelLoader.load_model, "_dsv41_ats", False):
        return
    orig = DefaultModelLoader.load_model

    def load_model(self, *args, **kwargs):
        model = orig(self, *args, **kwargs)
        try:
            from vllm_exl3 import ats_alias

            n = ats_alias.alias_embedding(model)
            ats_alias.prefetch()
            ats_alias.log_stats()
            print(
                f"ATS ALIAS post-load: embed_modules={n} " + _mem_line("POST_ATS"),
                flush=True,
            )
        except Exception as exc:  # never fail the boot over the optional path
            print(f"ATS ALIAS failed: {exc!r}", flush=True)
        return model

    load_model._dsv41_ats = True
    DefaultModelLoader.load_model = load_model


def _install_rope_trim() -> None:
    """Truncate the shared RoPE cos/sin caches to ``max_model_len``.

    ``vllm/model_executor/layers/rotary_embedding/base.py:94`` sizes the cache
    as ``[max_position_embeddings, rotary_dim]`` in the *default* torch dtype
    (fp32 here -- ``get_rope`` is called without ``dtype``, so
    ``rotary_embedding/__init__.py:40`` picks ``torch.get_default_dtype()``).
    This checkpoint declares ``max_position_embeddings = 1048576`` and
    ``qk_rope_head_dim = 64``, i.e. 256 MiB per distinct cache.  ``get_rope``
    memoizes on ``(head_size, rotary_dim, max_position, ..., dtype)``
    (``rotary_embedding/__init__.py:73``) and DSV4.1 builds two distinct
    parameter sets (compress_ratio > 0 uses ``compress_rope_theta`` and
    ``deepseek_yarn``, ``models/deepseek_v41/common/rope.py:20-47``), so the
    process holds exactly two of them: **512 MiB** that the engine never
    indexes past ``max_model_len``.

    Every consumer is ``cos_sin_cache[positions]`` / a raw pointer plus
    ``position * rotary_dim`` (``base.py:267``, ``attention.py:882``,
    ``attention.py:1220``, ``compressor.py:318``), with ``positions <
    max_model_len``, so the tail rows are dead weight.

    ``VLLM_DS41_ROPE_TRIM=1`` enables it; ``VLLM_DS41_ROPE_TRIM_MAX_POS``
    overrides the row count (default ``max_model_len``).
    """

    if os.environ.get("VLLM_DS41_ROPE_TRIM", "0") in ("0", "", "false"):
        return
    try:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    except Exception:  # pragma: no cover
        return
    if getattr(DefaultModelLoader.load_model, "_dsv41_rope_trim", False):
        return
    orig = DefaultModelLoader.load_model

    def load_model(self, *args, **kwargs):
        model = orig(self, *args, **kwargs)
        try:
            import torch
            from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

            want = os.environ.get("VLLM_DS41_ROPE_TRIM_MAX_POS", "").strip()
            if want:
                n = int(want)
            else:
                # ``DefaultModelLoader`` has no ``vllm_config`` attribute; the
                # active config is a module-level context var in V1.  Fall back
                # to MAX_MODEL_LEN_FALLBACK so a trim never depends on it.
                n = 0
                try:
                    from vllm.config import get_current_vllm_config

                    n = int(get_current_vllm_config().model_config.max_model_len)
                except Exception:
                    n = 0
                if n <= 0:
                    n = int(os.environ.get("MAXLEN", "32768") or 32768)
            if n < 4096:
                print(f"ROPE TRIM: refusing n={n} (<4096)", flush=True)
                return model
            seen = set()
            freed = 0
            for mod in list(_ROPE_DICT.values()):
                if id(mod) in seen:
                    continue
                seen.add(id(mod))
                c = getattr(mod, "cos_sin_cache", None)
                if not torch.is_tensor(c) or c.dim() != 2 or c.shape[0] <= n:
                    continue
                old_bytes = c.numel() * c.element_size()
                mod.cos_sin_cache = c[:n].contiguous()
                del c
                freed += old_bytes - mod.cos_sin_cache.numel() * mod.cos_sin_cache.element_size()
            if freed:
                print(
                    f"ROPE TRIM: {len(seen)} cache(s) -> {n} rows, freed {freed / 2**30:.3f} GiB",
                    flush=True,
                )
        except Exception as exc:  # never fail the boot over the optional path
            print(f"ROPE TRIM failed: {exc!r}", flush=True)
        return model

    load_model._dsv41_rope_trim = True
    DefaultModelLoader.load_model = load_model


def _install_sampling_caps() -> None:
    """Reject over-size sampling lists in the API server with a 400.

    DSpark runs on Model Runner V2, whose sampler pre-sizes per-request buffers
    (``v1/worker/gpu/sample/logit_bias.py``, ``bad_words.py``) and raises
    ``ValueError`` inside EngineCore when a request exceeds them. EngineCore
    treats that as fatal, so one request with 1,025 ``allowed_token_ids`` takes
    the whole server down. The frontend only range-checks these lists
    (upstream: vllm-project/vllm#55164). Check every limit the sampler enforces
    here, after ``update_from_tokenizer`` has filled in the final stop and
    bad-word ids: allowed token ids, logit-bias entries, stop ids under
    ``min_tokens``, number of bad words and their total tokens. The limits are
    read from the sampler module so they cannot drift from what it enforces.
    """
    from vllm import envs
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import SamplingParams

    orig = SamplingParams.update_from_tokenizer
    if getattr(orig, "_dsv41_caps", False):
        return
    try:
        from vllm.v1.worker.gpu.sample import logit_bias as _lb

        max_allowed = int(_lb.MAX_NUM_ALLOWED_TOKEN_IDS)
        max_bias = int(_lb.MAX_NUM_LOGIT_BIAS_TOKENS)
        max_stop = int(_lb.MAX_NUM_STOP_TOKEN_IDS)
    except Exception:  # sampler module moved: fall back to the values it had
        max_allowed, max_bias, max_stop = 1024, 1024, 128

    def _reject(msg, parameter):
        raise VLLMValidationError(msg, parameter=parameter)

    def update_from_tokenizer(self, tokenizer):
        orig(self, tokenizer)
        n = len(self.allowed_token_ids or ())
        if n > max_allowed:
            _reject(f"allowed_token_ids has {n} ids; the max is {max_allowed}", "allowed_token_ids")
        n = len(self.logit_bias or ())
        if n > max_bias:
            _reject(f"logit_bias has {n} entries; the max is {max_bias}", "logit_bias")
        n = len(self.all_stop_token_ids or ())
        if (self.min_tokens or 0) > 0 and n > max_stop:
            _reject(f"min_tokens needs at most {max_stop} stop ids (incl. EOS); got {n}", "stop_token_ids")
        words = self.bad_words_token_ids or ()
        max_words = int(getattr(envs, "VLLM_MAX_NUM_BAD_WORDS", 128))
        if len(words) > max_words:
            _reject(f"bad_words has {len(words)} entries; the max is {max_words}", "bad_words")
        n = sum(len(w) for w in words)
        max_tok = int(envs.VLLM_MAX_BAD_WORDS_TOTAL_TOKENS)
        if n > max_tok:
            _reject(f"bad_words tokenize to {n} tokens; the max is {max_tok}", "bad_words")

    update_from_tokenizer._dsv41_caps = True
    SamplingParams.update_from_tokenizer = update_from_tokenizer


def _pack_head_trellis() -> tuple[str, int] | None:
    """Return (shard, K_bits) for the pack's EXL3 output head, or None."""
    import json

    d = os.environ.get("VLLM_EXL3_MODEL_DIR") or os.environ.get("VLLM_ENGRAM_MODEL_DIR")
    if not d:
        return None
    try:
        with open(os.path.join(d, "model.safetensors.index.json"), encoding="utf-8") as fh:
            wm = json.load(fh)["weight_map"]
    except Exception:
        return None
    shard = wm.get("head.trellis")
    if shard is None:
        return None
    try:
        with open(os.path.join(d, shard), "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            hdr = json.loads(fh.read(n))
        k_words = int(hdr["head.trellis"]["shape"][-1])
    except Exception:
        return None
    return shard, max(2, k_words // 16)


def _install_exl3_head() -> None:
    """Serve an EXL3 (trellis) output head from the pack, shim-only.

    Pack v3 replaced ``head.weight`` (BF16) with ``head.trellis``/``suh``/
    ``svh``/``mul1``.  The engine builds ``lm_head`` and its weight mappers only
    know ``head.weight``, so the checkpoint keys do not resolve.  This wraps the
    served model class (the VL wrapper, whose ``language_model.lm_head`` this
    build uses) to

      1. declare ``lm_head`` a ``non_routed_exl3`` layer at the head's own K,
      2. rebuild ``lm_head`` against the EXL3 quant config so its
         ``create_weights`` allocates the trellis parameters, and
      3. map ``head.{trellis,suh,svh,mul1}`` onto ``language_model.lm_head.*``.

    Nothing in vLLM is edited; the model class is patched from the plugin
    entry point.
    """
    got = _pack_head_trellis()
    if got is None:
        return
    _shard, bits = got
    classes = []
    try:
        from vllm.models.deepseek_v4_1.nvidia.vl_model import (
            DeepseekV41ForCausalLM as _VL,
        )

        classes.append((_VL, "language_model.lm_head"))
    except Exception:
        pass
    try:
        from vllm.models.deepseek_v4_1.nvidia.model import (
            DeepseekV41LLMForCausalLM as _LLM,
        )

        classes.append((_LLM, "lm_head"))
    except Exception:
        pass
    if not classes:
        print("EXL3 head: model class not importable", flush=True)
        return

    import functools

    def _wrap(orig, head_prefix):
        @functools.wraps(orig)
        def __init__(self, *, vllm_config, prefix: str = ""):
            orig(self, vllm_config=vllm_config, prefix=prefix)
            try:
                qc = getattr(vllm_config, "quant_config", None)
                if qc is None or not hasattr(qc, "non_routed_exl3"):
                    return
                nr = dict(getattr(qc, "non_routed_exl3", None) or {})
                layers = dict(nr.get("layers") or {})
                layers[head_prefix] = {"bits": int(bits)}
                nr["layers"] = layers
                nr["bits"] = int(bits)
                mods = list(nr.get("modules") or [])
                if "lm_head" not in mods:
                    mods.append("lm_head")
                nr["modules"] = mods
                qc.non_routed_exl3 = nr
                mp = getattr(self, "hf_to_vllm_mapper", None)
                if mp is not None:
                    for suf in ("trellis", "suh", "svh", "mul1"):
                        mp.orig_to_new_suffix["head." + suf] = head_prefix + "." + suf
                from vllm.model_executor.layers.vocab_parallel_embedding import (
                    ParallelLMHead,
                )

                cfg = getattr(self, "config", None) or vllm_config.model_config.hf_config
                inner = self
                if "." in head_prefix:
                    for part in head_prefix.rsplit(".", 1)[0].split("."):
                        inner = getattr(inner, part)
                old = getattr(inner, "lm_head", None)
                inner.lm_head = ParallelLMHead(
                    cfg.vocab_size,
                    cfg.hidden_size,
                    quant_config=qc,
                    prefix=head_prefix,
                )
                print(
                    "EXL3 head: %s rebuilt as EXL3 K=%d (old=%s) shard=%s"
                    % (head_prefix, bits, type(old).__name__, _shard),
                    flush=True,
                )
            except Exception as exc:  # never fail the boot over the optional path
                print(f"EXL3 head install failed: {exc!r}", flush=True)

        __init__._dsv41_exl3_head = True
        return __init__

    for cls, head_prefix in classes:
        if getattr(cls.__init__, "_dsv41_exl3_head", False):
            continue
        cls.__init__ = _wrap(cls.__init__, head_prefix)


def _install_dsml_stop() -> None:
    """End a tool-call reply once a DSML invoke has closed and anything but DSML follows.

    After ``</|DSML| invoke>`` the only valid continuations are another
    ``<|DSML| invoke`` or ``</|DSML| calls>``.  At temperature 1.0 the model
    sometimes skips the calls terminator and writes an unbounded run of
    invented closing tags (``</arguments>``, ``</function>`` ...) until EOS or
    ``max_tokens``.  The tool parser keeps the valid invokes and drops the rest,
    so the reply looks normal but can cost tens of thousands of decode steps.

    The stop string ``</|DSML| invoke>\n</`` fires on both the valid
    terminator and the junk, and is kept in the output, so every closed invoke
    still parses.  Only requests that carry tools are touched.
    """
    if os.environ.get("DSV41_DSML_STOP", "0") in ("0", "", "false"):
        return
    try:
        from vllm.parser.deepseek_v41 import DeepSeekV41Parser
    except Exception as exc:  # pragma: no cover
        print(f"DSML stop install failed: {exc!r}", flush=True)
        return
    if getattr(DeepSeekV41Parser.adjust_request, "_dsv41_dsml_stop", False):
        return
    orig = DeepSeekV41Parser.adjust_request
    stop_str = "</\uff5cDSML\uff5c invoke>\n</"

    def adjust_request(self, request):
        request = orig(self, request)
        if getattr(request, "tools", None) and getattr(request, "tool_choice", None) != "none":
            stop = request.stop
            stop = [] if stop is None else ([stop] if isinstance(stop, str) else list(stop))
            if stop_str not in stop:
                request.stop = stop + [stop_str]
                request.include_stop_str_in_output = True
        return request

    adjust_request._dsv41_dsml_stop = True
    DeepSeekV41Parser.adjust_request = adjust_request


def register() -> None:
    """Install the DSV4.1 EXL3 config. Idempotent."""
    import vllm_exl3
    from vllm.model_executor.layers.quantization import (
        register_quantization_config,
    )
    from vllm_exl3.exl3 import Exl3Config

    # Installs the plugin's V4.1 compat wrappers (weight_block_size probe,
    # TP-geometry resolution, physical-K guard, ...) on Exl3Config.
    vllm_exl3.register()
    # ATS alias first, so the probe's POST_LOAD accounting sees the freed bytes.
    _install_ats_alias()
    _install_rope_trim()
    _install_exl3_head()
    _install_mem_probe()
    _install_dsml_stop()
    _install_sampling_caps()

    if getattr(register, "_installed", False):
        return

    @register_quantization_config("exl3_dsv41")
    class Dsv41Exl3Config(Exl3Config):
        """EXL3 routed experts, native block-32 FP8 for everything else."""

        # The pack's non-routed weights are the DeepSeek release layout:
        # E4M3 blocks with one UE8M0 scale per 32x32 block.
        weight_block_size = [32, 32]
        has_blocked_weights = True
        is_scale_e8m0 = True

        #: Every non-routed V4.1 weight is checkpoint block-32 FP8, whatever the
        #: config dict that built this instance happened to carry.
        DSV41_NON_ROUTED_FP8 = {
            "quant_method": "deepseek_v41_fp8",
            "weight_block_size": [32, 32],
            "is_checkpoint_fp8_serialized": True,
            "activation_scheme": "dynamic",
        }

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # The pack's config.json declares only ``{"quant_method": "exl3"}``;
            # every other field comes from --hf-overrides. Paths that re-derive a
            # config from the checkpoint instead of from the CLI overrides (the
            # DSpark drafter does: ``get_draft_quant_config`` ->
            # ``VllmConfig.get_quantization_config``) therefore lose
            # ``non_routed_quantization``, and every non-routed linear silently
            # falls back to ``UnquantizedLinearMethod``. V4.1's attention then
            # rejects it in ``_WOProjectionWeightMethod`` ("requires block32 FP8
            # weights"). Restore the declaration -- for this class it is a
            # property of the architecture, not of the config dict.
            if not getattr(self, "non_routed_quantization", None):
                self.non_routed_quantization = dict(self.DSV41_NON_ROUTED_FP8)

        @classmethod
        def from_config(cls, config):
            # ``Exl3Config.from_config`` assigns ``non_routed_quantization``
            # *after* construction, so the ``__init__`` default above is
            # overwritten with ``None`` on every path that re-derives the config
            # from the checkpoint dict. Re-state it here, which is the one entry
            # point both the CLI overrides and the DSpark drafter use.
            inst = super().from_config(config)
            if not getattr(inst, "non_routed_quantization", None):
                inst.non_routed_quantization = dict(cls.DSV41_NON_ROUTED_FP8)
            return inst

        @classmethod
        def get_name(cls) -> str:
            # The name DeepseekV4Model accepts. Not the registry key.
            return "deepseek_v41_fp8"

        @classmethod
        def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
            # The checkpoint declares quant_method "exl3"; the plugin's own
            # Exl3Config also claims that (it is probed later in the override list
            # and would win, restoring the name the model rejects). Claim it here,
            # under our registry key, so --quantization exl3_dsv41 survives
            # vLLM's _verify_quantization.
            method = str((hf_quant_cfg or {}).get("quant_method", "")).strip().lower()
            if method in ("exl3", "exl3_dsv41"):
                return "exl3_dsv41"
            return None

    register._installed = True