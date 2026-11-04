# DSV4.1-Flash-1017-EXL3-Spark-recipe

Serving recipe for **two packs of DeepSeek-V4.1-Flash-1017 on one NVIDIA DGX
Spark (GB10, 128 GB unified memory)**, served by vLLM from a single shared engine
image. Each pack has its own folder with its own measured flag set, its own
README, and its own numbers; the engine image, the memory guard, the bench
suite and the base-image build are shared at the top level.

| | Pack 1 (primary) | Pack 2 (secondary) |
|---|---|---|
| Model | [vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw) | [vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw) (through a local overlay) |
| Quantization | SAGE MixedK EXL3 routed experts, 1.62 bpw | SAGE REXL3 routed experts, 1.55 bpw (head to BF16, non-routed linears to FP8 by the overlay) |
| Engine path | exllamav3 mixed-K kernels (ATS alias side file) | vLLM through AgentSparko's REXL3 overlay (same image, same forks) |
| Vision | text only (`--language-model-only`; every 1.62 number was measured that way) | on (one image per prompt by default; `VISION_IMAGES=0` for text only) |
| Measured speed | 36.0 tok/s bench7, 4.7 accepted tok/step, 131 ms/step (see the pack README) | ~30 tok/s on real agent requests, 35-40 tok/s short prompts, prefill ~910-1,020 tok/s at 32K-128K (AgentSparko's measurements) |
| Cold experts | fixed measured 1,418-expert side file | SSD cold set re-picked from your own traffic at every start |
| Memory for other apps | none reserved | `RESERVE_GIB` (default 12) reserved and capped for tenants |
| Folder | [`packs/1.62bpw/`](packs/1.62bpw/README.md) | [`packs/1.55bpw-rexl3/`](packs/1.55bpw-rexl3/README.md) |

## Quick start (either pack)

The shared engine image is built once, from the repository root:

```bash
bash scripts/build_image.sh          # base image (fetched from HF if missing) + our layer, ~10 min
```

Then follow the pack's own quick start:

- **1.62 bpw pack:** download the pack with `packs/1.62bpw/scripts/download_and_verify.sh`,
  build the alias side file with `packs/1.62bpw/scripts/build_cold_alias.sh`, serve with
  `packs/1.62bpw/scripts/serve.sh`. Full measured quick start, prerequisites and numbers:
  [`packs/1.62bpw/README.md`](packs/1.62bpw/README.md).
- **1.55 bpw REXL3 pack:** download [the pack](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw) to
  `~/models/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw`, build the overlay with
  `packs/1.55bpw-rexl3/scripts/build_rexl3_overlay.sh`, then serve with
  `MODEL_NAME=rexl3-vic-overlay HF_OVERRIDES_JSON="$(python3 packs/1.55bpw-rexl3/tools/rexl3_overrides.py)"`
  and the reserve/route flags. Full quick start:
  [`packs/1.55bpw-rexl3/README.md`](packs/1.55bpw-rexl3/README.md).

Both serves arm the memory guard (`scripts/guard.sh`: 0.5 GiB MemAvailable floor,
64 MiB swap-growth allowance) and are verified with `bash scripts/smoke.sh`
(must answer "Paris"). `DRY_RUN=1` before either `serve.sh` prints the full launch
configuration without touching the box. The API is OpenAI-compatible at
`http://<host>:8000/v1`, model id `dsv41`.

## What is shared and what is per-pack

Shared (top level): the base-image build (`base/`), the engine layer
(`docker/`), the quant shim (`shim/`), the image build and fetch scripts, the
memory guard, the smoke check, the alias gatherer (`tools/ats_build_alias.py`),
the shared cold-alias membership (`configs/cold-alias-selection.json`), the bench
suite (`bench/`) and the engineering notes (`docs/ENGINEERING_NOTES.md`), plus
`scripts/env.sh` with the pins both packs measured identically (base/engine
image, fork and plugin SHAs, host paths, and the serve knobs they share).

Per-pack (`packs/<pack>/`): the model pins, the serve wrappers, the in-container
vLLM invocation, and the pack's own tooling, configs and documentation. Nothing
in one pack's folder is read by the other pack's serve path.

## Credits and license

- Packs: [vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw) and
  [vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw), both quantized
  from DeepSeek-V4.1-Flash-1017 by vcruz305 (Victor Cruz), who also wrote this
  recipe's engine path: the derived vLLM image, the tuned flag set, the
  cold-expert alias side file, the memory guard, and every 1.62 bpw measurement.
- Engine: [christopherowen/spark-ds41f](https://github.com/christopherowen/spark-ds41f)
  release r6w (Apache-2.0), built by `base/docker/Dockerfile` from justincasey's
  DeepSeek-V4.1-Flash-1017 recipe; [ExLlamaV3](https://github.com/turboderp-org/exllamav3)
  by turboderp; [vLLM](https://github.com/vllm-project/vllm). Runtime forks pinned
  here: `vcruz305/exllamav3` and `vcruz305/vllm-exl3`.
- Outside contributions merged into the recipe: adapt-ai-systems (opt-in prefix
  caching, the cold-set re-select from a route histogram, the DSML tool-call
  stop) and danielkuykendall23-boop (guard hardening and sampling caps).
- Recipe scripts: MIT. Model weights keep their own licenses (see the packs'
  model cards).

### AgentSparko

The 1.55 bpw REXL3 pack's serving path is the work of
[AgentSparko](https://github.com/AgentSparko), built in
[AgentSparko/DSV4.1-Flash-1017-REXL3-Spark-vision-experts-offload](https://github.com/AgentSparko/DSV4.1-Flash-1017-REXL3-Spark-vision-experts-offload)
on top of this recipe and upstreamed here: the REXL3-on-vLLM path (the hard-link
overlay that converts the EXL3 head to BF16 and the non-routed linears to FP8),
the memory-reserve planner and the tenant caps, the traffic-picked SSD cold set
and the permanent route record behind it, vision on this build, the server-side
repetition stop (PR #29), the agent-grade benchmarks and evals, the LIMCARS
dashboard views, and the README and TESTING-AND-RESULTS write-up that documents
all of it. His files are included under MIT with his commits preserved in this
repository's history; his measurements are his and are linked, not re-measured:
[`packs/1.55bpw-rexl3/TESTING-AND-RESULTS.md`](packs/1.55bpw-rexl3/TESTING-AND-RESULTS.md).

## Repository layout

- [`packs/1.62bpw/`](packs/1.62bpw/README.md) - the 1.62 bpw SAGE EXL3 pack: the original measured recipe
- [`packs/1.55bpw-rexl3/`](packs/1.55bpw-rexl3/README.md) - the 1.55 bpw SAGE REXL3 pack (AgentSparko's path)
- `scripts/`, `base/`, `docker/`, `shim/`, `bench/`, `docs/`, `tools/`, `configs/` - shared engine pieces (see the pack READMEs for the details)
- [`AGENTS.md`](AGENTS.md) - short brief for AI agents: what this serves, the Do-not list, where the commands are
