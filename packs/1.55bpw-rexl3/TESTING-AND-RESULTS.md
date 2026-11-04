# DSV4.1-Flash-1017 on one DGX Spark: REXL3 1.55 bpw on vLLM, SSD cold experts, traffic-driven residency

Serves **DeepSeek-V4.1-Flash-1017** on a single NVIDIA DGX Spark (GB10, 128 GB unified
memory) with vLLM, using vcruz305's **SAGE REXL3 1.55 bpw** pack. That pack is not
natively loadable by vLLM; this project makes it serve and keeps it fast while other programs
(an agent, its tools) share the box. Every number below was measured on that deployment.

What this project adds:
- **REXL3 1.55 bpw on vLLM**: the EXL3 head is dequantized to BF16 and the EXL3 non-routed
  linears transcoded to the block-32 FP8 the model's vLLM path expects, in a hard-link overlay
  of the pack (`packs/1.55bpw-rexl3/scripts/build_rexl3_overlay.sh`, `packs/1.55bpw-rexl3/tools/rexl3_*.py`). Quality through the served
  API: HumanEval 158/164 (96.3 %), GSM8K 194/200 (97.0 %).
- **Memory reserve for other tenants**: `RESERVE_GIB` sizes the cold-expert set so the rest of
  the box keeps that much RAM; a cgroup (`packs/1.55bpw-rexl3/scripts/tenant_scope.sh`) caps those tenants.
- **Cold set picked from real traffic**: a permanent, fixed-size (~146 KB) per-expert route
  record, saved across restarts; every boot re-picks the SSD cold set from it when that cuts
  the expected SSD bytes read per step by more than 5 %. On held-out agent (Hermes) requests
  this took decode from 4.6 to 16.4 tok/s with the reserve in use (README "Experimental").
- **Benchmarks and evals** for this box: `bench/` (HumanEval, GSM8K, long context up to 128K,
  parallel requests), and `touch <record dir>/pause` so they do not pollute the record.

Built on [vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe](https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe)
(Victor Cruz): the derived vLLM image, the measured flag set, the cold-expert alias loader, the
memory guard and his own 1.62 bpw pack. That serve path is kept and still works unchanged
(`MODEL_NAME` default); see "Credits and license".

**Models (Hugging Face):** [vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw) and
[vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw) (gated: accept the terms once, then use a token).

Both packs hold SAGE MixedK EXL3 routed experts, EXL3 MTP draft experts and the Engram n-gram
tables; the Engram tables are streamed from NVMe. Neither pack fits fully resident under vLLM on
one Spark, so both are served with a **cold-expert alias side file**: the least-routed experts
(plus the embedding) are gathered into one 64-byte-aligned file the GPU reads through a
read-only mapping, and every hot expert stays resident.

## Status

| Deployment | Model | Engine | Status |
|---|---|---|---|
| 1x Spark, TP1 | SAGE **REXL3 1.55 bpw** (overlay, `MODEL_NAME=rexl3-vic-overlay`) | spark-ds41f r6w + vllm-exl3 + exllamav3 (tile-K 16) + shim, DSpark k=5 | **Measured, serving here** |
| 1x Spark, TP1 | SAGE EXL3 1.62 bpw (default `MODEL_NAME`) | same image | Measured (Vic's recipe numbers) |

### REXL3 1.55 bpw quick start

After the image is built (Quick start below) and the REXL3 pack is downloaded to
`$MODELS_DIR/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw`:

```bash
bash packs/1.55bpw-rexl3/scripts/build_rexl3_overlay.sh  # hard-link overlay + BF16 head + FP8 non-routed (GPU free)
MODEL_NAME=rexl3-vic-overlay HF_OVERRIDES_JSON="$(python3 packs/1.55bpw-rexl3/tools/rexl3_overrides.py)" \
  RESERVE_GIB=12 ROUTE_STATS=1 DYN_RESIDENCY=1 DYN_RESIDENCY_MAX_SWAPS=0 bash packs/1.55bpw-rexl3/scripts/serve.sh
bash scripts/smoke.sh                        # must contain "Paris"
```

The FP8 re-quantization of the non-routed linears is lossy on top of EXL3 (error printed per
tensor); the quality numbers above are for this overlay.

## Quick start (one Spark)

Disk, measured on the measurement box: the pack is **326.6 GB** (304 GiB; two
shards are ~101.5 GB each), the alias side file **8.6 GiB**, the base engine image
**34.9 GB** (fetched from Hugging Face instead: an 8.8 GB download, deleted after loading, and
**~33 GB** in docker for the base plus our layer, measured), and the derived image adds **~2 GB**
on top of it (36.9 GB as listed by `docker images`, shared layers). The first boot adds a few GB of JIT cache under
`~/.cache/dsv41-ds41f/kkref/jit`. Budget **~400 GB free** on the NVMe the serve will
read cold experts from, and keep it well short of full (see "Keep the disk quiet").
`download_and_verify.sh` and `build_cold_alias.sh` check free space before writing.

Host prerequisites: passwordless `sudo -n docker`, `zstd`, and `huggingface_hub` with
`hf_xet` for the host-side downloads. `build_image.sh` fetches the base engine image
`dsv41-1017-ds41f:r6w` from Hugging Face when it is missing (see "Base image" below). On a stock Ubuntu 24.04 the documented install needs
`--break-system-packages` (PEP 668 marks the system python externally managed):

```bash
sudo apt-get install -y zstd
python3 -m pip install --user --break-system-packages -U 'huggingface_hub[hf_xet]'
```

(`numpy` is only needed for the optional `--select-npz` route-histogram path in
`tools/ats_build_alias.py`; the shipped `--experts-json` path is stdlib-only.)

`serve.sh` and `guard.sh` run everything through `sudo -n`, so the box's sudoers
must grant the operator at least: `docker`, `sysctl` (the three `vm.*` keys below),
`swapoff`/`swapon`, and `tee /proc/sys/vm/drop_caches` (the cache drop is
`echo 1 | sudo -n tee /proc/sys/vm/drop_caches`; the older
`sudo sh -c 'echo 1 > ...'` form needs a sudoers rule for `sh` itself, which
hardened boxes do not grant), plus `pkill` for `guard.sh stop`.

```bash
export HF_TOKEN=hf_...               # account must have accepted the model terms
bash packs/1.62bpw/scripts/download_and_verify.sh  # hf_xet download at the pinned revision + sha256 verify
bash scripts/build_image.sh          # base image (fetched from HF if missing) + our layer, ~10 min
bash packs/1.62bpw/scripts/build_cold_alias.sh     # 8.59 GiB side file, read-only on the pack
bash packs/1.62bpw/scripts/serve.sh  # detached; arms the memory guard
bash scripts/guard.sh status         # container + endpoint + memory snapshot
bash scripts/smoke.sh                # plain completion, must contain "Paris"
```

The API is OpenAI-compatible at `http://<host>:8000/v1`, model id `dsv41`.
`DRY_RUN=1 bash packs/1.62bpw/scripts/serve.sh` prints the full launch configuration without
touching the box.

If `$ALIAS_DIR` already holds a side file from an earlier version of this recipe,
rebuild it (`build_cold_alias.sh` rewrites `experts.bin` and `manifest.json` in
place, ~10 s): the shipped selection changed from 1,380 experts over 24 layers to
1,418 over 43, and the two files are the same size, so a stale one is not
detectable from the directory listing. `serve.sh` compares `$ALIAS_DIR/manifest.json`
with `configs/cold-alias-selection.json` and refuses a side file built from another
selection (`ALLOW_ALIAS_MISMATCH=1` overrides).

The first boot after the steps above can run 1.5-3x slow for its whole life (an image
build just before it leaves the cold working set out of the page cache). Once it is ready, run `bash scripts/guard.sh
stop` and `bash packs/1.62bpw/scripts/serve.sh` again; the second boot runs at the measured speed (see
"Keep the disk quiet").

`build_image.sh` and `serve.sh` share one image tag (`IMAGE`, default
`dsv41-1017-exl3:r6w`). If you build under another tag, pass the same `IMAGE=` to
`serve.sh`. `serve.sh` refuses an image whose build labels do not match this
recipe's pins (vllm-exl3, exllamav3, MoE tile K): an older image under the default
tag serves older shim code and fails at weight load with `no module or parameter
named 'head'`. Rebuild it with `build_image.sh`.

### Base image

The recipe's layer goes on top of `dsv41-1017-ds41f:r6w`: Christopher Owen's spark-ds41f r6w
with justincasey's DeepSeek-V4.1-Flash-1017 patches, built by `base/docker/Dockerfile` (see
`base/README.md`). That build needs no GPU but takes hours, so when the image is missing
`build_image.sh` runs `scripts/fetch_base.sh`, which:

1. downloads [vcruz305/DSV4.1-Flash-1017-Spark-base-image](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-Spark-base-image)
   at the revision pinned in `scripts/env.sh` (one 8.8 GB `.tar.zst`) and checks its sha256;
2. loads it as `dsv41-1017-ds41f:r6w-redist` and checks the digest of its single layer and its
   label set against the pins in `scripts/env.sh`;
3. builds `dsv41-1017-ds41f:r6w` from it with `base/Dockerfile.cutlass-dsl`, which installs NVIDIA
   CUTLASS DSL 4.7.1 from PyPI with `pip --require-hashes` (`base/cutlass-dsl.lock`).

The published image leaves out CUTLASS DSL because its licence allows redistributing only its
Python source; step 3 gets it from NVIDIA's PyPI release under NVIDIA's terms. The Hugging Face
repo carries the licences of everything else in the image, including the NVIDIA Deep Learning
Container License and the CUDA EULA that cover the CUDA libraries.

Measured on the Spark (2026-10-20): the download ran at ~55 MB/s (2.5 min), and `build_image.sh`
with no base present took 402 s (checksum, load, CUTLASS DSL, our layer). The resulting base has
the same image config and label set as the base this README was measured on, and the same 202,541
files: every source file is identical, and 18 differ, all build outputs. Those 18 are vLLM's two
compiled `_C` extensions, NCCL, TileLang's `libtvm_compiler.so` and an NCCL checksum file (rebuilt
from the same pinned sources; their compiled binaries are not byte-reproducible), pip records of
three source-built packages, and apt/ldconfig logs. Served from it, the recipe passed smoke and tool calls
(streaming and non-streaming) and measured 131.0 ms/step, 4.72 accepted tok/step, 36.0 tok/s on
`bench/bench7.py` (n=9, 0 engine errors), within the spread of the image on the measured base on
the same day (129.7 ms/step at 4.60 tok/step, and 127.6 at 5.07).

`BASE_FROM_SOURCE=1 bash scripts/build_image.sh` builds the base from `base/` instead.
`BASE_KEEP_ARCHIVE=1` keeps the downloaded archive in `$BASE_DL_DIR`.

## The measured flag set (do not deviate without re-measuring)

The engine is a derived image: `dsv41-1017-ds41f:r6w` (Christopher Owen's
spark-ds41f r6w, upstreams pinned at `2dc95329`) plus our two forks plus a
quant-config shim, plus TP1 decode tiles for the base image's FP8 TileLang
linears (`docker/tp1_decode_tiles.py`). Nothing else in the base image is replaced.

| Flag | Why it is there |
|---|---|
| `MOE_TILESIZE_K=16` (compile-time) | the served MoE kernel instance: 190.6 -> 142.0 ms/step, i.e. the 32-row tile is ~34 % slower (the `sed` in `docker/Dockerfile` only matters for older `FORK_REF`s; at the pinned `909ad85` the header already reads 16 and the build asserts the value) |
| `--quantization exl3_dsv41` + `--hf-overrides {...}` | EXL3 routed experts with the engine's native block-32 FP8 for every non-routed weight; the override restates it because the DSpark drafter re-derives the config |
| `--speculative-config dspark k=5` | k=5 is the step/token knee at this step time; k=7 costs +17 ms for +0.55 tok/step |
| `VLLM_EXL3_MIXEDK_MODE=unified` | one kernel for all K groups instead of one dispatch per group (+9 % mean) |
| `VLLM_DS41_MARKOV_NVFP4=1` | the engine's DSpark Markov draft block in NVFP4; measured structured-prompt gain (7.6 -> 11.2 tok/s at the time) |
| `VLLM_DS41_L2_PREFETCH=0` | prefetch measured 8.7 ms/step of work that hid nothing |
| `VLLM_EXL3_ATS_ALIAS=1` + `_LAYERS=38,39` + `_MTP=1` + `_EMBED=1` | ATS aliasing on, with the per-expert manifest in `/alias`. When the manifest lists experts (the cold set does), it decides membership and the layer list is ignored; the values are kept as measured |
| `--compilation-config cudagraph_mode=FULL_DECODE_ONLY`, sizes 1..7 | decode runs as full CUDA graphs (measured +11-14 % over eager) |
| TP1 decode tiles (`docker/tp1_decode_tiles.py`, baked into the image) | the base image's decode tile tables are keyed on TP4 shapes; at TP1 `block_K=256` for wo_b / wq_b / shared gate_up / down and a 32x256 tile for WO-A's 16-row bucket: -3.4 ms/step, bitwise-identical outputs |
| `--max-model-len 524288` | 512K tokens, the largest limit measured clean on one Spark: needle retrieval at 520K, minimum MemAvailable 2.23 GiB under that prefill, short-prompt decode 147.2 ms/step against 140.5-142.0 at 32K (see Measured long context). The model is rated for 1,048,576 (YaRN x16 over 65,536); 1M does not fit (the prefill transient grows with prompt length) |
| `--max-num-batched-tokens 2112 --long-prefill-token-threshold 2047` | the aligned-prefill chunk; 8192 does not fit |
| `--kv-cache-memory-bytes 1073741824` | 1 GiB KV at the engine's `auto` (bf16) cache dtype. The token count vLLM reports depends on `--max-model-len` for this hybrid attention (sliding-window and compressed groups): 236,089 at 32K, 394,892 at 64K, 595,001 at 128K, 760,070 at 229K, 959,768 at 524K (1.83x one full-length request) |
| `--gpu-memory-utilization 0.90` (`GPU_MEM_UTIL`) | **inert in this recipe**: vLLM ignores `--gpu-memory-utilization` whenever `--kv-cache-memory-bytes` is set, and `serve_inner.sh` always passes the latter — at boot the engine logs `reserved ... for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling. This does not respect the gpu_memory_utilization config`. Changing `GPU_MEM_UTIL` changes nothing; change `KVB`. The flag is kept only so the invocation matches the measured one |
| `--enable-prefix-caching` | default on: a request that repeats a cached prefix (follow-up turns, new questions on the same document) skips its prefill, 518K tokens to first token in 2.5-3.7 s instead of 464 s; no step-time cost measured. `PREFIX_CACHING=0` turns it off (see Prefix caching) |
| `B12X_AUTOTUNE=0` | the base image's optional per-kernel autotune sweep is hours; prebuilt artifacts are the measured path |
| `--security-opt seccomp=unconfined --security-opt label=disable --ulimit memlock=-1` | the Engram reader initialises io_uring; a default seccomp profile aborts it |
| `--shm-size=2g` | measured value; the engine's shared-memory use is small at `--max-num-seqs 1` |

Host side, while serving: `vm.compaction_proactiveness=0`,
`vm.watermark_boost_factor=0`, `vm.swappiness=0`, swap reset before boot.
`scripts/guard.sh stop` restores them.

**Engine of record.** The vLLM tree the server actually runs is the patched tree at
`/opt/spark3/candidate/vllm` inside the container image — not the stock
`dist-packages` copy. Before renaming, removing or "cleaning up" any
`--speculative-config` key, grep that tree
(`sudo -n docker exec dsv41-tp1 grep -rn <key> /opt/spark3/candidate/vllm/vllm/`):
for example `adaptive_verification_row_cost` (engine default 0.02; this recipe
sets 0.01) is a real field of the engine's speculative config
(`vllm/config/speculative.py`), while `adaptive_verification_cost_scale` does not
exist there; renaming the key leaves the engine at its default row cost without any error.

## Request defaults

`scripts/env.sh` sets `DEFAULT_MAX_TOKENS=32768`, which `serve_inner.sh` serves
as `max_new_tokens` inside `--override-generation-config` (together with the sampling
defaults below). It caps only
requests that send no `max_tokens` of their own — agent clients such as Hermes
send none, and without the cap a reply that falls into a repetition loop runs
until the 524,288-token window is full (observed: one reply generated for over
an hour at ~48 tok/s until the serve was reloaded). A client-sent `max_tokens`
always wins over the cap. `DEFAULT_MAX_TOKENS=0` passes no override and
disables the cap.

### Sampling defaults

The pack ships no `generation_config.json`, so a request that sends no sampling parameters
would run at vLLM's neutral defaults (temperature 1.0, top_p 1.0). Agent clients such as
Hermes send neither, which is how every agent chat on this serve ran at temperature 1.0 on a
1.62 bpw quant whose MTP drafts are greedy. `scripts/env.sh` now sets
`DEFAULT_TEMPERATURE=0.6` and `DEFAULT_TOP_P=0.95`, served through the same
`--override-generation-config` as the output cap (vLLM accepts `max_new_tokens`,
`temperature`, `top_k`, `top_p`, `min_p` and `repetition_penalty` there). A client that sends
its own value still wins per request; an empty value in `env.sh` ships no default. The greedy
bench (`bench/bench7.py` sends `temperature 0`) is unaffected.

Measured on the live serve, one request at a time (2026-10-21; raw samples in the campaign
files, `stability/lanes/{sampling,toolcalls}`):

- **Tool calling**, 12 agent scenarios (8 Hermes-like tools, streaming and non-streaming):
  valid call (right tool, parseable and complete arguments, no leaked markup) in 25/30 (83 %)
  at the neutral default, 24/28 (86 %) greedy, 25/28 (89 %) at 0.6 / 0.95. Every failure was a
  wrong or absent call, never a malformed one. The neutral default spent a mean 906 completion
  tokens and 46.7 s per call (worst short call 2,907 tokens, 167 s); 0.6 / 0.95 did the same
  calls in 277 tokens and 12.3 s. Greedy fell into a repetition loop on a 60K-token prompt and
  ran into both 600 s client timeouts.
- **Chat replies** (1200-token answers, 3 paired prompts): neutral default 15.7-19.5 tok/s,
  0.6 / 0.95 17.7-22.5 tok/s, greedy 19.3-34.5 tok/s but with a 27x repetition loop on one of
  the three prompts (the loop's acceptance is what makes it fast; the same signature as the
  >1 h runaway reply above). No loop at 0.6 / 0.95 on any measured prompt.

`min_p`, `top_k` and `repetition_penalty` were not measured and are not set. Two scenarios
failed under every setting and are model behaviour, not serve configuration: a ~60K-token
pasted-log prompt (2/12 valid calls) and a request to record a checklist (1/12).

## Client timeouts and queueing

The serve runs one request at a time (`--max-num-seqs 1`), so a request that
arrives while another is running waits in vLLM's queue and can sit there for
minutes with no bytes sent to the client. Prefill itself is also long: at the
measured ~1K tok/s a ~400K-token prompt takes ~7 min before the first token.
Set client timeouts accordingly: an agent client needs a stream-stale/read
timeout of at least 2400 s for long-context requests (a 900 s timeout was
observed killing a 424K-token prefill mid-stream). The manager's keep-alive wrapper (port 8889) exists precisely so
this does not kill legitimate prefills: while a request waits in the queue or
prefills, the wrapper sends SSE comments and leading-whitespace heartbeat
chunks, so a proxy or client that cuts an "idle" connection after ~180 s does
not terminate the request — but a client that treats those bytes as
first-token data will mis-measure TTFT, and a client connected directly to
port 8000 gets no keep-alive at all and must rely on its own timeout alone.

## Measured bring-up (single Spark)

| Phase | Measured |
|---|---|
| Weights routed | 118 s |
| Engine init | 13.9 s |
| Graph capture | ~1 s |
| **API ready (container start to `/v1/models` 200)** | **3 min 11 s** |

The three stages above are not the whole 3 min 11 s: ~58 s goes to process start,
tokenizer/index load and engine construction before the first shard is read.
Re-boots are faster once the JIT caches under `~/.cache/dsv41-ds41f/kkref/jit`
are warm; the first boot after an image build pays the TileLang/B12X compile
once.

## Measured memory (121.7 GiB unified)

| Metric | Value |
|---|---|
| Engine reservation (`cuda_alloc` at ready) | 110.5 GiB |
| Steady-state MemAvailable | ~4.0 GiB at `--max-model-len 32768`, 3.7-3.8 GiB at 229376, 2.9-3.2 GiB at 524288 |
| Idle MemAvailable before boot (after drop_caches) | 116.7-119.3 GiB; every one of these reached 2.9-3.2 GiB at ready, so the idle reading does not predict the steady one |
| MemAvailable minimum under a 520K-token prefill | 2.23 GiB (0.93 GiB transient; floor 0.5 GiB) |
| Swap | 0 KiB growth, 0 kills |
| Cold-expert alias side file | 9,228,943,360 B (8.59 GiB) on disk, mapped read-only |
| KV pool | 1 GiB (959,768 tokens reported at 524K) |

All-experts-resident does not fit under vLLM on this part: un-aliasing layers
38 and 39 alone costs 4.6 GiB of transient load pressure and the guard kills the
boot. That is the whole reason for the side file. Guard policy: 0.5 GiB
MemAvailable floor, 64 MiB swap-growth allowance, 0.05 s sampling
(`scripts/guard.sh`). Do not serve without it armed.
The guard runs as root and keeps its files (`stage`, `guard.log`, `sysctl.before`) in the
run directory, which the container cannot see: it gets only `$RUN/ctr` at `/mem`.
`guard.sh stop` restores only the three sysctls `serve.sh` changes, and only integer values.

**Host memory at ready is a hard requirement, not a tuning knob.** The boots this
recipe was measured on came up with 2.6-3.3 GiB MemAvailable and 2.5-3.3 GiB of
page cache at ready (normal boots 2.97-3.41 GiB of cache). With under ~1.5 GiB of
page cache at ready, the cold experts are re-read from NVMe on every step and
decode runs ~3x slower for the whole life of the boot — a third party measured
395 ms/step at 0.5 GiB free against our ~130 ms/step. `serve.sh` writes the
ready-state split to `$RUN/ready-mem` and leaves a `slow-boot` warning when the
page cache at ready is under 1.5 GiB; `scripts/guard.sh status` prints both.
The fix is to free host memory first — `systemctl isolate multi-user.target` and
stopping heavy services — and then restart the serve once (`scripts/guard.sh
stop`, then `packs/1.62bpw/scripts/serve.sh`). Restarting without freeing memory does not
help: the condition does not decay on its own (see "Keep the disk quiet").

## Measured speed

**Cold-expert set.** Every speed figure on this page through the W2 row was measured with
the first cold set (1,380 experts over 24 layers). The shipped set (1,418 experts over 43
layers, re-selected from a route histogram that includes long and agentic prompts; same
side-file size and resident footprint) was measured on another GB10 box at `MAXLEN=32768`:
bench7 131.3 / 132.0 ms/step against 139.8 for the first set, an 18-request agentic replay
407.6 / 400.4 s against 763.4 s, NVMe read during one bench7 9.7-10.8 GiB against
13.5-18.2 (details in `docs/ENGINEERING_NOTES.md`).

On the measurement box at the shipped default `MAXLEN=524288` the shipped set is faster than
the first one: two boots each, alternating old/new with only `ALIAS_DIR` switched, bench7 mean
**130.5 ms/step** (129.7 / 131.3; per class prose 126.0, code 134.5, structured 131.0)
against **137.1** (138.4 / 135.7; 134.6 / 140.6 / 136.0), -4.8 %. NVMe read during one
bench7 fell from 10.27 / 10.60 GiB to 7.04 / 7.28 GiB. Ready time and resident footprint are
unchanged (193-198 s, 3.07-3.21 GiB MemAvailable at ready, 2.92-3.20 GiB page cache).

| Cold set | ms/step, two boots | tok/step | tok/s | NVMe read, one bench7 |
|---|---:|---:|---:|---:|
| first (1,380 experts / 24 layers) | 138.4, 135.7 | 4.73 | 34.1, 34.9 | 10.27, 10.60 GiB |
| **shipped (1,418 / 43)** | **129.7, 131.3** | 4.69 | 35.6, 36.3 | 7.04, 7.28 GiB |

One row per configuration. `bench7.py`, single stream, 512 generated tokens with
`ignore_eos`, three unique prompts per class (prose / code / structured).
`ms/step` and `tok/step` are read from the engine's own spec-decode counters;
`tok/s` is client-observed. Each row is one boot; this box's acceptance varies
run to run, so compare `ms/step` first.

| Configuration | ms/step | accepted tok/step | tok/s |
|---|---:|---:|---:|
| No draft (cold alias, pre-tile-K kernel) | 74.4 (ms/token) | 1.00 | 13.4 |
| DSpark k=6, grouped MoE, whole-layer alias (38/39 + MTP) | 296-427 | 4.7-5.7 | 13.5 |
| + cold-expert ATS alias | 202.8 | 5.02 | 24.2 |
| + second kernel variant (k4) | 188.8-191.2 | 5.00 | 25.7-27.5 |
| + tile-K 16 build, k=5 | 140.0-145.6 | 4.35-4.74 | 31.3-34.0 |
| + TP1 decode tiles for the non-MoE linears | 141.0-141.8 | 4.14-4.58 | 29.5-32.4 |
| + warp-private MoE pipeline stages (W2), at 524288 (serving) | 133.0-141.0 | 4.55-4.91 | 32.5-38.2 |

The decode-tile row is the mean of two boots alternated with two boots of the row above
(141.4 vs 144.8 ms/step); its tok/s spread is acceptance, not kernel speed.

Tile-K 16 configuration per class, two ship boots: prose 137.6-138.4 ms/step,
code 143.2-146.2, structured 142.2-144.5.

The rows above the W2 row were measured at `--max-model-len 32768`; at the default
524288 that configuration runs 145.9-147.2 ms/step (see Measured long context). The W2
row was measured at 524288: four boots of recipe-built images, mean 137.4 ms/step (per
class prose 134.4, code 138.3, structured 139.8). Two of them alternated with two boots
of the previous build, whose warm boot read 145.9; on the measurement launcher two
alternating pairs read 136.35 vs 144.2 ms/step (-5.4 %). Outputs are bit-identical to
the previous kernel (see Quality and correctness). Not counted: one first boot straight
after an image build that read 190 ms/step (see Keep the disk quiet).

Where the step time goes (nsys before the W2 kernel, attributed by CUDA graph, 139.4 ms/step): the 40
main layers verifying 6 rows take 124.4 ms (89 %), the DSpark drafter 9.2 ms (three
MTP layers over 5 rows 5.4, draft head 2.8, Markov steps 0.5), verify head and
sampling 2.9 ms. The MoE dominates the main layers. Its kernel already keeps enough
bytes in flight to saturate DRAM and is not short of warps (two CTAs per SM measured
12 % slower); what limits it is a fixed dependent chain per K-slice. The W2 kernel
took the CTA-wide barriers out of that chain; what is left is exposed latency on the
weight reads. Details in `docs/ENGINEERING_NOTES.md`.

### Parallel requests (`MAX_NUM_SEQS`)

The default serves one request at a time (`MAX_NUM_SEQS=1`); others queue. Raising it does not
add throughput here: every extra sequence routes to its own experts, so a step reads more
weight bytes (and more cold experts from the SSD) and each request slows down about in
proportion. `bench/conc_decode.py`, 256 tokens per request, Hermes-traffic cold set:

| `MAX_NUM_SEQS` | requests at once | total tok/s | per request decode tok/s |
| --- | --- | --- | --- |
| 1 | 1 | 27.7 | 35.4 |
| 1 | 2 (queued) | 15.3 | 12.9 / 22.2 (2nd waits 13.7 s) |
| 2 | 2 | 20.8–21.1 | 10.7–10.9 / 14.4–16.1 |
| 4 | 1 | 30.5 | 39.8 |
| 4 | 4 | 28.1 | 7.2 / 9.1 / 10.2 / 8.4 |

Single-request speed and bench7 are unchanged by the setting (prose 35.2, code 29.5–30.3 tok/s
median); the lowest MemAvailable seen was 12.4–12.8 GiB. So `MAX_NUM_SEQS=4` only helps when
several short requests should all start at once (time to first token ~1 s each instead of
waiting in line); total throughput stays ~28 tok/s.

Decision: `MAX_NUM_SEQS=1` stays the default. The decode step here is bound by weight reads, and
4 sequences route to ~4x the distinct experts, so a step costs nearly 4x and total throughput
stays flat. The cold set needs no concurrency term: the record counts every routed token of every
sequence, which is what a batched step reads.

## Measured prefill

Repeated-text ladder on the earlier (pre-tile-K) kernel (`longprompt.py`, plain
completions, `max_tokens 8`):

| Prompt tokens | TTFT | Prefill tok/s |
|---|---:|---:|
| 1,907 | 1.931 s | 988 |
| 7,631 | 6.586 s | 1,159 |
| 27,949 | 22.556 s | 1,239 |

Varied-text ladder on the serving configuration (`bench/prefill_varied_probe.py`,
python stdlib source text, plain completions, chunk 2112 / threshold 2047):

| Prompt tokens (approx.) | Prefill tok/s |
|---|---:|
| 1,600 | 533 |
| 7,000 | 718 |
| 14,000 | 720 |
| 24,000 | 963 |

Use ~720 tok/s as the real-text figure. A repeated sentence reads higher on the
same server (8K: 1,262 tok/s) because it routes to few experts. A 4096-token
chunk (threshold 4090; the threshold must be chunk - 6) was measured and roughly
halves varied-text prefill (446 vs 718 tok/s at ~7K), so 2112 stays.

## Measured long context

1 GiB KV pool throughout. One request per row: varied log-style filler (unique ids and
numbers on every line), a passphrase hidden at the given depth, a plain-text question at
the end, greedy, 256 generated tokens (192 on the two longest rows, 128 on the prefix-caching
row) (`bench/ctx_probe.py`). Every row is a full prefill: the earlier rows were measured with
prefix caching off, and the probe now starts each prompt with a per-run nonce, so a repeated
run cannot hit the cache.

| `--max-model-len` | Prompt tokens | Needle depth | TTFT | Prefill tok/s | Decode tok/s at depth | Retrieved | Min MemAvailable |
|---:|---:|---:|---:|---:|---:|---|---:|
| 229,376 | 32,741 | 0.5 | 32.6 s | 1,005 | 31.5 | yes | 3.28 GiB |
| 229,376 | 131,034 | 0.5 | 111.1 s | 1,179 | 26.5 | yes | 3.13 GiB |
| 229,376 | 224,983 | 0.5 | 196.1 s | 1,147 | 28.9 | yes | 2.98 GiB |
| 229,376 | 224,983 | 0.1 | 195.5 s | 1,151 | 31.5 | yes | 3.01 GiB |
| **524,288 (default)** | 393,176 | 0.1 | 380.6 s | 1,033 | 29.3 | yes | 2.07 GiB |
| **524,288 (default)** | 519,984 | 0.5 | 516.4 s | 1,007 | 29.3 | yes | 2.23 GiB |
| **524,288 (default), shipped cold set** | 518,103 | 0.5 | 470.2 s | 1,102 | 25.6 | yes | 2.19 GiB |
| **524,288 (default), shipped cold set, prefix caching on (default)** | 517,979 | 0.3 / 0.7 | 466.6 s | 1,110 | 22.9 | yes | 2.39 GiB |

Short prompts (bench7, 512 tokens, k=5, warm boot), by `--max-model-len`:

| `--max-model-len` | Ready MemAvailable | ms/step mean (prose / code / structured) | tok/step | tok/s |
|---:|---:|---|---:|---:|
| 32,768 | 3.9-4.0 GiB | 140.5-142.0 | 4.35-4.67 | 31.3-34.0 |
| 229,376 | 3.7-3.8 GiB | 145.4 (139.7 / 150.6 / 145.9) | 4.38 | 30.3 |
| 524,288 | 3.1-3.2 GiB | 147.2 (141.4 / 153.8 / 146.4) | 4.10 | 27.1 |
| **524,288, W2 kernel (serving)** | 3.12-3.21 GiB | **137.4 (134.4 / 138.3 / 139.8)** | 4.73 | 35.1 |

The first three rows are the previous kernel; the W2 row is the mean of four boots.

The step-time cost of the longer limit is 1-5 %; the tok/s spread is mostly acceptance
(tok/step moves between runs on the same prompts). Use `MAXLEN=32768` if you never need
long prompts. Prefill does not slow with depth on this text (1.0-1.2K tok/s from 32K to
520K); it is faster than the stdlib-source ladder above because log lines route to fewer
experts.

**Prefix caching (default on).** `serve.sh` serves with `--enable-prefix-caching`;
`PREFIX_CACHING=0` turns it off. A request whose prompt starts with a prefix that is still in the
KV pool skips the prefill of that prefix: follow-up turns in a chat or an agent loop, and new
questions about a document already sent. Measured at the shipped default (`MAXLEN=524288`,
1 GiB KV pool, shipped cold set, W2 kernel, one boot) on varied log text, greedy
(`bench/prefix_probe.py`; the rows with tok/step are `bench/prefix_accept.py`, which reads the
spec-decode and prefix-cache counters from `/metrics` around each request):

| Prompt tokens | Request | TTFT | Decode tok/s | tok/step | Retrieved |
|---:|---|---:|---:|---:|---|
| 517,979 | first (full prefill) | 466.6 s | 22.9 | | yes |
| 517,976 | same document, new question | 3.46 s, 3.64 s | 13.2, 33.0 | | yes |
| 517,979 | same request again | 3.70 s, 3.36 s | 13.6, 15.1 | | yes |
| 518,141 | follow-up turn (first request + its answer + a question) | 2.67 s, 3.51 s | 35.6, 28.1 | | yes |
| 517,916 | first (full prefill), 384 tokens | 464.3 s | 26.2 | 3.60 | yes |
| 517,916 | same request again, twice | 2.64 s, 2.48 s | 27.2, 23.7 | 3.69, 3.03 | yes |
| 131,051 | first (full prefill) | 108.0 s | 38.1 | | yes |
| 131,048-131,213 | new question / same request / follow-up turn | 1.26-1.41 s | 16.4-41.3 | | yes |
| 129,614 | first, then the same request twice, 384 tokens | 104.7 s, then 1.21 s, 1.12 s | 22.4, 24.6, 24.7 | 2.92, 3.14, 3.10 | yes |

- The cache served 99.95 % of the prompt tokens on the repeated 518K requests
  (`vllm:prefix_cache_hits_total` / `queries_total`).
- Decode is not slower after a cache hit. The spread in decode tok/s is draft acceptance, which
  follows what the model writes: the same 518K request decoded 3.60 tokens per step cold and
  3.69 / 3.03 cached, at 127-137 ms per step, the same step time as short prompts. Summarizing a
  random log is harder to draft than bench7's prompts, so acceptance at depth is lower.
- No cost when nothing is reused: bench7 (its prompts share no long prefix) gave 127.6 ms/step
  (123.0-133.8), 5.07 tok/step, 39.6 tok/s with prefix caching on, against 129.7 / 131.3 ms/step
  off on two boots of the same image and cold set.
- Memory: minimum MemAvailable 2.39 GiB over the whole run (0.2 s sampling), no swap.
- The pool holds 959,768 tokens at this `MAXLEN`, 1.85x one full-length request, so one ~518K
  document stays cached at a time: a second long document evicts the first (least recently used).
  The cache lives in the KV pool on the GPU; a restart empties it.
- The first request with a new document still pays the full prefill (~1.1K tok/s on this text).

Earlier, on two other boxes at `MAXLEN=32768`, `KVB=268435456` (off on one, on on the other;
bench7 within 1 % between them), eight greedy lookups (`req=<id>` -> `latency_ms`) against one
shared 7.5K-token log prompt:

| Prefix caching | Lookups correct | Request time, first | Requests 2-8 | bench7 ms/step |
|---|---:|---:|---:|---:|
| off | 8/8 | 10.6 s | 9.9-12.0 s | 142.4-149.4 |
| on | 8/8 | 10.4 s | 1.6-2.6 s | 143.4 |

**Smaller KV pool at 32K.** At `MAXLEN=32768` one request needs far less than the 1 GiB
pool. `KVB=268435456` (256 MiB; vLLM reports 59,022 tokens, 1.80x one 32K request) gives
~1 GiB back to the page cache that holds the cold-expert alias. Measured on two other
boxes that have only ~1.0-1.1 GiB MemAvailable at ready in this config, bench7 as above,
`MAXLEN=32768` on every row; mean step time -9 % on box A, -14 % on box B:

| Box | `KVB` | Ready MemAvailable | ms/step mean | tok/s |
|---|---:|---:|---:|---:|
| A | 1 GiB | 1.07 GiB | 156.6, 163.1 | 28.1, 27.3 |
| A | 256 MiB | 1.93-1.98 GiB | 149.4, 142.4 | 30.5, 31.7 |
| B | 1 GiB | 1.1 GiB | 161.4, 160.2 | 27.4, 27.8 |
| B | 256 MiB | 2.15 GiB | 141.0, 136.5 | 32.4, 31.7 |

Not measured on the box above (~4 GiB free at 32K). Do not use 256 MiB with a long
`MAXLEN`: the pool must hold at least one full-length request.

**Why not 1M.** The prefill transient grows with prompt length (about 0.3 GiB at 225K,
0.4 GiB at 393K, 0.9 GiB at 520K). A `--max-model-len 1048576` server boots (KV 1.25 GiB,
ready MemAvailable 1.99 GiB, no RoPE-cache trim to give memory back), but a 524K prompt
on it drove MemAvailable to 0.44 GiB and the guard stopped it. 524,288 is the largest
limit that passed with margin.

**Keep the disk quiet while serving, and restart once after building.** Cold experts
are read from NVMe on every step. The first boot after an image build runs slow for its
whole life and the next boot is normal. Measured on the measurement box (98 % full NVMe,
same image and config, warm JIT cache, 1 s disk/vmstat/clock sampling, `bench7`):

| Boot | ms/step | NVMe read ms | NVMe MiB/s | compact_stall/s | pgscan_direct/s | page cache at ready |
|---|---:|---:|---:|---:|---:|---:|
| control, boot after a serve stop | 138.5 (126-147) | 0.63 | 58 | 3 | 0 | 2.97 GiB |
| after 12 GiB of `O_DIRECT` writes, no build | 138.9 (132-149) | 0.77 | 56 | 4 | 81 | 3.41 |
| after 119 GiB of pack reads, no writes | 135.3 (125-145) | 0.75 | 58 | 0 | 0 | 3.32 |
| first boot after `docker build --no-cache` | 230.1 (170-403) | 0.25 | 174 | 252 | 3614 | 0.81 |
| ... the same boot, re-benched at +10 / +20 min | 203.9 / 206.1 | | | | | |
| ... after a build with `drop_caches=3` + `compact_memory` before the boot | 420.9 (179-726) | 0.24 | 261 | 1145 | 8692 | 0.40 |
| ... and then stop + serve again | 141.3 (132-158) | 0.59 | 60 | 3 | 91 | 3.32 |

The drive is not the problem: it reads *faster* in the slow boots (0.25 ms, 174-261 MiB/s)
than in the control (0.63 ms, 58 MiB/s), and 12 GiB of direct writes does not reproduce
it. Boot-time free memory and fragmentation are not the problem either: the cleanest
pre-boot state of the matrix (117.3 GiB free, 67 % of it in >= 2 MiB blocks, 0.1 GiB of
cache) was the slowest boot. What changes is how much of the cold working set is resident
in the page cache once the engine owns its ~107 GiB. `MemAvailable` is ~3.1 GiB in every
arm; the split of that residual is what moves. With ~3 GiB of cold-expert pages resident a
step reads ~8 MiB from NVMe; with ~0.5 GiB resident it re-reads 40-110 MiB per step, and
the kernel runs direct reclaim and compaction on the faulting path (3.6k-8.7k pages/s
scanned, 250-1150 compaction stalls/s, 4-7x the major faults) even though ~3 GiB is free,
while the GPU sits at its normal 2496-2528 MHz. Why that free memory is not taken up by
the cache after a build is not pinned down; the reclaim apparently keeps evicting the pages
just read, because the state does not decay.

The condition is set by the load and does not decay: 230.1, then 203.9 at +10 min and
206.1 at +20 min on the same boot. So after `build_image.sh`, `download_and_verify.sh` or
`build_cold_alias.sh`, start the server, stop it (`scripts/guard.sh stop`) and start it
again before you measure or rely on it; waiting on a slow boot does not help, and
`drop_caches`/`compact_memory` before the boot do not either. `serve.sh` writes the
ready-state split to `$RUN/ready-mem` and a `slow-boot` warning file when the
cache at ready is under 1.5 GiB, and `scripts/guard.sh status` prints both and warns
(normal boots 2.97-3.41 GiB, slow ones 0.40-0.81). Earlier,
less instrumented runs agree: 358.6 ms/step after a build, 214.3 with `drop_caches` +
`compact_memory`, 190 -> 228 on one slow boot re-benched after 10 minutes, and 137-140 after
a restart. An older case on a 97 %-full disk ran 509.5 ms/step and 253 tok/s prefill for
25 minutes. Do not build images or copy large files on the serving disk while it serves.

## Quality and correctness

- **W2 MoE kernel (warp-private pipeline stages) versus the previous kernel:**
  bit-identical outputs in all 12 probe cells (rows 1 and 7, both projections, three
  layers), cold-bench parity rel-L2 5.8e-08, and the same 0.51 ulp against the fp32
  reference. In model, a teacher-forced comparison against a boot of the previous build
  (32 positions) read top-1 87.5-96.9 % and max |dlogp| 0.32-1.01; the same comparison
  between two boots of the previous build read 93.75 % and 0.771. Greedy decode on this
  engine is not bit-deterministic (atomic adds), so boots of one build differ too; at 32
  positions one token is 3.1 points of top-1.
- **Tile-K 16 numerics versus a true fp32 reference:** rel-L2 2.491e-4 (up) /
  2.188e-4 (down) against a 0.50-0.51 fp16 ulp floor, uniform in K and N. The
  build is not one ulp worse than the base kernel anywhere; the base-vs-tile-K
  delta is two independent realizations of the same 0.5-ulp C-buffer rounding.
- **Teacher-forced logprob checks** on every shipped arm (32 positions, full
  vocabulary): top-1 agreement 78.1-93.8 % against the non-speculative reference
  depending on the arm, max |dlogprob| inside that session's own noise floor.
  The gate exists to catch a wrong kernel, not to score quality.
- **Greedy output is not bit-deterministic** on this engine: batched atomics in
  the MoE reduce reorder sums. Two runs of the same prompt can differ in the
  tail. Do not build exact-match tests on it; compare logprobs.
- Pack-level quality figures are on the model card. This recipe did not re-run
  them. Measured here through the served API on the REXL3 overlay: **HumanEval
  158/164 = 96.3 % pass@1** (greedy, thinking off; `bench/humaneval_api.py`, which runs
  the tests in a network-less container; 1156 s at `--max-num-seqs 1`). **GSM8K 194/200 = 97.0 %**
  (`bench/gsm8k_api.py`, first 200 test problems, greedy, thinking off, flex-scored; 1 of
  the 6 misses hit the 1024-token cap). Pause the route record
  (`touch <record dir>/pause`) while these run so eval traffic does not pick the cold set.
- **Tool-call replies can run away without the DSML stop.** (In the DSML tags below, `|` stands for
  U+FF5C, the fullwidth bar the model actually emits.) At `temperature=1.0` the
  model sometimes closes an invoke and then, instead of `</|DSML| calls>`, writes
  invented closing tags (`</arguments>`, then `</function>` thousands of times) until
  EOS or `max_tokens`. The tool parser keeps the valid invokes and drops the rest, so
  the reply looks normal; only `usage.completion_tokens` shows it. Replaying two agent
  conversation states (1.9K and 63.7K-token prompts), 2 of 6 replies failed this way,
  one at 32,313 tokens / 648 s for 6 valid calls. `DSML_STOP=1` (the default) adds the
  stop string `</|DSML| invoke>\n</`, kept in the output, to requests that carry tools:
  12 of 12 replays of the same states ended in 6-278 s with every closed invoke parsed,
  streaming included. `DSML_STOP=0` turns it off.
  Checked on the measurement box with `DSML_STOP=1` (the default): one tool request at
  `temperature=0` and five at `temperature=1.0` (`get_weather(city)`, `max_tokens` 2048-4096)
  all ended right after the closed invoke — `finish_reason: tool_calls`, one parsed call with
  `{"city": "Lisbon"}`, 49-56 `completion_tokens` — and a request without tools was unchanged
  (`finish_reason: stop`, 21 tokens).

- **A missing parameter closer is repaired (`DSML_PARAM_REPAIR=1`, the default).** The rarer
  form of the same failure: the model leaves out `</|DSML| parameter>` and writes the invented
  closing tags into the argument (`git status</del></chat></dash></b>`). Without the repair the
  parser drops that argument in a normal reply (the call comes back as `{}`) and keeps the junk
  inside it when streaming; an agent would run either. The shim repairs only that structural
  violation: when a parameter's closer is missing, it strips the trailing run of closing tags
  that have no matching opening tag inside the value and restores the closer, printing one
  `DSML PARAM REPAIR` line per repaired call. A value whose closer is present is never touched,
  whatever it contains (`</div></body></html>`, `</div>\n</section>`). When streaming, each
  call's arguments are held until that call is complete and then sent in one piece (the call's
  id and name still arrive first); a valid call's assembled arguments are byte-identical to
  the parser's. Not repaired: a value cut off with no junk run, junk followed by more text, and
  requests without tools. Tested on the real parser and tokenizer with 17 synthetic replies
  built from the shape in issue #15, on all four entry points (`parse`, `extract_tool_calls`,
  `parse_delta`, `extract_tool_calls_streaming`, token by token): the 9 valid / no-junk /
  no-tools replies come out identical with the repair on and off, the 8 broken ones come out
  with the model's real argument. Live on this box with the repair on: 16 tool-call replies (two
  agent prompts, one and three calls per reply, `temperature` 0 and 1.0, streaming and not) all
  ended `tool_calls` with valid JSON arguments, and the streamed and non-streamed replies at
  `temperature=0` were identical. One of them hit the failure for real: the model closed a `bash`
  `cmd` with a plain `</parameter>` instead of the DSML closer; the repair stripped it and the
  reply's three calls came back clean (without it that call is `{}`). bench7 with the repair:
  129.7 ms/step. `DSML_PARAM_REPAIR=0` turns it off.

## Experimental: cold set from your own traffic (REXL3 overlay, local)

Which experts go to the SSD side file matters more than how many. Measured on REXL3 in this
image with a 17 GiB cold set (`RESERVE_GIB=12`), on held-out Hermes agent requests, greedy:

| cold set picked from | reserve unused | 11 GiB of the reserve in use |
|---|---|---|
| bench prompts (bench7 + prefill ladder) | 23.5 tok/s | 4.6 tok/s |
| Hermes traffic (permanent record) | 25.7 tok/s | 16.4 tok/s |

The bench-picked set takes ~11 % of agent routes; the traffic-picked one ~0.5-1.3 %.

- Record: `ROUTE_STATS=1 DYN_RESIDENCY=1 DYN_RESIDENCY_MAX_SWAPS=0` keeps a permanent
  per-expert route count in `$DYN_RESIDENCY_STATE_DIR` (default
  `~/.cache/dsv41-ds41f/route-heat/route-heat-$MODEL_NAME.npz`), added to across boots and
  saved by a background thread every `DYN_RESIDENCY_SAVE_S` (300) s when the counts changed
  (idle servers included), with an atomic rename. `serve.sh` asks the running server to save
  first (`flush.now` in the record dir, 15 s timeout), so a restart loses no counts.
  `DYN_RESIDENCY_RECORD=0` turns it off; a file whose shape does not match the pack is not mixed in.
  Benchmark traffic is recorded too: `touch <record dir>/pause` before a benchmark or eval replay and
  `rm` it after; paused traffic is left out of the record (it still drives this boot's swaps).
  The routes vLLM's profile run and CUDA-graph capture make at boot (dummy tokens, ~1.5 M per
  boot, piled on a few experts) are never added: the counters are baselined at the first real step.
  A marker that exists at boot pauses the record from that first step. (Before this fix it was
  only read by the saver thread, which starts at the first residency tick, 64 steps in, so the
  first minutes of a benchmark leaked in: 1.5-7.8 M routes per boot.) Checked live: 6 boots with
  the marker set, 72 replayed Hermes requests and the restarts' flushes, record unchanged except for
  the fading below (250,608,186 -> 250,560,557 routes).
- Old traffic fades: at each boot the record's counts are scaled by 0.5^(days since its last save /
  `DYN_RESIDENCY_HALF_LIFE_DAYS`) (14; 0 keeps every count). A week of coding a month ago then counts
  for about a quarter of the same week now, so the cold set follows a change of use while the
  history keeps its shape. Files written before this change carry no save time and are not scaled
  at their first load.
- Automatic: with `DYN_RESIDENCY=1` (or `AUTO_COLD=1`) every `serve.sh` start runs
  `packs/1.55bpw-rexl3/tools/auto_cold.py` first. It re-picks the cold set from the record at the same budget
  (`$ALIAS_DIR/plan.json` `cold_gib`, else `AUTO_COLD_GIB`) and rebuilds the side file only when
  that cuts the expected bytes a decode step reads from cold experts (sum of q_e x expert bytes, the
  quantity the picker's coldest-first fill minimizes for a byte budget; experts are 4.2-16.9 MiB) by
  more than `AUTO_COLD_MIN_GAIN` (0.05 = 5 %); the new file
  is built beside the old one and renamed in. `AUTO_COLD=0` keeps the side file as it is.
  Measured: a forced rebuild (`AUTO_COLD_MIN_GAIN=-1`, 17.02 GiB) added ~25 s to the restart
  (boot to ready 233 s vs 222 s when skipped); the build needs ~18 GiB free disk beside the old file.
- Rebuild by hand: `RESERVE_GIB=12 python3 packs/1.55bpw-rexl3/tools/plan_reserve.py --npz <record> --build`
- Sizing the reserve: `plan_reserve.py` also prints the reserve cgroup's peak (`reserve.slice`, or
  `RESERVE_CGROUP`) and the `RESERVE_GIB` that 1.5x that peak needs. The peak resets with the slice, so
  judge it after a long stretch of real Hermes use (LIMCARS keeps the highest one seen).
  Measured (2026-11-04, 12 scored Hermes tool tasks plus 6 engine boots, every process outside the
  engine container sampled every 2 s): the whole rest of the box peaked at 3.47 GiB RSS (NX remote
  desktop 1.09, Hermes's background process ~0.45, the `hermes chat` CLI 0.12, gnome-shell 0.31), and
  most of that is already running when the planner measures free memory. What the reserve
  costs when nobody uses it (calibration boots): 34.0 tok/s with the 7.2 GiB shipped cold set
  (`RESERVE_GIB` ~2) vs 32.3 tok/s at 17 GiB cold (`RESERVE_GIB=12`), about 5 %; `RESERVE_GIB=6` puts
  10.8 GiB cold. The default stays 12 GiB: that 5 % buys room for a browser tool, a build or a second
  model client without the 0.5 GiB guard stopping the engine. A box that only runs Hermes and a
  dashboard can use `RESERVE_GIB=6`.
  (or `tools/ats_build_alias.py --select-npz <record> --budget-gib <n>`), then serve with the
  `ALIAS_DIR`/`COLD_SELECTION` it prints.
- Inspect: `python3 packs/1.55bpw-rexl3/tools/cold_share_eval.py --eval <record> <selection.json>` prints the share of
  recorded routes a selection would send to the SSD. Back up = copy the file; reset = delete it.
- Replay saved Hermes requests to seed it (no tools run): `bench/replay_hermes.py`;
  decode check: `bench/hermes_decode.py`.

## Known limits

- **Oversized sampling lists get a 400, not a crash.** The DSpark path samples on vLLM's
  Model Runner V2, which raises inside the engine core, and so stops the whole server, on a
  request with more than 1,024 `allowed_token_ids`, more than 1,024 `logit_bias` entries,
  more than 128 stop ids together with `min_tokens`, or more bad words (count or total
  tokens) than `VLLM_MAX_NUM_BAD_WORDS` / `VLLM_MAX_BAD_WORDS_TOTAL_TOKENS`. The shim checks
  the same limits in the API server and answers 400 instead.
- **50-60 tok/s is not reached.** 31-34 tok/s is the best measured so far;
  kernel work continues on the fork branch. At the measured 4.7 accepted tokens/step,
  ~48 tok/s needs ~42 ms out of a 140 ms step (~50 tok/s needs ~45 ms); the
  ranked candidates and the arithmetic are in `docs/ENGINEERING_NOTES.md`.
- **Vision is on by default** (`VISION_IMAGES=1`; `VISION_IMAGES=0` gives `--language-model-only`, which sets every image limit to 0; text-only was the default until the 6-boot A/B below). The image's
  SM12x model path (`vllm/models/deepseek_v4_1/nvidia/vl_model.py`) does build a native vision tower,
  and keeps MTP. The pack carries it: 259 tensors, 0.77 GiB BF16, all in shard 00001. `VISION_IMAGES=N`
  serves up to N images per prompt instead. Measured (REXL3 1.55 bpw, `VISION_IMAGES=1`, same flags):

  | | text only | vision on (`VISION_IMAGES=1`) |
  |---|---|---|
  | boot to ready | 220 s | 244 s |
  | MemAvailable at steady state | 13.72 GiB | 11.90 GiB (**~1.8 GiB** for the tower and its buffers) |
  | Hermes decode, 3 held-out requests x 2 passes | 24.4 tok/s | 25.5 tok/s (same within noise) |
  | 768x512 test image, "describe shapes, colours, text" | n/a | correct ("red circle", "blue rectangle", "SPARK 42"), 7.0 s, 62 tokens |

  **Tower weights from the SSD (`DSV41_VISION_ATS=1`, the default with vision on).** The 263 vision and
  aligner BF16 tensors (0.90 GiB) are mapped from shard 00001 through the page cache instead of a GPU copy;
  each is checked against the index (dtype, shape, 16-byte alignment) and byte-compared with the loaded
  tensor before the swap. GPU memory after load: 94.05 GiB, the same as text-only. The kernel can drop
  those pages under pressure and re-reads them on the next image (cold image 15.5 s, warm 6.7 s, both
  correct). Alternating A/B, same image, record paused, 3 boots each, 12 Hermes-derived requests per boot:

  | | text only | vision on, tower from the SSD |
  |---|---|---|
  | boot to ready | 209 / 210 / 209 s | 219 / 220 / 220 s |
  | MemAvailable idle after boot | 13.02 / 13.59 / 13.43 GiB (13.35) | 12.59 / 12.71 / 12.68 GiB (12.66, **~0.7 GiB**) |
  | decode, token-weighted, per boot | 22.3 / 23.4 / 23.1 tok/s | 23.0 / 23.6 / 23.4 tok/s |
  | decode, all 36 requests | **22.9 tok/s** | **23.3 tok/s** (same within noise) |

  An earlier single pair of boots read 25.2 vs 22.8 tok/s; the spread between boots of the same
  setting (up to 1.1 tok/s here) is larger than the difference between the settings.

  The image counts ~300 prompt tokens. With `max_tokens=200` the whole budget went to reasoning and the
  reply was empty; give image requests room to think.
- **Chat endpoint: no chat template ships in the pack, and none is needed.** The
  Hub revision has no `chat_template.jinja` and neither `tokenizer_config.json`
  nor `tokenizer.json` carries a `chat_template` field. `--tokenizer-mode
  deepseek_v41` does not need one: `vllm/tokenizers/deepseek_v41.py` overrides
  `apply_chat_template` and encodes the conversation in code. Measured on a
  from-scratch install without either file: one user message ("What is the
  capital of France?", `max_tokens 32`) returns HTTP 200,
  `finish_reason=stop`, `The capital of France is **Paris**.` Checked on a served
  request (fresh download, this recipe's flags): the thinking comes back in
  `message.reasoning` (`usage.completion_tokens_details.reasoning_tokens` counts it),
  and a request with `tools` returns `finish_reason: tool_calls` with parsed
  arguments (`--tool-call-parser deepseek_v41`). Reasoning effort goes in
  `chat_template_kwargs`: `{"reasoning_effort": "low"}` renders `Reasoning Effort: 50`;
  a top-level `reasoning_effort` field is ignored and the prompt keeps the default 75
  (checked with `/tokenize` + `/detokenize`). All speed figures here are plain
  `/v1/completions`. If you point the server at a different tokenizer mode, supply a
  template or chat will 400.
- **Image build: built and served from scratch on the measurement box.**
  `scripts/build_image.sh` on a fresh clone built the image in 269.6 s (+3.6 GB on
  the base); served through `packs/1.62bpw/scripts/serve.sh` with a freshly built alias and the
  Hub file set, it measured 145.5 ms/step, 4.71 tok/step, 31.9 tok/s at
  `--max-model-len 524288` (bench7, previous pin `exllamav3@42981e6`), passed
  `smoke.sh`, answered chat, and found a needle at 131K. The current pin
  (`exllamav3@909ad85`, W2) was built the same way from a clean clone of this repo and
  served 135.4 ms/step, 4.91 tok/step, 36.2 tok/s at 524288. Its MoE sources match the
  measured W2 build file for file except the aarch64 patcher edits (applied at build
  time), comments and a removed debug print. For the previous pin, every input of the
  baked image was also compared against what the measured server mounted:
  `plugin-moe-p9b` is byte-identical to `vllm-exl3@30e8b94` (21/21 `.py`),
  `build-tk16` is byte-identical to `exllamav3@42981e6` after the aarch64 patcher
  (14/14 files, including the K=16 header), the shim is identical, and
  `docker/tp1_decode_tiles.py` applied to the base image's `gemm.py`/`wo.py`
  reproduces `overlay-p12` exactly (comment-only diff). The compiled plugin
  extension is unchanged between the measured image's `2a740a3` and `30e8b94`
  (`git diff --stat 2a740a3 30e8b94 -- csrc setup.py` is empty). Not covered: the
  base image built on another machine (base-image drift).
- **First boot on a fresh box compiles the TileLang kernels, and the guard can stop
  it.** A from-scratch install on a tester's Spark was stopped by the guard at 432 MiB
  MemAvailable during this compile. Replayed from scratch on the measurement box
  (fresh clone, image, alias and an empty cache), the first boot compiled 156 kernels
  (1,817 files, 72 MB in `$CACHE_DIR/kkref/jit`). The compile fan-out is the cause: up to 25 concurrent
  nvcc/cicc/cc1plus/ptxas processes at 20-390 MB each, ~2.5 GB on top of the
  engine, at a point where the model reservation is already 107.4 GiB.
  Measured on one box, same image and pack: cold cache with the stock worker
  counts **min 1.43 GiB**; cold cache with `B12X_*_COMPILE_WORKERS=1` and
  `TVM_NUM_THREADS=1` / `TVM_BIND_THREADS=1` **min 2.67 GiB**, ready 302 s vs 305 s.
  `packs/1.62bpw/scripts/serve.sh` applies those overrides automatically while the JIT cache is
  cold, i.e. until a boot has reached ready once and left `$CACHE_DIR/kkref/jit/.warm`
  (a boot killed half-way leaves no marker, so the next one is serialised again).
  That first boot serves normally, but the speed figures were measured with the
  stock worker settings: run `packs/1.62bpw/scripts/serve.sh` once more after it is ready. The
  kernels persist in `$CACHE_DIR/kkref/jit` (mounted at `/cache`) and later boots
  compile nothing. Every number in this README was measured with a warm cache.
- **Steady MemAvailable can be lower than measured here.** The same from-scratch
  install read 1.6 GiB after the smoke test, against 3.1-3.2 GiB measured at
  `--max-model-len 524288` on the measurement box (4.0 at 32768). The from-scratch
  replay on the measurement box (fresh image, alias and cache, Hub file set) read
  2.87-3.05 GiB at ready and ~3.0 GiB after smoke and bench, with the engine's
  reservation identical to the measured runs (`cuda_alloc` 107.43 GiB, within
  0.02 GiB at `POST_LOAD`). The recipe's own footprint is the same on both runs, so
  the tester's 1.6 GiB points at what else that box holds. The idle reading before
  boot does not predict it: boots that started at 116.7, 117.2, 119.2 and 119.3 GiB
  idle all reached 2.9-3.2 GiB at ready. Check `docker stats` and the host's resident
  processes while serving. The 0.5 GiB guard floor still holds; if your box reads well
  under 3 GiB at ready, `MAXLEN=32768` buys back about 0.9 GiB.
- **`--max-num-batched-tokens 8192` does not fit.** 2112 is the measured value.
- **Context: 524,288 tokens, not the rated 1M.** Retrieval passed from 32K to 520K (see
  Measured long context); 1M does not fit this box's memory with the guard armed. The
  older speed rows were measured at `--max-model-len 32768`, where short prompts are
  1-5 % faster than at 524288; the W2 row was measured at 524288.

## Do not

- Do not serve without `scripts/guard.sh` armed (0.5 GiB floor, 64 MiB swap).
- Do not drop the cold-expert alias (`VLLM_EXL3_ATS_ALIAS=0`) and expect a boot:
  the pack does not fit resident.
- Do not raise `--max-num-batched-tokens` to 8192 without re-measuring; it OOMs.
- Do not build the fork without `MOE_TILESIZE_K=16`; you get the 32-row tile,
  ~34 % slower per step (190.6 vs 142.0 ms), and none of these numbers apply.
- Do not enable `B12X_AUTOTUNE` (hours of bring-up) as a "helpful" default.
- Do not run the container without `seccomp=unconfined`; io_uring init fails.
- Do not print or commit `HF_TOKEN`.

## Cards tested on

| Card | VRAM | Status |
|---|---|---|
| NVIDIA DGX Spark (GB10) | 128 GB unified | Measured (this recipe) |

Nothing else has been verified. The recipe needs ~110.5 GiB of device
reservation plus 3.1-4.0 GiB of headroom depending on `--max-model-len`, and
depends on unified memory: the alias
side file only works because device and host memory are the same pool.

## Pack revision

The numbers were measured on a pack downloaded at revision `7cdebca3`. The Hub
repo's history has since been replaced three times, each time with a single
`Initial commit` holding the same weights (`04f7138b`, `2a6059359c68`, then
`c852150e06f5`), so old commit ids stop resolving. `packs/1.62bpw/scripts/env.sh` pins the current
one, `c852150e06f5e5ee8c4c14f9cd1a9afc6bb21960`; the Hub revision has 27 files and the
21 this recipe needs all match `packs/1.62bpw/scripts/SHA256SUMS` (sha256 of the 17 shards, the
index, `config.json`, `tokenizer.json` and `tokenizer_config.json`; checked
2026-10-20). The other six (`.gitattributes`, `LICENSE`, `README.md`,
`chat_template.jinja`, `engram_token_map.u32`, `generation_config.json`) are not in
`SHA256SUMS`.

**`packs/1.62bpw/scripts/SHA256SUMS` is the authority, not the commit id.** If the pinned
revision is gone, `download_and_verify.sh` downloads `main` with a warning and
the sha256 check decides whether it is the measured pack. A mismatch means the
weights changed; do not serve it as this recipe.

Earlier Hub revisions did not ship `engram_token_map.u32` (517,120 B) or
`chat_template.jinja` (4,015 B), both of which exist in the measurement box's
copy of the pack. Neither is in `SHA256SUMS` and neither is required: a
from-scratch install without them boots, passes `smoke.sh`, answers
`/v1/chat/completions` and retrieves a needle at 131K, with the same
`cuda_alloc` (107.43 GiB) and `AnonPages` (2.96 GiB) as the measurement box at
`POST_LOAD`.

## Credits and license

- Engine: [christopherowen/spark-ds41f](https://github.com/christopherowen/spark-ds41f)
  release r6w (Apache-2.0): vLLM fork, TileLang SM121 kernels, DeepSeek
  TileKernels, sparknet. The base image is built by `base/docker/Dockerfile`:
  justincasey's DeepSeek-V4.1-Flash-1017-4x-DGX-Spark recipe at `fa77c26`
  (Apache-2.0, its `LICENSE` and `NOTICE` kept in `base/`), copied unmodified because
  that repository is no longer online. The prebuilt copy on Hugging Face
  (`vcruz305/DSV4.1-Flash-1017-Spark-base-image`) carries the licences and NOTICE of every
  component in the image; NVIDIA CUTLASS DSL is not in it and is installed from PyPI.
- Runtime forks pinned here: `vcruz305/exllamav3` (mixed-K unified MoE kernels
  with warp-private pipeline stages, aarch64 guards) and `vcruz305/vllm-exl3` (EXL3 quant method, ATS loader).
- Models: [vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw) and
  [vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw),
  DeepSeek-V4.1-Flash-1017 (MIT) quantized by vcruz305.
- REXL3-on-vLLM overlay, reserve planner, tenant caps, route record, automatic cold-set
  rebuild and the `bench/` evals: AgentSparko, on top of the above.
- Scripts: MIT.

## Repo layout

Shared engine pieces (identical for both packs):

| Path | What |
|---|---|
| `AGENTS.md` | short brief for AI agents: what this serves, the Do-not list, where the commands are |
| `scripts/env.sh` | shared pins: base/engine image, fork/plugin SHAs, host paths, serve knobs both packs measured the same |
| `scripts/build_image.sh` | base engine + forks + shim image build |
| `scripts/fetch_base.sh` | fetches the prebuilt base from Hugging Face, verifies it, reinstalls CUTLASS DSL |
| `scripts/guard.sh` | memory guard, status, stop (restores sysctls) |
| `scripts/smoke.sh` | plain completion check |
| `base/` | the base image's build files (justincasey's, unmodified), `Dockerfile.cutlass-dsl`, `cutlass-dsl.lock` |
| `docker/Dockerfile` | the layer: vllm-exl3, exllamav3 at tile-K 16, shim, TP1 decode tiles |
| `docker/tp1_decode_tiles.py` | TP1 decode tile tables for the base image's FP8 TileLang linears |
| `shim/` | `dsv41_exl3_shim`: quant-config name shim, no vLLM source changes |
| `tools/ats_build_alias.py` | the gatherer (64-byte alignment, manifest, selection) |
| `configs/cold-alias-selection.json` | the measured 1,418-expert membership |
| `bench/` | the bench suite both packs measure through (bench7, prefill/long-context probes, and the 1.55 pack's API evals) |
| `docs/ENGINEERING_NOTES.md` | what was tried, what won, and what lost |

The 1.55 bpw REXL3 pack (`packs/1.55bpw-rexl3/`): `scripts/` holds env.sh (this
pack's knobs: MAX_NUM_SEQS, VISION_IMAGES, the overlay model name), serve.sh,
serve_inner.sh, build_rexl3_overlay.sh, tenant_scope.sh and the py/ loader hooks;
`tools/` holds the rexl3 overlay tools and the reserve/route-record tooling
(auto_cold, plan_reserve, cold_share_eval, pagecache_resident, test_dyn_record);
`configs/` holds reserve-calibration.json and the route-stats ladder; README.md
and TESTING-AND-RESULTS.md are this pack's measured record.
## Hermes multi-step tool tasks (measured 2026-11-04)

`bench/hermes_tasks.py` runs 12 real tasks through `hermes chat` against :8000 (REXL3,
repetition stop on), each in a fresh folder, and scores it with a hidden check the model
never sees. The route record is paused for the run.

| Task | Pass | Time | Tool calls |
|---|---|---|---|
| fix-bug | ✅ | 80 s | 9 |
| rename across files | ✅ | 136 s | 16 |
| CSV totals | ✅ | 31 s | 2 |
| find a file | ✅ | 25 s | 2 |
| module + tests | ✅ | 195 s | 9 |
| git branch flow | ✅ | 43 s | 5 |
| CLI run | ✅ | 51 s | 3 |
| JSON config edit | ✅ | 30 s | 3 |
| LRU cache | ✅ | 182 s | 7 |
| traceback fix | ✅ | 41 s | 6 |
| multi-file feature | ✅ | 208 s | 14 |
| log report | ✅ | 65 s | 4 |
| **total** | **12/12** | **18.1 min** | **80** |

No loops, no repetition stops, and the Hermes process tree peaked at 0.12 GiB RSS.

## Repetition stop and loop detection (measured 2026-11-04)

The shim gives every request that sets no `repetition_detection` the default
`REP_STOP=64:512:8`: a reply ends with `finish_reason="repetition"` once its last 8 blocks
of 64-512 tokens are identical. `REP_STOP=0` turns it off; a request's own setting wins.

| Test | Result |
|---|---|
| Forced exact loop ("write this sentence 400 times") | stopped after 1,417 tokens (~35 s), `finish_reason="repetition"` |
| Same loop, request opts out (`min_count: 0`) | ran to the 12,000-token cap (300 s), `finish_reason="length"` |
| Short reply that legitimately repeats a line | `finish_reason="stop"`, full text |
| Hermes FizzBuzz + tool calls | completed, no repetition event |
| GSM8K, first 100 problems | 99/100 (0 truncated) |
| HumanEval, first 40 problems | 40/40 |

### What a loop looks like in the MTP counters

With MTP (K=5), a model stuck in an exact loop is predicted perfectly by its own draft head:
every draft is accepted, so acceptance pins at 6.00 tokens/step and the decode-speed chart
shows a flat band at the ceiling. A real 9-minute Hermes reasoning loop held 6.00 every
minute; normal reasoning sits at 2-3. Copying a file back verbatim does **not** pin it:

| Case (acceptance every 5 s) | Tokens accepted per step | 5 s windows fully accepted | Longest run |
|---|---|---|---|
| Exact copy of a 13 KB JavaScript file | 4.6-6.0, mostly 5.2-5.8 | 1 of 25 | 5 s |
| Exact copy of a Python file (after reasoning) | 4.5-5.8 while copying | 0 of 56 | 0 s |
| Exact copy of 9 KB README prose | 4.4-5.5 | 0 of 18 | 0 s |
| Code audit + full rewrite of a 22 KB script | 2.0-5.0, mostly ~3 | 0 of 68 | 0 s |
| Real Hermes reasoning loop | 6.00 | every minute | ~9 min |

[LIMCARS](https://github.com/AgentSparko/LIMCARS) marks the Gen speed tile "possible loop"
after 30 s of full acceptance.

Live check of a client-side cut (a Hermes patch that polls `/metrics` every 2 s and cuts a
reply after 60 s of full acceptance), with the server stop opted out so the loop could run:
the "write this sentence 400 times" prompt drove the model's reasoning into
"I'll write 400 lines. I'll count chunks..." on repeat. Every draft was accepted from 68.7 s,
and the cut fired at 128.3 s. Per-second counters from a second run: 0.2-0.6 of drafts
accepted while reasoning normally, then 35/35 or 40/40 every second once the loop formed.
A verbatim 13 KB file echo under the same watch ran to completion (longest fully accepted
stretch 6.3 s, no cut).
