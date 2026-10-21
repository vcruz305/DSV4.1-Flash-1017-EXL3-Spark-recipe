# DSV4.1-Flash-1017-EXL3-Spark-recipe

Serving recipe for **vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw** on a single
NVIDIA DGX Spark (GB10, 128 GB unified memory), using vLLM. Every number below
was measured on that exact deployment; the flag set in `scripts/` is the one that
produced them.

**Model (Hugging Face):** <https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw>
(public, gated: accept the terms once, then use a token).

Base model: <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017>

The pack holds SAGE MixedK EXL3 routed experts (mixed K1-K4 per projection,
1.62 bpw), EXL3 MTP draft experts, an EXL3 K8 output head, native block-32 FP8
for every other linear, and the Engram n-gram tables. The Engram tables are
streamed from NVMe; everything else is meant to be device-resident. Under vLLM
on one Spark that does not quite fit, so this recipe serves it with a
**cold-expert alias side file**: the 1,418 least-routed experts plus the
embedding are gathered into one 64-byte-aligned file that the GPU reads through
a read-only mapping, which leaves every hot expert resident with ~4 GiB to spare
at `--max-model-len 32768` and 3.1-3.2 GiB at the shipped default of 524,288.

## Status

| Deployment | Folder | Model | Engine | Status |
|---|---|---|---|---|
| 1x Spark, TP1 | `scripts/` | vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw | spark-ds41f r6w + vllm-exl3 + exllamav3 (tile-K 16, warp-private MoE stages) + shim, DSpark k=5 | **Measured, serving** |

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

```bash
export HF_TOKEN=hf_...               # account must have accepted the model terms
bash scripts/download_and_verify.sh  # hf_xet download at the pinned revision + sha256 verify
bash scripts/build_image.sh          # base image (fetched from HF if missing) + our layer, ~10 min
bash scripts/build_cold_alias.sh     # 8.59 GiB side file, read-only on the pack
bash scripts/serve.sh                # detached; arms the memory guard
bash scripts/guard.sh status         # container + endpoint + memory snapshot
bash scripts/smoke.sh                # plain completion, must contain "Paris"
```

The API is OpenAI-compatible at `http://<host>:8000/v1`, model id `dsv41`.
`DRY_RUN=1 bash scripts/serve.sh` prints the full launch configuration without
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
stop` and `bash scripts/serve.sh` again; the second boot runs at the measured speed (see
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
| `--enable-prefix-caching` | default on: a request that repeats a cached prefix (follow-up turns, new questions on the same document) skips its prefill, 518K tokens to first token in 2.5-3.7 s instead of 464 s; no step-time cost measured. `PREFIX_CACHING=0` turns it off (see Prefix caching) |
| `B12X_AUTOTUNE=0` | the base image's optional per-kernel autotune sweep is hours; prebuilt artifacts are the measured path |
| `--security-opt seccomp=unconfined --security-opt label=disable --ulimit memlock=-1` | the Engram reader initialises io_uring; a default seccomp profile aborts it |
| `--shm-size=2g` | measured value; the engine's shared-memory use is small at `--max-num-seqs 1` |

Host side, while serving: `vm.compaction_proactiveness=0`,
`vm.watermark_boost_factor=0`, `vm.swappiness=0`, swap reset before boot.
`scripts/guard.sh stop` restores them.

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
ready-state split to `$RUN/ready-mem`, and `scripts/guard.sh status` prints it and warns
when the cache is under 1.5 GiB (normal boots 2.97-3.41 GiB, slow ones 0.40-0.81). Earlier,
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
  them.
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
- **Vision is not enabled** (`--language-model-only`); no image path was measured.
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
  the base); served through `scripts/serve.sh` with a freshly built alias and the
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
  `scripts/serve.sh` applies those overrides automatically while the JIT cache is
  cold, i.e. until a boot has reached ready once and left `$CACHE_DIR/kkref/jit/.warm`
  (a boot killed half-way leaves no marker, so the next one is serialised again).
  That first boot serves normally, but the speed figures were measured with the
  stock worker settings: run `scripts/serve.sh` once more after it is ready. The
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
`c852150e06f5`), so old commit ids stop resolving. `scripts/env.sh` pins the current
one, `c852150e06f5e5ee8c4c14f9cd1a9afc6bb21960`; the Hub revision has 27 files and the
21 this recipe needs all match `scripts/SHA256SUMS` (sha256 of the 17 shards, the
index, `config.json`, `tokenizer.json` and `tokenizer_config.json`; checked
2026-10-20). The other six (`.gitattributes`, `LICENSE`, `README.md`,
`chat_template.jinja`, `engram_token_map.u32`, `generation_config.json`) are not in
`SHA256SUMS`.

**`scripts/SHA256SUMS` is the authority, not the commit id.** If the pinned
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
- Model: DeepSeek-V4.1-Flash-1017 (MIT), quantized by
  [vcruz305](https://huggingface.co/vcruz305).
- Recipe scripts: MIT.

## Repo layout

| Path | What |
|---|---|
| `AGENTS.md` | short brief for AI agents: what this serves, the Do-not list, where the commands are |
| `scripts/env.sh` | pins (model revision, image, fork/plugin SHAs), paths, serve knobs |
| `scripts/download_and_verify.sh` | hf_xet download at the pinned revision + sha256 verification |
| `scripts/build_image.sh` | base engine + forks + shim image build |
| `scripts/fetch_base.sh` | fetches the prebuilt base from Hugging Face, verifies it, reinstalls CUTLASS DSL |
| `base/` | the base image's build files (justincasey's, unmodified), `Dockerfile.cutlass-dsl`, `cutlass-dsl.lock` |
| `docker/Dockerfile` | the layer: vllm-exl3, exllamav3 at tile-K 16, shim, TP1 decode tiles |
| `docker/tp1_decode_tiles.py` | TP1 decode tile tables for the base image's FP8 TileLang linears |
| `scripts/build_cold_alias.sh` | builds the cold-expert side file from your own pack copy |
| `tools/ats_build_alias.py` | the gatherer (64-byte alignment, manifest, selection) |
| `configs/cold-alias-selection.json` | the measured 1,418-expert membership |
| `scripts/serve.sh` | host wrapper: docker run, sysctls, memory guard |
| `scripts/serve_inner.sh` | the exact measured in-container vLLM invocation |
| `scripts/guard.sh` | memory guard, status, stop (restores sysctls) |
| `scripts/smoke.sh` | plain completion check |
| `scripts/SHA256SUMS` | per-file sha256 for the pinned pack revision |
| `shim/` | `dsv41_exl3_shim`: quant-config name shim, no vLLM source changes |
| `bench/bench7.py` | ms/step, accepted tok/step, tok/s per class |
| `bench/longprompt.py` | repeated-text prefill ladder |
| `bench/prefill_varied_probe.py` | varied-text prefill ladder |
| `bench/ctx_probe.py` | long-context probe: retrieval, prefill and decode at depth in one request |
| `bench/compare_tf.py` | teacher-forced top-1 / dlogprob comparison |
| `docs/ENGINEERING_NOTES.md` | what was tried, what won, and what lost |