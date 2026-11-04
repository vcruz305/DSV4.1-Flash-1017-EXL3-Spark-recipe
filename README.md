# 🐋 DeepSeek V4.1 Flash on one DGX Spark

**DeepSeek-V4.1-Flash-1017 at ~30 tok/s on a single NVIDIA DGX Spark (GB10, 128 GB), served by vLLM
from the 1.55 bpw SAGE REXL3 pack, with the rarely used experts read from the SSD and picked from
your own traffic.**

> 📚 Every measurement, flag and limit is in **[TESTING-AND-RESULTS.md](TESTING-AND-RESULTS.md)**.
> This page is the short version.

## 🙏 Built on Vic's work

This project starts from [vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe](https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe)
by Victor Cruz (vcruz305): the vLLM image, the tuned flags, the cold-expert side file and the memory
guard. Both model packs are his too. His 1.62 bpw path still works unchanged here.

## ✨ What's new on top of Vic's recipe

| | Feature | What it does for you |
|---|---|---|
| 🧩 | **The 1.55 bpw REXL3 pack runs on vLLM** | One command makes the REXL3 pack loadable: it converts the head to BF16 and the non-routed layers to FP8, and hard-links the rest (+5.7 GiB of disk). Before, vLLM could not load it. Quality: HumanEval 96.3 %, GSM8K 97.0 %. |
| 🧠 | **Memory reserve for your other apps** | The Spark's 128 GB is shared by the GPU and everything else, and the model alone would fill it. You set `RESERVE_GIB` (default 12): the engine moves enough experts to the SSD that this much RAM stays free, and Hermes, tools and dashboards run inside a capped memory group of that size. Your apps get their RAM, and they can't crash the engine by taking more. 💡 The reserve shows as free in `free` or `htop`, but it isn't idle: until your apps need it, it caches SSD experts so decode stays fast, and it's handed back to them automatically the moment they ask. |
| 🎯 | **SSD cold set picked from your own traffic** | About 17 GiB of experts can't fit in RAM and are read from the SSD when a token needs them. Which experts go there decides the speed. The engine counts which experts your real requests use and sends the least-used ones to the SSD: only ~1.4 % of your agent's expert routes hit the SSD, against ~11 % with a generic pick. In a side-by-side test with 11 GiB of the reserve in use by other apps, Hermes decoded **3.6× faster** (16.4 vs 4.6 tok/s). |
| 🔄 | **Self-tuning at every restart** | Each start re-picks the cold set from all traffic recorded so far, and rebuilds it only when that cuts SSD reads by more than 5 % (~25 s). The record is a fixed ~146 KB file. Old traffic fades (14-day half-life), so the cold set follows a change in how you use it. Benchmark traffic can be paused out of it. |
| 👁️ | **Vision: send it images** | The pack's vision tower works on this vLLM build. On by default: send an image and the model reads it. Its weights are read from the SSD through the page cache, so it costs only ~0.7 GiB of RAM and ~10 s more boot, and text decode speed stays the same within noise (23.3 vs 22.9 tok/s over 3 alternating boots each). Shown a test image, it answered *"a red circle on the left, a blue rectangle on the right, and black text reading SPARK 42"* in 7 s. |
| 📊 | **Live SSD and cold-set view in [LIMCARS](https://github.com/AgentSparko/LIMCARS)** | The dashboard's hardware sidebar shows SSD read/write rates, whole-drive totals and temperature, the share of routes going to the SSD, the record size, the last rebuild, and how much of the reserve is in use. |
| 🧪 | **Agent-grade benchmarks** | Quality (HumanEval, GSM8K), 12 scored multi-step Hermes tool tasks, long context up to 128K, parallel requests and replays of real saved Hermes requests, all through the running server. |

## 🚀 Speed (REXL3 1.55 bpw, one Spark)

| Test | Result |
|---|---|
| ⚡ Decode, one request (short prompts) | **35–40 tok/s** |
| 🤖 Decode on real Hermes agent requests (17–21K-token prompts) | **30.3 tok/s** |
| 📥 Prefill (reading the prompt) | **~910–1,020 tok/s** at 32K–128K tokens |
| 📏 Long context, needle found at 50 % depth | ✅ 32K · ✅ 64K · ✅ 128K (128K prompt read in 129 s) |
| 🔁 Boot to ready | ~220–260 s |
| ✅ HumanEval (164) / GSM8K (200) | **96.3 %** / **97.0 %** |

👥 **Several requests at once:** the default serves one request at a time. `MAX_NUM_SEQS=4` lets 4 start
together (~1 s to first token each), but the total stays ~28 tok/s, because every request needs its own
experts.

## 🧠 Memory: what happens when your apps use the reserve

The engine leaves a **12 GiB reserve** for other programs. While they don't use it, that RAM caches cold
experts from the SSD. When they do use it, part of that cache is pushed out and re-read from the SSD, so
decode slows **gradually**, in proportion to how much of the reserve is in use.

Decode speed on held-out Hermes requests:

| Reserve in use | Cold set picked by Vic's bench prompts | 🎯 Cold set picked from your traffic |
|---|---|---|
| 0 GiB | 23.5 tok/s | **25.7 tok/s** |
| 11 GiB | 🐢 4.6 tok/s | **16.4 tok/s** |

Why: with the traffic-picked set only ~1.4 % of your agent's expert routes go to the SSD, against ~11 %
before.

⚖️ **Why 12 GiB and not less:** a smaller reserve is only ~5 % faster while your apps sit idle (34.0 vs
32.3 tok/s at ~3 vs 12 GiB). But with ~3 GiB, a speech (STT/TTS) pipeline (3-4 GB) or a browser agent on a
heavy site (~1 GB each) would leave too little free memory, and the guard would stop the engine. 12 GiB keeps
a Spark you can still use for other work. Running only the model? `RESERVE_GIB=6` takes that 5 % back.

## 🏁 Quick start

You need: one DGX Spark, ~400 GB free on NVMe, passwordless `sudo -n` for docker, sysctl, swapoff/swapon
and `tee /proc/sys/vm/drop_caches`, and a Hugging Face token with the pack's terms accepted.
Full prerequisites: [TESTING-AND-RESULTS.md → Quick start](TESTING-AND-RESULTS.md#quick-start-one-spark).

```bash
git clone https://github.com/AgentSparko/DSV4.1-Flash-1017-REXL3-Spark-vision-experts-offload.git
cd DSV4.1-Flash-1017-REXL3-Spark-vision-experts-offload
bash scripts/build_image.sh                    # vLLM image (base fetched from Hugging Face)
# download vcruz305's SAGE REXL3 1.55 bpw pack to ~/models/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw
bash scripts/build_rexl3_overlay.sh            # make it loadable by vLLM (GPU must be free)
MODEL_NAME=rexl3-vic-overlay HF_OVERRIDES_JSON="$(python3 tools/rexl3_overrides.py)" \
  RESERVE_GIB=12 ROUTE_STATS=1 DYN_RESIDENCY=1 DYN_RESIDENCY_MAX_SWAPS=0 bash scripts/serve.sh
bash scripts/smoke.sh                          # must answer "Paris"
```

### 👁️ Vision

Vision is on by default (one image per prompt; `VISION_IMAGES=N` allows N). Send images the OpenAI way, as
`{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` in the message content.
`VISION_IMAGES=0` serves text only (`--language-model-only`), which leaves ~0.7 GiB more for cold experts.

Switching needs a restart (~4 min). The author runs a Hermes skill that hot-swaps its own engine: it writes
a small script and runs it detached, with traps that fall back to the previous engine if anything goes wrong.
The same idea can turn vision on only when an image arrives, wherever a ~4 min reload is fine. It isn't part
of this repo; if enough people ask for it, it may be cleaned up and published.

The API is OpenAI-compatible on `http://127.0.0.1:8000/v1` (model `dsv41`). Requests that don't set
`temperature` or `top_p` get 0.6 / 0.95, and replies are capped at 32,768 tokens unless `max_tokens` says
otherwise.

⚠️ Always serve with the memory guard armed (`serve.sh` arms it). Keep the cold-expert side file and the
tested flags; the reasons are in [TESTING-AND-RESULTS.md](TESTING-AND-RESULTS.md).

## 🛠️ The scripts we added

| Script | What it does |
|---|---|
| `scripts/build_rexl3_overlay.sh` | One command to make the REXL3 pack loadable: converts the head to BF16 and the non-routed layers to FP8, and hard-links the rest (+5.7 GiB of disk). |
| `tools/rexl3_head_bf16.py`, `tools/rexl3_nonrouted_fp8.py` | The two conversions it runs. |
| `tools/rexl3_overlay_index.py` | Writes the overlay's tensor index. |
| `tools/rexl3_overrides.py` | Prints the quantization settings vLLM needs for REXL3. |
| `tools/plan_reserve.py` | Sizes the SSD cold set so `RESERVE_GIB` of RAM stays free for other programs. |
| `scripts/tenant_scope.sh` | Runs Hermes, tools or dashboards inside that reserve (a memory cgroup), so they can't starve the engine. |
| `scripts/py/dyn_residency.py` | Inside the engine: keeps a permanent count of which experts each request uses (a fixed ~146 KB file). |
| `tools/auto_cold.py` | At every start, re-picks the cold set from that count. It rebuilds only if SSD reads drop by more than 5 % (~25 s). |
| `tools/cold_share_eval.py`, `tools/pagecache_resident.py` | Show how many routes hit the SSD and how much of the cold set sits in RAM. |
| `bench/humaneval_api.py`, `bench/gsm8k_api.py` | Quality through the served API. |
| `bench/hermes_decode.py`, `bench/replay_hermes.py` | Speed on real, saved Hermes agent requests. |
| `bench/conc_decode.py` | Speed with several requests at once. |
| `bench/greedy_capture.py` | Saves greedy outputs, to check that a change leaves answers unchanged. |

Before running benchmarks, run `touch ~/.cache/dsv41-ds41f/route-heat/pause` so test traffic doesn't skew the record.

## 📄 Credits and license

- 🐋 Model: [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) by DeepSeek.
- 📦 SAGE REXL3 1.55 bpw and SAGE EXL3 1.62 bpw packs, the base recipe ([DSV4.1-Flash-1017-EXL3-Spark-recipe](https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe)), [vllm-exl3](https://github.com/vcruz305/vllm-exl3)
  and the [exllamav3 fork](https://github.com/vcruz305/exllamav3): **vcruz305 (Victor Cruz)**.
- ⚙️ Base engine: [christopherowen/spark-ds41f](https://github.com/christopherowen/spark-ds41f), on justincasey's base recipe;
  [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp; [vLLM](https://github.com/vllm-project/vllm).
- 🛠️ The REXL3 path, the reserve planner and tenant caps, the traffic-picked cold set, the LIMCARS views and the evals: AgentSparko.
- Scripts: MIT. Model weights keep their own licenses (see the packs' model cards).
