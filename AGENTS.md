# DSV4.1-Flash-1017 on one DGX Spark (REXL3 1.55 bpw on vLLM)

Serve vcruz305's SAGE REXL3 1.55 bpw pack (through the overlay built by
`scripts/build_rexl3_overlay.sh`; `MODEL_NAME=rexl3-vic-overlay`), or his SAGE EXL3
1.62 bpw pack (the default `MODEL_NAME`), on ONE NVIDIA DGX Spark using the derived
vLLM image this repo builds. Read the README first: the
measured flag set, the cold-expert alias and the memory guard are not optional.

Do NOT:
- drop the cold-expert alias (`VLLM_EXL3_ATS_ALIAS=0`) and expect a boot: the
  pack does not fit resident on 121.7 GiB
- build the exllamav3 fork without `MOE_TILESIZE_K=16`: you get the 32-row tile,
  ~34 % slower per step (190.6 vs 142.0 ms), and none of the README numbers apply
- raise `--max-num-batched-tokens` to 8192 without re-measuring: it OOMs
- enable `B12X_AUTOTUNE` as a "helpful" default (hours of bring-up); prefix caching
  is on by default and measured (README "Prefix caching"), do not turn it off by default
- run the container without `seccomp=unconfined --security-opt label=disable
  --ulimit memlock=-1`: the Engram reader's io_uring init fails
- serve without `scripts/guard.sh` armed (0.5 GiB MemAvailable floor, 64 MiB
  swap-growth allowance)
- print or commit `HF_TOKEN`
- leave benchmark/eval traffic in the route record: `touch <record dir>/pause` first

Exact commands: see README "Quick start". Verify a serve with
`bash scripts/smoke.sh` (must contain "Paris").

## Vision (image input)
- On by default (`VISION_IMAGES=1`, one image per prompt; N for more) on the usual
  `bash scripts/serve.sh` command; switching needs a restart (~4 min). It costs ~0.7 GiB of RAM (the tower's weights are served
  from the pack through the page cache, `DSV41_VISION_ATS=1`; text decode speed is
  unchanged within noise). `VISION_IMAGES=0` serves text only (`--language-model-only`).
- Check it: `bash scripts/smoke.sh`, then one chat request whose content holds an
  `image_url` part (base64 data URL) plus a text question. Give it
  `max_tokens` >= 1000: with 200 the reasoning used the whole budget.
- Only one engine fits: stopping it drops every client's in-flight request.
  Switch when nobody is using it, serve from this checkout (the guard's sudoers
  rule is tied to its path), and if the vision boot fails, boot text-only again.

## Repetition stop
- Requests without their own `repetition_detection` get `64:512:8`
  (`REP_STOP`; `REP_STOP=0` turns it off): a reply ends with
  `finish_reason="repetition"` once its last 8 blocks of 64-512 tokens repeat
  exactly. It stops reasoning loops that otherwise run to the 32,768-token cap.
