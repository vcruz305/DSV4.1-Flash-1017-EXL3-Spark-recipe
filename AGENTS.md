# DSV4.1-Flash-1017 on one DGX Spark (two packs, one shared engine)

Serve one of two packs of vcruz305's DeepSeek-V4.1-Flash-1017 quantizations on
ONE NVIDIA DGX Spark using the derived vLLM image this repo builds: the SAGE
EXL3 1.62 bpw pack (`packs/1.62bpw/`, the original measured recipe) or the
SAGE REXL3 1.55 bpw pack (`packs/1.55bpw-rexl3/`, served through
AgentSparko's overlay path with `MODEL_NAME=rexl3-vic-overlay`). Read the pack
README first: the measured flag set, the cold-expert alias and the memory guard
are not optional.

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
- leave benchmark/eval traffic in the 1.55 pack's route record: `touch <record dir>/pause` first

Exact commands: see the pack READMEs ("Quick start"). Verify a serve with
`bash scripts/smoke.sh` (must contain "Paris").

## Pack paths

- Shared engine: `scripts/build_image.sh`, `scripts/fetch_base.sh`,
  `scripts/guard.sh`, `scripts/smoke.sh`, `scripts/env.sh` (shared pins),
  `base/`, `docker/`, `shim/`, `tools/ats_build_alias.py`,
  `configs/cold-alias-selection.json`, `bench/`, `docs/`.
- 1.62 bpw pack: `packs/1.62bpw/scripts/` (`env.sh`, `download_and_verify.sh`,
  `build_cold_alias.sh`, `serve.sh`, `serve_inner.sh`, `SHA256SUMS`).
- 1.55 bpw REXL3 pack: `packs/1.55bpw-rexl3/scripts/` (`env.sh`,
  `build_rexl3_overlay.sh`, `serve.sh`, `serve_inner.sh`, `tenant_scope.sh`,
  `py/` loader hooks), `packs/1.55bpw-rexl3/tools/` (rexl3 overlay tools,
  auto_cold, plan_reserve, cold_share_eval, pagecache_resident,
  test_dyn_record), `packs/1.55bpw-rexl3/configs/`, and the pack's README.md and
  TESTING-AND-RESULTS.md.

## Vision (image input, 1.55 pack)

- On by default (`VISION_IMAGES=1`, one image per prompt; N for more) on the usual
  `bash packs/1.55bpw-rexl3/scripts/serve.sh` command; switching needs a restart (~4 min). It costs ~0.7 GiB of RAM (the tower's weights are served
  from the pack through the page cache, `DSV41_VISION_ATS=1`; text decode speed is
  unchanged within noise). `VISION_IMAGES=0` serves text only (`--language-model-only`).
  The 1.62 pack has no vision: its `serve_inner.sh` always passes
  `--language-model-only`, and every 1.62 number was measured that way.
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
