#!/bin/bash
# Host wrapper: launch the serving container on ONE DGX Spark. Detached.
#
#   bash scripts/serve.sh              # measured configuration
#   DRY_RUN=1 bash scripts/serve.sh    # print the docker command, run nothing
#
# Requires passwordless docker (sudo -n), the pack downloaded and verified, and
# the alias side file built (scripts/build_cold_alias.sh).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$HERE/env.sh"

DRY_RUN="${DRY_RUN:-0}"
TS=$(date +%Y%m%d-%H%M%S)
RUN="${RUN_DIR:-$RUNS_DIR/$TS}"
# JIT caches (TileLang, Triton, B12X, CuTe DSL, CUDA, vLLM). $CACHE_DIR is mounted at
# /cache, so the env vars below must name the CONTAINER path; a host path there makes
# every compiled kernel land in the container's own filesystem and vanish with it, and
# the next boot compiles everything again.
JIT_HOST="$CACHE_DIR/kkref/jit"
JIT="/cache/kkref/jit"

PCF=--no-enable-prefix-caching
[ "$PREFIX_CACHING" = "1" ] && PCF=--enable-prefix-caching

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1: would launch container=$CONTAINER image=$IMAGE"
  echo "  model  : $MODEL_DIR -> /models/$MODEL_NAME"
  echo "  alias  : $ALIAS_DIR -> /alias"
  echo "  run dir: $RUN/ctr -> /mem (guard files stay in $RUN, outside the container)"
  echo "  jit    : $JIT_HOST -> $JIT (via $CACHE_DIR -> /cache)"
  echo "  port   : $API_PORT"
  echo "  maxlen : $MAXLEN  kv=$KVB  mnbt=$MNBT  lpt=$LPT  mtp_k=$MTP_K"
  echo "  graphs : $CG_MODE sizes=$CAPTURE_SIZES  gpu_util=$GPU_MEM_UTIL"
  echo "  ats    : layers=$ATS_LAYERS mtp=$ATS_MTP embed=$ATS_EMBED"
  echo
  echo "env:"
  echo "  B12X_AUTOTUNE=0"
  echo "  B12X_BIND_COMPILE_WORKERS=2 B12X_STATE_COMPILE_WORKERS=2 B12X_WEIGHTS_COMPILE_WORKERS=4"
  echo "  B12X_DENSE_SPLITK_TURBO=1 B12X_W4A8_TINY_DECODE=0"
  echo "  VLLM_DS41_MARKOV_NVFP4=1 VLLM_DS41_L2_PREFETCH=0 VLLM_DS41_ROPE_TRIM=1"
  echo "  VLLM_EXL3_MIXEDK_MODE=unified VLLM_EXL3_MOE_KERNEL=exllamav3 VLLM_EXL3_V41_NATIVE_MOE=0"
  echo "  VLLM_EXL3_ATS_ALIAS=1 VLLM_EXL3_ATS_ALIAS_DIR=/alias"
  echo "  VLLM_EXL3_ATS_ALIAS_LAYERS=$ATS_LAYERS VLLM_EXL3_ATS_ALIAS_MTP=$ATS_MTP VLLM_EXL3_ATS_ALIAS_EMBED=$ATS_EMBED"
  echo "  VLLM_DS41_ATTENTION_COMPUTE=auto VLLM_DS41_KERNEL_BACKEND=tilelang VLLM_DS41_HEAD_DTYPE=float32"
  echo "  VLLM_USE_FLASHINFER_SAMPLER=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  echo "  DSV41_DSML_STOP=$DSML_STOP"
  echo
  echo "vllm serve /models/$MODEL_NAME --served-model-name dsv41 --tokenizer-mode deepseek_v41 \\"
  echo "  --dtype bfloat16 --tensor-parallel-size 1 --distributed-executor-backend uni \\"
  echo "  --quantization exl3_dsv41 --block-size 256 $PCF --enable-chunked-prefill \\"
  echo "  --max-model-len $MAXLEN --max-num-seqs 1 --max-parallel-prefills 1 \\"
  echo "  --max-num-batched-tokens $MNBT --gpu-memory-utilization $GPU_MEM_UTIL \\"
  echo "  --kv-cache-memory-bytes $KVB --generation-config vllm --attention-backend tilelang \\"
  echo "  --linear-backend tilelang --moe-backend tilelang --jit-monitor-mode warn \\"
  echo "  --compilation-config {\"cudagraph_mode\":\"$CG_MODE\",...,\"cudagraph_capture_sizes\":[$CAPTURE_SIZES]} \\"
  echo "  --speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":$MTP_K,...} \\"
  echo "  --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice \\"
  echo "  --long-prefill-token-threshold $LPT --async-scheduling --no-scheduler-reserve-full-isl \\"
  echo "  --no-enable-flashinfer-autotune --language-model-only --mm-processor-cache-gb 0 \\"
  echo "  --hf-overrides {...exl3 routed experts + deepseek_v41_fp8 non-routed...}"
  exit 0
fi

# The container gets $RUN/ctr only: guard.sh runs as root and reads/appends files in $RUN
# (stage, guard.log, sysctl.before), so the container must not be able to rewrite them or
# swap them for symlinks.
mkdir -p "$RUN/ctr" "$JIT_HOST"/{b12x/compile,b12x/cute,cute-dsl,tilelang,triton,nv-compute,vllm-r5o}

# ---- first-boot compile headroom ------------------------------------------
# An empty JIT cache makes this boot compile ~156 TileLang/B12X kernels. The
# compile fan-out is the largest transient the recipe ever has: measured 25
# concurrent nvcc/cicc/cc1plus/ptxas processes, 20-390 MB each, ~2.5 GB on top
# of the engine. Min MemAvailable on a cold cache was 1.43 GiB against ~3.0 GiB
# warm, so a box with ~1 GiB less headroom trips the 0.5 GiB guard and the
# first boot is killed (the failure a from-scratch install reported at 432 MiB).
# Serialising the compile costs nothing measurable (ready 302 s vs 305 s) and
# buys +1.24 GiB of headroom (min 2.67 GiB). Warm caches are untouched.
# A boot that reaches ready drops $JIT_HOST/.warm (see the stage watcher below),
# so a cache left half-built by a killed first boot still counts as cold.
if [ ! -f "$JIT_HOST/.warm" ]; then
  export B12X_BIND_COMPILE_WORKERS="${B12X_BIND_COMPILE_WORKERS:-1}"
  export B12X_STATE_COMPILE_WORKERS="${B12X_STATE_COMPILE_WORKERS:-1}"
  export B12X_WEIGHTS_COMPILE_WORKERS="${B12X_WEIGHTS_COMPILE_WORKERS:-1}"
  export TVM_NUM_THREADS="${TVM_NUM_THREADS:-1}"
  export TVM_BIND_THREADS="${TVM_BIND_THREADS:-1}"
  echo "cold JIT cache ($JIT_HOST, no .warm): serialising compile workers (B12X_*=1, TVM_NUM_THREADS=1)"
  echo "  this first boot serves normally; once it is ready, run serve.sh again so the server"
  echo "  runs with the measured worker settings (the cache is then warm and nothing recompiles)"
fi

[ -d "$MODEL_DIR" ] || { echo "model dir not found: $MODEL_DIR" >&2; exit 1; }
[ -f "$ALIAS_DIR/experts.bin" ] || { echo "alias side file not found: $ALIAS_DIR/experts.bin (run build_cold_alias.sh)" >&2; exit 1; }
[ -f "$ALIAS_DIR/manifest.json" ] || { echo "alias manifest not found: $ALIAS_DIR/manifest.json" >&2; exit 1; }
# The side file must hold the membership this recipe ships. A side file built by an
# earlier version has the same size but a different expert set (1,380 over 24 layers
# before, 1,418 over 43 now) and serves measurably slower.
if ! python3 - "$ALIAS_DIR/manifest.json" "$RECIPE_HOME/configs/cold-alias-selection.json" <<'PY'
import json, sys
have = json.load(open(sys.argv[1])).get("experts") or {}
want = json.load(open(sys.argv[2]))
norm = lambda d: {k: sorted(int(e) for e in v) for k, v in d.items()}
sys.exit(0 if norm(have) == norm(want) else 1)
PY
then
  echo "alias side file $ALIAS_DIR was built from a different expert selection than" >&2
  echo "configs/cold-alias-selection.json (an earlier recipe version?). Rebuild it:" >&2
  echo "  bash scripts/build_cold_alias.sh        (~10-20 s, rewrites it in place)" >&2
  if [ "${ALLOW_ALIAS_MISMATCH:-0}" != "1" ]; then
    echo "(ALLOW_ALIAS_MISMATCH=1 serves it anyway)" >&2
    exit 1
  fi
  echo "ALLOW_ALIAS_MISMATCH=1: serving it anyway" >&2
fi
sudo -n docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image not found: $IMAGE (run build_image.sh)" >&2; exit 1; }
# The image must be the one build_image.sh makes from these pins. An older image under
# the same tag (an earlier recipe version, or a hand-built one) serves different
# plugin/shim code, e.g. one that cannot load the v3 pack's EXL3 head, and fails at
# weight load with "no module or parameter named 'head'".
img_label() { sudo -n docker image inspect "$IMAGE" --format "{{index .Config.Labels \"$1\"}}" 2>/dev/null; }
IMG_PLUGIN=$(img_label local.dsv41.exl3.vllm-exl3)
IMG_FORK=$(img_label local.dsv41.exl3.exllamav3)
IMG_TK=$(img_label local.dsv41.exl3.moe-tilesize-k)
if [ "$IMG_PLUGIN" != "$PLUGIN_REF" ] || [ "$IMG_FORK" != "$FORK_REF" ] || [ "$IMG_TK" != "$MOE_TILESIZE_K" ]; then
  echo "image $IMAGE was not built from this recipe's pins:" >&2
  echo "  vllm-exl3 ${IMG_PLUGIN:-<none>} (want $PLUGIN_REF)" >&2
  echo "  exllamav3 ${IMG_FORK:-<none>} (want $FORK_REF)" >&2
  echo "  moe-tilesize-k ${IMG_TK:-<none>} (want $MOE_TILESIZE_K)" >&2
  if [ "${ALLOW_IMAGE_MISMATCH:-0}" != "1" ]; then
    echo "rebuild it with scripts/build_image.sh, or set IMAGE to the tag you built" >&2
    echo "(ALLOW_IMAGE_MISMATCH=1 serves it anyway)" >&2
    exit 1
  fi
  echo "ALLOW_IMAGE_MISMATCH=1: serving it anyway" >&2
fi

# ---- host memory policy (restored by guard.sh stop) ------------------------
SYSCTL_KEYS="vm.compaction_proactiveness vm.watermark_boost_factor vm.swappiness"
: >"$RUN/sysctl.before"
for k in $SYSCTL_KEYS; do
  printf '%s=%s\n' "$k" "$(cat "/proc/sys/${k//.//}")" >>"$RUN/sysctl.before"
done

sudo -n sysctl -q vm.compaction_proactiveness=0 vm.watermark_boost_factor=0 vm.swappiness=0
# swapoff fails when any page is still swapped in (common on a box that has been
# up a while). Under `set -e` that aborted the launch *after* the sysctls were
# changed and before the guard was armed. Warn and keep the pre-boot base.
if ! sudo -n swapoff -a; then
  echo "WARNING: swapoff -a failed; swap is still in use. Continuing, but the" >&2
  echo "         guard's swap-growth check will measure from a non-zero base." >&2
fi
sudo -n swapon -a 2>/dev/null || true
sync; sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches'
sleep 3

BASE_SWAP=$(awk '/^SwapTotal:/{t=$2}/^SwapFree:/{f=$2}END{printf "%d", t-f}' /proc/meminfo)
echo "MemAvailable: $(awk '/^MemAvailable:/{printf "%.1f", $2/1048576}' /proc/meminfo) GiB  swap_base: $BASE_SWAP KiB"

sudo -n docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

# Optional compile-thread overrides for a cold-cache first boot. Forwarded only
# when set: the measured server did not set them, and TVM reads an explicit value.
TVM_ENV=()
[ -n "${TVM_NUM_THREADS:-}" ] && TVM_ENV+=(-e "TVM_NUM_THREADS=$TVM_NUM_THREADS")
[ -n "${TVM_BIND_THREADS:-}" ] && TVM_ENV+=(-e "TVM_BIND_THREADS=$TVM_BIND_THREADS")

sudo -n docker run -d --name "$CONTAINER" \
  --gpus all \
  --memory=118g --memory-swap=118g --oom-score-adj 1000 --ulimit core=0 \
  --shm-size=2g \
  --security-opt=seccomp=unconfined --security-opt=label=disable \
  --ulimit=memlock=-1:-1 \
  -v "$MODELS_DIR":/models:ro \
  -v "$CACHE_DIR":/cache:rw \
  -v "$ALIAS_DIR":/alias:ro \
  -v "$RUN/ctr":/mem:rw \
  -v "$HERE/serve_inner.sh":/root/serve.sh:ro \
  -p "$API_PORT":8000 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e B12X_AUTOTUNE=0 \
  -e B12X_BIND_COMPILE_WORKERS="${B12X_BIND_COMPILE_WORKERS:-2}" \
  -e B12X_STATE_COMPILE_WORKERS="${B12X_STATE_COMPILE_WORKERS:-2}" \
  -e B12X_WEIGHTS_COMPILE_WORKERS="${B12X_WEIGHTS_COMPILE_WORKERS:-4}" \
  "${TVM_ENV[@]}" \
  -e B12X_COMPILE_CACHE_DIR="$JIT/b12x/compile" \
  -e B12X_CUTE_COMPILE_CACHE_DIR="$JIT/b12x/cute" \
  -e B12X_DENSE_SPLITK_TURBO=1 -e B12X_W4A8_TINY_DECODE=0 \
  -e CUDA_CACHE_MAXSIZE=4294967296 -e CUDA_CACHE_PATH="$JIT/nv-compute" \
  -e CUTE_DSL_ARCH=sm_121a -e CUTE_DSL_CACHE_DIR="$JIT/cute-dsl" \
  -e TILELANG_CACHE_DIR="$JIT/tilelang" -e TRITON_CACHE_DIR="$JIT/triton" \
  -e XDG_CACHE_HOME="$JIT" -e VLLM_CACHE_ROOT="$JIT/vllm-r5o" \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_DS41_ATTENTION_COMPUTE=auto \
  -e VLLM_DS41_KERNEL_BACKEND=tilelang \
  -e VLLM_DS41_HEAD_DTYPE=float32 \
  -e VLLM_DS41_PACKED_BF16_LM_HEAD=0 \
  -e VLLM_L2_PREFETCH_KERNELS=tilelang \
  -e VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1024 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_DISABLE_PYNCCL=0 -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900 \
  -e VLLM_EXL3_MOE_KERNEL=exllamav3 \
  -e VLLM_EXL3_V41_NATIVE_MOE=0 \
  -e VLLM_EXL3_MIXEDK_LEGACY=1 \
  -e VLLM_EXL3_MIXEDK_MODE=unified \
  -e VLLM_EXL3_TRIM_AFTER_LAYER=1 \
  -e VLLM_EXL3_MODEL_DIR="/models/$MODEL_NAME" \
  -e VLLM_EXL3_ATS_ALIAS=1 \
  -e VLLM_EXL3_ATS_ALIAS_DIR=/alias \
  -e VLLM_EXL3_ATS_ALIAS_LAYERS="$ATS_LAYERS" \
  -e VLLM_EXL3_ATS_ALIAS_MTP="$ATS_MTP" \
  -e VLLM_EXL3_ATS_ALIAS_EMBED="$ATS_EMBED" \
  -e VLLM_EXL3_ATS_PREFETCH=1 \
  -e VLLM_DS41_ROPE_TRIM=1 \
  -e VLLM_DS41_MARKOV_NVFP4=1 \
  -e VLLM_DS41_L2_PREFETCH=0 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e DSV41_MEM_PROBE=1 \
  -e DSV41_DSML_STOP="$DSML_STOP" \
  -e VLLM_EXL3_MEM_WATERFALL=0 \
  -e VLLM_EXL3_MEM_WATERFALL_PATH= \
  -e MAXLEN="$MAXLEN" -e KVB="$KVB" -e MNBT="$MNBT" -e LPT="$LPT" \
  -e PREFIX_CACHING="$PREFIX_CACHING" -e MTP_K="$MTP_K" -e CAPTURE_SIZES="$CAPTURE_SIZES" -e CG_MODE="$CG_MODE" \
  -e GPU_MEM_UTIL="$GPU_MEM_UTIL" \
  --entrypoint /bin/bash \
  "$IMAGE" /root/serve.sh \
  >"$RUN/container_id.txt" 2>&1 || {
    echo "docker run failed:" >&2
    cat "$RUN/container_id.txt" >&2
    exit 1
  }

echo "LAUNCHED $CONTAINER ($(cat "$RUN/container_id.txt"))"
echo "run dir: $RUN"

# ---- stage breadcrumbs -----------------------------------------------------
# $RUN/stage used to be written once ("preflight") and never updated, so every
# line of the guard log -- including the KILL line -- named a stage that had
# long finished. Map vLLM's own log lines to a short stage instead.
echo "preflight" >"$RUN/stage"
# Detached (setsid nohup) so it outlives this shell and an ssh logout; it ends
# when the container's log stream ends.
# The awk program goes in as an argument: a single-quoted program inside the
# single-quoted bash -c body would end the outer quote and kill the watcher.
READY_MEM_AWK='/^Cached:/{c=$2} /^MemFree:/{f=$2} END{printf "page cache %.2f GiB, free %.2f GiB\n", c/1048576, f/1048576}'
setsid nohup bash -c '
  sudo -n docker logs -f "$1" 2>&1 | while IFS= read -r line; do
    case "$line" in
      *"Starting to load model"*)       echo "load-weights" >"$2/stage" ;;
      *"Graph capturing"*)              echo "graph-capture" >"$2/stage" ;;
      *"init engine"*)                  echo "engine-init" >"$2/stage" ;;
      *"Application startup complete"*) echo "ready" >"$2/stage"; : >"$3/.warm"
        awk "$4" /proc/meminfo >"$2/ready-mem" ;;
    esac
  done' _ "$CONTAINER" "$RUN" "$JIT_HOST" "$READY_MEM_AWK" >/dev/null 2>&1 &
echo "stage watcher: pid $!"

# ---- ready-state page-cache split ------------------------------------------
# A boot that comes up with ~3 GiB of page cache and little free memory decodes
# at 138-141 ms/step. A boot that comes up with ~0.5 GiB of cache and ~3 GiB
# free re-reads 40-110 MiB/step from NVMe instead of ~8, runs 3.6k-8.7k pages/s
# of direct reclaim and 250-1150 compaction stalls/s on the step thread, and is
# 1.5-3x slower for its whole life. That state is produced by an image build
# just before the boot and is not fixed by drop_caches, compact_memory or
# waiting: stop the serve and start it again. The split is written to
# $RUN/ready-mem when the server reports ready (see "Keep the disk quiet").

# ---- memory guard ---------------------------------------------------------
sudo -n setsid nohup bash "$HERE/guard.sh" watch "$CONTAINER" "$RUN/stage" "$RUN/guard.log" 524288 "$BASE_SWAP" \
  >/dev/null 2>&1 &
for _ in $(seq 1 100); do grep -q "guard: armed" "$RUN/guard.log" 2>/dev/null && break; sleep 0.1; done
if ! grep -q "guard: armed" "$RUN/guard.log" 2>/dev/null; then
  echo "WARNING: the memory guard did not arm (see $RUN/guard.log). The serve is UNGUARDED;" >&2
  echo "         stop it with: bash scripts/guard.sh stop" >&2
fi
echo "guard armed: MemAvailable floor 0.5 GiB, swap growth <= 64 MiB, 0.05 s"
echo "watch: bash scripts/guard.sh status   stop: bash scripts/guard.sh stop"