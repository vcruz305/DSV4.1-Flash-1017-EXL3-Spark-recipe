# Serving image for vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw on one DGX Spark.
#
#   base : dsv41-1017-ds41f:r6w   Christopher Owen's spark-ds41f r6w (upstreams
#                                 pinned at 2dc95329), built by base/docker/Dockerfile
#                                 (justincasey's 4x recipe at fa77c26, kept in base/).
#                                 The FP8 block-32 TileLang
#                                 linears, attention, Engram io_uring reader and
#                                 the loader stay exactly as that recipe pinned them.
#   +    : vcruz305/vllm-exl3     Exl3MoEMethod / ATS loader
#   +    : vcruz305/exllamav3     exl3_moe_mixedk (unified mixed-K), aarch64 guards
#   +    : dsv41_exl3_shim        quant-config name shim, no vLLM source changes
#
#   +    : TP1 decode tiles       docker/tp1_decode_tiles.py edits the base image's
#                                 TileLang decode tile tables (gemm.py, wo.py) for
#                                 TP1 shapes; bitwise-identical outputs, -3.4 ms/step.
#
# Nothing else in the base image is replaced.

ARG BASE_IMAGE=dsv41-1017-ds41f:r6w
FROM ${BASE_IMAGE}

# Re-declared inside the stage: an ARG before FROM is not visible to RUN/LABEL,
# and buildkit warns "UndefinedVar: $BASE_IMAGE" without it.
ARG BASE_IMAGE

ARG PLUGIN_REPO=https://github.com/vcruz305/vllm-exl3.git
ARG PLUGIN_REF=30e8b941376ce2d43bc8438439937c4809106635
ARG FORK_REPO=https://github.com/vcruz305/exllamav3.git
ARG FORK_REF=909ad8564fcb9e070caca8a65d380c741681e79c
ARG MOE_TILESIZE_K=16

ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=12.1a \
    MAX_JOBS=8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN set -eux; \
    if ! command -v git >/dev/null 2>&1; then \
        apt-get update; \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git; \
        rm -rf /var/lib/apt/lists/*; \
    fi; \
    git clone "$PLUGIN_REPO" /opt/vllm-exl3; \
    git -C /opt/vllm-exl3 checkout "$PLUGIN_REF"; \
    git -C /opt/vllm-exl3 rev-parse HEAD

# The served MoE kernel instance is compiled at tile-K 16, a #define in the fork.
# At the pinned FORK_REF the header already reads 16; the build still rewrites it to
# MOE_TILESIZE_K and asserts the result, so the build argument is never silently ignored.
RUN set -eux; \
    git clone "$FORK_REPO" /opt/exllamav3; \
    git -C /opt/exllamav3 checkout "$FORK_REF"; \
    git -C /opt/exllamav3 rev-parse HEAD; \
    PATCHER=/opt/vllm-exl3/tools/patch_exllamav3_aarch64.py; \
    if [ -f "$PATCHER" ]; then \
        sed -i 's|raise SystemExit(f"expected one x86 pause block in {path}")|print(f"already guarded (x86 pause): {path}")|' "$PATCHER"; \
        grep -q 'already guarded (x86 pause)' "$PATCHER"; \
        python3 "$PATCHER" /opt/exllamav3/exllamav3/exllamav3_ext; \
    fi; \
    F=/opt/exllamav3/exllamav3/exllamav3_ext/quant/exl3_moe_common.cuh; \
    sed -i -E "s/#define MOE_TILESIZE_K [0-9]+/#define MOE_TILESIZE_K ${MOE_TILESIZE_K}/" "$F"; \
    grep -n "define MOE_TILESIZE_K" "$F"; \
    grep -qx "#define MOE_TILESIZE_K ${MOE_TILESIZE_K}" "$F"; \
    CUSPARSE_DIR="$(find /usr/local/lib/python3*/dist-packages/nvidia -name cusparse.h -exec dirname {} \; 2>/dev/null | head -1)"; \
    if [ -z "$CUSPARSE_DIR" ]; then echo "ERROR: cusparse.h not found under NVIDIA Python packages" >&2; exit 2; fi; \
    export CPATH="$CUSPARSE_DIR${CPATH:+:$CPATH}"; \
    cd /opt/exllamav3; \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps .

RUN set -eux; \
    CUSPARSE_DIR="$(find /usr/local/lib/python3*/dist-packages/nvidia -name cusparse.h -exec dirname {} \; 2>/dev/null | head -1)"; \
    export CPATH="$CUSPARSE_DIR${CPATH:+:$CPATH}"; \
    export EXL3_EXT_INCLUDE=/opt/exllamav3/exllamav3/exllamav3_ext; \
    cd /opt/vllm-exl3; \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps .

COPY docker/tp1_decode_tiles.py /opt/dsv41-tools/tp1_decode_tiles.py
RUN python3 /opt/dsv41-tools/tp1_decode_tiles.py \
        /opt/spark3/candidate/vllm/vllm/models/deepseek_v4_1/tilelang

COPY shim /opt/dsv41-shim
RUN set -eux; \
    cd /opt/dsv41-shim; \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps .

LABEL local.dsv41.exl3.base="${BASE_IMAGE}" \
      local.dsv41.exl3.vllm-exl3="${PLUGIN_REF}" \
      local.dsv41.exl3.exllamav3="${FORK_REF}" \
      local.dsv41.exl3.moe-tilesize-k="${MOE_TILESIZE_K}" \
      local.dsv41.exl3.shim="dsv41_exl3_shim:register (quantization exl3_dsv41)"

ENTRYPOINT []