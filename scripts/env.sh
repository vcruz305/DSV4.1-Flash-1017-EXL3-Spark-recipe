#!/bin/bash
# Shared pins and paths. Source this; do not execute it.
#
# Every value here is the measured configuration. Changing one without
# re-measuring means the numbers in README.md no longer describe what you serve.

# ---- model ----------------------------------------------------------------
MODEL_REPO="vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw"
# Hub revision to download. The repo's history has been replaced more than once
# (each time a single "Initial commit" with the same files), so a commit id can
# stop resolving. scripts/SHA256SUMS is the authority: download_and_verify.sh
# falls back to main when this revision is gone, and the sha256 check decides.
# See README "Pack revision".
MODEL_REV="${MODEL_REV:-c852150e06f5e5ee8c4c14f9cd1a9afc6bb21960}"
MODEL_NAME="dsv41-1.62bpw-v3"          # directory name under $MODELS_DIR

# ---- images ---------------------------------------------------------------
# Base engine: Christopher Owen's spark-ds41f r6w (upstreams pinned at 2dc95329),
# built by base/docker/Dockerfile (see base/README.md).
# That Dockerfile's own base is vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf.
BASE_IMAGE="${BASE_IMAGE:-dsv41-1017-ds41f:r6w}"
IMAGE="${IMAGE:-dsv41-1017-exl3:r6w}"
# Prebuilt base: the redistributable copy of $BASE_IMAGE on Hugging Face (CUTLASS DSL removed,
# reinstalled from PyPI by scripts/fetch_base.sh). Pinned by revision, archive sha256, layer
# digest and label-set sha256.
BASE_HF_REPO="${BASE_HF_REPO:-vcruz305/DSV4.1-Flash-1017-Spark-base-image}"
BASE_HF_REV="${BASE_HF_REV:-17ed433666f4cc6dfa86bcc1d3f4448255fd58fb}"
BASE_ARCHIVE="dsv41-1017-ds41f-r6w-redist.tar.zst"
BASE_ARCHIVE_SHA256="97d31344b0e5f8e17bf69e05d679a25b3256a452a7275bc294eb6bd149f58e9a"
BASE_REDIST_IMAGE="dsv41-1017-ds41f:r6w-redist"
BASE_REDIST_LAYER="sha256:8d743923e09ad9d435255f9436b5aa1a42d2e88cced04d121b238f8a726990cc"
BASE_LABELS_SHA256="436ee3f65e6be1cd8f897eeb343c56a9a057a02069eb0d07aaccfcee2b4e73e4"
BASE_DL_DIR="${BASE_DL_DIR:-$HOME/.cache/dsv41-base-image}"

# Runtime forks baked into the layer above the base, pinned to the commits whose
# file hashes match the measured server (branches named below).
FORK_REPO="https://github.com/vcruz305/exllamav3.git"
FORK_BRANCH="feat/moe-warp-private-stages"
FORK_REF="${FORK_REF:-909ad8564fcb9e070caca8a65d380c741681e79c}"          # exllamav3: mixed-K unified MoE + aarch64 guards + warp-private MoE pipeline stages
PLUGIN_REPO="https://github.com/vcruz305/vllm-exl3.git"
PLUGIN_BRANCH="feat/dsv41-1017-tp1"
PLUGIN_REF="${PLUGIN_REF:-30e8b941376ce2d43bc8438439937c4809106635}"    # vllm-exl3: Exl3MoEMethod / ATS loader

# MoE tile-K the served kernel was compiled at. It is a compile-time constant in
# the fork, so the build rewrites the header before compiling.
MOE_TILESIZE_K="${MOE_TILESIZE_K:-16}"

# ---- paths ----------------------------------------------------------------
RECIPE_HOME="${RECIPE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
MODEL_DIR="${MODEL_DIR:-$MODELS_DIR/$MODEL_NAME}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/dsv41-ds41f}"
ALIAS_DIR="${ALIAS_DIR:-$CACHE_DIR/alias-cold}"
RUNS_DIR="${RUNS_DIR:-$RECIPE_HOME/runs}"

# ---- serve ----------------------------------------------------------------
CONTAINER="${CONTAINER:-dsv41-tp1}"
API_PORT="${API_PORT:-8000}"
MAXLEN="${MAXLEN:-524288}"      # largest measured clean limit (520K retrieval); 32768 is 1-5 % faster on short prompts
KVB="${KVB:-1073741824}"        # 1 GiB KV pool (vLLM reports 959,768 tokens at MAXLEN=524288)
MNBT="${MNBT:-2112}"
LPT="${LPT:-2047}"
MTP_K="${MTP_K:-5}"
CAPTURE_SIZES="${CAPTURE_SIZES:-1,2,3,4,5,6,7}"
CG_MODE="${CG_MODE:-FULL_DECODE_ONLY}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
PREFIX_CACHING="${PREFIX_CACHING:-1}"   # 1 = reuse KV of repeated prefixes (default); 0 = every request is a full prefill
DSML_STOP="${DSML_STOP:-1}"             # 1 = end tool-call replies at the first closing tag after a DSML invoke
DSML_PARAM_REPAIR="${DSML_PARAM_REPAIR:-1}"  # 1 = strip junk closing-tag runs from DSML parameters whose closer is missing
ATS_LAYERS="${ATS_LAYERS:-38,39}"
ATS_MTP="${ATS_MTP:-1}"
ATS_EMBED="${ATS_EMBED:-1}"

export MODEL_REPO MODEL_REV MODEL_NAME BASE_IMAGE IMAGE
export BASE_HF_REPO BASE_HF_REV BASE_ARCHIVE BASE_ARCHIVE_SHA256 BASE_REDIST_IMAGE BASE_REDIST_LAYER BASE_LABELS_SHA256 BASE_DL_DIR
export FORK_REPO FORK_BRANCH FORK_REF PLUGIN_REPO PLUGIN_BRANCH PLUGIN_REF MOE_TILESIZE_K
export RECIPE_HOME MODELS_DIR MODEL_DIR CACHE_DIR ALIAS_DIR RUNS_DIR
export CONTAINER API_PORT MAXLEN KVB MNBT LPT MTP_K CAPTURE_SIZES CG_MODE GPU_MEM_UTIL PREFIX_CACHING DSML_STOP DSML_PARAM_REPAIR
export ATS_LAYERS ATS_MTP ATS_EMBED