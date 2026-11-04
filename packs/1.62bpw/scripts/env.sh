#!/bin/bash
# Pins and knobs of the 1.62 bpw SAGE EXL3 pack. Source this; do not execute it.
#
# The shared engine pins (base/engine image, runtime forks, host paths, and the
# serve knobs both packs measured the same) come from the repository root:
# scripts/env.sh. This file adds only what belongs to this pack.
#
# Every value here is the measured configuration. Changing one without
# re-measuring means the numbers in this pack's README no longer describe what
# you serve.

REPO_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=../../../scripts/env.sh
. "$REPO_HOME/scripts/env.sh"

# ---- model (this pack) ----------------------------------------------------
MODEL_REPO="vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw"
# Hub revision to download. The repo's history has been replaced more than once
# (each time a single "Initial commit" with the same files), so a commit id can
# stop resolving. the SHA256SUMS file next to this file is the authority:
# download_and_verify.sh falls back to main when this revision is gone, and
# the sha256 check decides. See this pack's README "Pack revision".
MODEL_REV="${MODEL_REV:-c852150e06f5e5ee8c4c14f9cd1a9afc6bb21960}"
MODEL_NAME="dsv41-1.62bpw-v3"          # directory name under $MODELS_DIR
MODEL_DIR="$MODELS_DIR/$MODEL_NAME"
# decode graph sizes: the measured single-request configuration. This pack
# serves one request at a time and text only (serve_inner.sh passes
# --max-num-seqs 1 and --language-model-only; every number in this pack's
# README was measured that way).
CAPTURE_SIZES="${CAPTURE_SIZES:-1,2,3,4,5,6,7}"

export MODEL_REPO MODEL_REV MODEL_NAME MODEL_DIR CAPTURE_SIZES
