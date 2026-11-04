#!/bin/bash
# Pins and knobs of the 1.55 bpw SAGE REXL3 pack. Source this; do not execute it.
#
# The shared engine pins (base/engine image, runtime forks, host paths, and the
# serve knobs both packs measured the same) come from the repository root:
# scripts/env.sh. This file adds only what belongs to this pack.
#
# Every value here is the measured configuration. Changing one without
# re-measuring means the numbers in this pack's README and
# TESTING-AND-RESULTS.md no longer describe what you serve.

REPO_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# shellcheck source=../../../scripts/env.sh
. "$REPO_HOME/scripts/env.sh"

# ---- model ----------------------------------------------------------------
# Default MODEL_NAME is the 1.62 pack's directory (the two packs share one
# models tree and one alias side file). The REXL3 overlay built by
# build_rexl3_overlay.sh is served with MODEL_NAME=rexl3-vic-overlay.
MODEL_NAME="${MODEL_NAME:-dsv41-1.62bpw-v3}"          # directory name under $MODELS_DIR
MODEL_DIR="$MODELS_DIR/$MODEL_NAME"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"   # requests decoded together; >1 is opt-in (README "Parallel requests")
# decode graphs for every batch size up to (MTP_K+1) tokens x MAX_NUM_SEQS (1..7 at the defaults)
CAPTURE_SIZES="${CAPTURE_SIZES:-$(seq -s, 1 $(( MAX_NUM_SEQS > 1 ? (MTP_K + 1) * MAX_NUM_SEQS : 7 )))}"
VISION_IMAGES="${VISION_IMAGES:-1}"       # N = accept up to N images per prompt (default 1; tower served from the pack, ~0.7 GiB); 0 = text only (--language-model-only)

export MODEL_NAME MODEL_DIR MAX_NUM_SEQS CAPTURE_SIZES VISION_IMAGES
