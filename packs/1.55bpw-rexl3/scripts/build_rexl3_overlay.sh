#!/bin/bash
# Build the hard-link overlay that lets vLLM serve the SAGE REXL3 1.55 bpw pack:
# EXL3 head -> BF16, EXL3 non-routed linears -> block-32 FP8, index rewritten; the routed
# experts and every other shard are hard links (no extra disk beyond the two side files, ~6 GiB).
# Usage: bash packs/1.55bpw-rexl3/scripts/build_rexl3_overlay.sh [pack dir] [overlay dir]   (GPU must be free)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
SRC="${1:-$MODELS_DIR/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw}"
DST="${2:-$MODELS_DIR/rexl3-vic-overlay}"
[ -f "$SRC/model.safetensors.index.json" ] || { echo "no REXL3 pack at $SRC (download vcruz305's SAGE-REXL3 1.55bpw first)" >&2; exit 1; }
[ "$(stat -c %d "$SRC")" = "$(stat -c %d "$(dirname "$DST")")" ] || { echo "$DST must be on the same filesystem as $SRC (hard links)" >&2; exit 1; }
mkdir -p "$DST"
for f in "$SRC"/*; do b=$(basename "$f"); [ "$b" = model.safetensors.index.json ] || ln -f "$f" "$DST/$b"; done
run(){ sudo -n docker run --rm --gpus all --ipc=host -v "$SRC:/src:ro" -v "$DST:/dst" -v "$HERE/../tools:/tools:ro" \
  --entrypoint python3 "$IMAGE" "$@"; }
[ -f "$DST/head-bf16.safetensors" ] || run /tools/rexl3_head_bf16.py /src /dst/head-bf16.safetensors
[ -f "$DST/nonrouted-fp8.safetensors" ] || run /tools/rexl3_nonrouted_fp8.py /src /dst/nonrouted-fp8.safetensors
sudo -n docker run --rm -v "$DST:/dst" --entrypoint chown "$IMAGE" "$(id -u):$(id -g)" /dst/head-bf16.safetensors /dst/nonrouted-fp8.safetensors
python3 "$HERE/../tools/rexl3_overlay_index.py" "$SRC" "$DST"
echo "overlay ready: MODEL_NAME=$(basename "$DST") HF_OVERRIDES_JSON=\"\$(python3 packs/1.55bpw-rexl3/tools/rexl3_overrides.py)\" bash packs/1.55bpw-rexl3/scripts/serve.sh"
