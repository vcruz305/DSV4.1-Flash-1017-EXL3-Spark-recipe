#!/bin/bash
# Get the base engine image ($BASE_IMAGE) without the multi-hour source build.
#
#   bash scripts/fetch_base.sh        (build_image.sh runs this when $BASE_IMAGE is missing)
#
# 1. Downloads the redistributable copy of the base from Hugging Face ($BASE_HF_REPO at the
#    pinned $BASE_HF_REV) and checks its sha256. That copy is the source build of base/ with
#    NVIDIA CUTLASS DSL removed (its licence allows redistributing only its Python source),
#    flattened to one layer.
# 2. Loads it and checks it is that exact image: the digest of its one layer and its label set
#    (the labels of the base the recipe was measured on).
# 3. Builds $BASE_IMAGE from it with base/Dockerfile.cutlass-dsl, which installs CUTLASS DSL
#    4.7.1 from PyPI, hash-locked by base/cutlass-dsl.lock.
#
# Needs zstd, host python3 with huggingface_hub, and free disk for the 8.8 GB archive plus the
# loaded image (see README "Base image"). The archive is deleted after a good load unless
# BASE_KEEP_ARCHIVE=1.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$HERE/env.sh"

for t in zstd sha256sum python3; do
  command -v "$t" >/dev/null || { echo "missing: $t (Ubuntu: sudo apt-get install -y zstd)" >&2; exit 1; }
done

layers() { sudo -n docker image inspect "$1" --format '{{json .RootFS.Layers}}' 2>/dev/null || true; }
labels_sha() {
  sudo -n docker image inspect "$1" --format '{{json .Config.Labels}}' | python3 -c '
import hashlib, json, sys
d = json.load(sys.stdin)
d.pop("local.dsv41.redist", None)  # the one label the redistributable copy adds
print(hashlib.sha256("".join(f"{k}={v}\n" for k, v in sorted(d.items())).encode()).hexdigest())'
}
WANT_LAYERS="[\"$BASE_REDIST_LAYER\"]"

if [ "$(layers "$BASE_REDIST_IMAGE")" != "$WANT_LAYERS" ]; then
  f="$BASE_DL_DIR/$BASE_ARCHIVE"
  mkdir -p "$BASE_DL_DIR"
  if [ -f "$f" ] && echo "$BASE_ARCHIVE_SHA256  $f" | sha256sum -c --status; then
    echo "archive present and verified: $f"
  else
    python3 -c "import huggingface_hub" 2>/dev/null || {
      echo "host python3 lacks huggingface_hub; see README Quick start" >&2; exit 1; }
    echo "downloading $BASE_HF_REPO@${BASE_HF_REV:0:12} $BASE_ARCHIVE (8.8 GB) to $BASE_DL_DIR"
    python3 - "$BASE_HF_REPO" "$BASE_HF_REV" "$BASE_ARCHIVE" "$BASE_DL_DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo, rev, name, out = sys.argv[1:]
hf_hub_download(repo_id=repo, filename=name, revision=rev, local_dir=out)
PY
    echo "checking sha256"
    echo "$BASE_ARCHIVE_SHA256  $f" | sha256sum -c --status || {
      echo "sha256 mismatch for $f (expected $BASE_ARCHIVE_SHA256); delete it and rerun" >&2; exit 1; }
  fi
  echo "loading $f"
  zstd -dc "$f" | sudo -n docker load
  got="$(layers "$BASE_REDIST_IMAGE")"
  if [ "$got" != "$WANT_LAYERS" ]; then
    echo "$BASE_REDIST_IMAGE has layers $got, expected $WANT_LAYERS" >&2; exit 1
  fi
  [ "${BASE_KEEP_ARCHIVE:-0}" = 1 ] || rm -f "$f"
fi

got="$(labels_sha "$BASE_REDIST_IMAGE")"
if [ "$got" != "$BASE_LABELS_SHA256" ]; then
  echo "$BASE_REDIST_IMAGE label set $got, expected $BASE_LABELS_SHA256" >&2; exit 1
fi
echo "redistributable base ok: $BASE_REDIST_IMAGE, layer ${BASE_REDIST_LAYER:7:12}, labels ${BASE_LABELS_SHA256:0:12}"

echo "installing NVIDIA CUTLASS DSL 4.7.1 from PyPI (base/cutlass-dsl.lock) -> $BASE_IMAGE"
sudo -n docker build \
  -f "$RECIPE_HOME/base/Dockerfile.cutlass-dsl" \
  --build-arg "REDIST_IMAGE=$BASE_REDIST_IMAGE" \
  -t "$BASE_IMAGE" \
  "$RECIPE_HOME/base"
echo "base ready: $BASE_IMAGE"
