#!/bin/bash
# Build the serving image: base engine + vllm-exl3 + exllamav3 (tile-K 16) + shim.
#
#   FORK_REF=<sha> PLUGIN_REF=<sha> bash scripts/build_image.sh
#
# If the base image is missing, scripts/fetch_base.sh fetches it from Hugging Face
# (8.8 GB; see README "Base image"). BASE_FROM_SOURCE=1 builds it from base/ instead
# (CPU only, hours).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$HERE/env.sh"


if ! sudo -n docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  if [ "${BASE_FROM_SOURCE:-0}" = 1 ]; then
    echo "building $BASE_IMAGE from base/ (CPU only, hours; see base/README.md)"
    sudo -n docker build --platform linux/arm64 -f "$RECIPE_HOME/base/docker/Dockerfile" \
      -t "$BASE_IMAGE" "$RECIPE_HOME/base"
  else
    bash "$HERE/fetch_base.sh"
  fi
fi

echo "base   : $BASE_IMAGE"
echo "plugin : $PLUGIN_REF ($PLUGIN_BRANCH)"
echo "fork   : $FORK_REF ($FORK_BRANCH), MOE_TILESIZE_K=$MOE_TILESIZE_K"
echo "image  : $IMAGE"

sudo -n docker build \
  -f "$RECIPE_HOME/docker/Dockerfile" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "PLUGIN_REPO=$PLUGIN_REPO" \
  --build-arg "PLUGIN_REF=$PLUGIN_REF" \
  --build-arg "FORK_REPO=$FORK_REPO" \
  --build-arg "FORK_REF=$FORK_REF" \
  --build-arg "MOE_TILESIZE_K=$MOE_TILESIZE_K" \
  -t "$IMAGE" \
  "$RECIPE_HOME"

sudo -n docker image inspect "$IMAGE" --format 'built {{.Id}} {{.Created}}'