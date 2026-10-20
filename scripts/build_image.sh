#!/bin/bash
# Build the serving image: base engine + vllm-exl3 + exllamav3 (tile-K 16) + shim.
#
#   FORK_REF=<sha> PLUGIN_REF=<sha> bash scripts/build_image.sh
#
# The base image is not built here. If it is missing, this prints the command
# that builds it from the 4x recipe.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$HERE/env.sh"


if ! sudo -n docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  cat >&2 <<EOF
Base image $BASE_IMAGE is not present.

Build it once from the files in base/ (CPU only, several hours; see base/README.md):

  sudo docker build --platform linux/arm64 -f "$RECIPE_HOME/base/docker/Dockerfile" -t $BASE_IMAGE "$RECIPE_HOME/base"

That image is Christopher Owen's spark-ds41f r6w (upstreams pinned at 2dc95329) with the
patches listed in base/NOTICE.
EOF
  exit 1
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