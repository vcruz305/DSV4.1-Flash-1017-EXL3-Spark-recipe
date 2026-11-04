#!/bin/bash
# Download the pack at the pinned revision with hf_xet, then verify every file
# against packs/1.62bpw/scripts/SHA256SUMS (the Hub's own LFS sha256 for the 17 shards and the
# index, plus sha256 of the three small files).
#
#   HF_TOKEN=hf_... bash packs/1.62bpw/scripts/download_and_verify.sh
#
# The repo is public but gated: the token must belong to an account that has
# accepted the model's terms. The token is never printed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$HERE/env.sh"

: "${HF_TOKEN:?Set HF_TOKEN to a token with access to $MODEL_REPO}"

# Runs on the host, not in the image (the image is built after the download).
if ! python3 -c "import huggingface_hub, hf_xet" 2>/dev/null; then
  echo "host python3 lacks huggingface_hub and/or hf_xet. Install them first:" >&2
  echo "  python3 -m pip install --user -U 'huggingface_hub[hf_xet]'" >&2
  exit 1
fi

export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_ENABLE_HF_TRANSFER=0   # xet path, not the legacy transfer

mkdir -p "$MODEL_DIR"

# Re-running this on a complete directory does NOT re-download the 326.6 GB:
# huggingface_hub 2.x keeps a per-file record in $MODEL_DIR/.cache/huggingface/
# download/*.metadata. With no record (e.g. a copied/hardlinked tree) it falls
# back to hashing each local LFS file against its sha256 etag and skips the
# transfer when they match -- a full 326.6 GB read pass, ~4 min, no network.
# Only the three small non-LFS files (config.json, tokenizer.json,
# tokenizer_config.json) are re-fetched. Measured, huggingface_hub 2.2.0.

# The pack is 326.6 GB (two shards are ~101.5 GB each). Refuse to start a download
# that cannot finish: an ENOSPC halfway leaves a directory that looks downloaded.
# Bytes already in $MODEL_DIR (a resumed download) count toward the total.
PACK_BYTES=326590218983
have_bytes=$(du -sb "$MODEL_DIR" | cut -f1)
need_kib=$(( (PACK_BYTES - have_bytes) / 1024 + 4 * 1024 * 1024 ))   # + 4 GiB slack
avail_kib=$(df -k --output=avail "$MODEL_DIR" | tail -1 | tr -dc '0-9')
if [ "${avail_kib:-0}" -lt "$need_kib" ]; then
  echo "not enough space on $(df --output=target "$MODEL_DIR" | tail -1): need $((need_kib / 1048576)) GiB more, have $((avail_kib / 1048576)) GiB" >&2
  exit 1
fi

python3 - "$MODEL_REPO" "$MODEL_REV" "$MODEL_DIR" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, RevisionNotFoundError

repo, rev, dest = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi(token=os.environ["HF_TOKEN"])
def _gone(exc: Exception) -> bool:
    # A replaced history is a 404 on the revision. Anything else (auth, gating,
    # network) must surface, not silently switch to another revision.
    if isinstance(exc, RevisionNotFoundError):
        return True
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None) == 404 and "Repository Not Found" not in str(exc)

try:
    api.model_info(repo, revision=rev)
except HfHubHTTPError as exc:
    if not _gone(exc):
        raise
    main = api.model_info(repo).sha
    print(f"WARNING: pinned revision {rev[:12]} no longer exists on the Hub "
          f"(history replaced); downloading main = {main[:12]} instead. "
          f"packs/1.62bpw/scripts/SHA256SUMS decides whether it is the measured pack.")
    rev = main
print(f"downloading {repo}@{rev[:12]} -> {dest}")
snapshot_download(
    repo_id=repo,
    revision=rev,
    local_dir=dest,
    token=os.environ["HF_TOKEN"],
    max_workers=2,          # measured: >2 workers grows swap on a 128 GB UMA part
    etag_timeout=60,
)
PY

echo "verifying against packs/1.62bpw/scripts/SHA256SUMS"
cd "$MODEL_DIR"
sha256sum -c "$HERE/SHA256SUMS"
echo "VERIFY PASS: every file matches the pinned manifest"
echo "total bytes on disk: $(du -sb "$MODEL_DIR" | cut -f1)"