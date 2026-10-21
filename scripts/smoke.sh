#!/bin/bash
# Smoke test: one plain completion. The serve is only "up" if a fresh request
# returns coherent text and finish_reason=stop.
set -euo pipefail

API_PORT="${API_PORT:-8000}"
MODEL_ID="${SERVED_MODEL_NAME:-dsv41}"
BASE="http://127.0.0.1:${API_PORT}/v1"

# `set -e` plus a failing command substitution exits silently; say what failed.
if ! RESP=$(curl -s -m 300 "$BASE/completions" -H 'Content-Type: application/json' -d '{
  "model": "'"$MODEL_ID"'",
  "prompt": "The capital of France is",
  "max_tokens": 16,
  "temperature": 0,
  "stream": false
}'); then
  echo "SMOKE FAIL: no response from $BASE/completions (curl rc=$?)" >&2
  exit 1
fi

python3 - "$RESP" <<'PY'
import json
import sys

try:
    r = json.loads(sys.argv[1])
except Exception as exc:  # noqa: BLE001
    print("SMOKE FAIL: unparseable response:", exc)
    sys.exit(1)

try:
    ch = r["choices"][0]
    text = (ch.get("text") or "").strip()
    finish = ch.get("finish_reason")
except Exception as exc:  # noqa: BLE001
    print("SMOKE FAIL: no completion in response:", exc)
    print(json.dumps(r)[:400])
    sys.exit(1)

print(f"finish_reason={finish!r} text={text[:120]!r}")
if finish in ("stop", "length") and "Paris" in text:
    print("SMOKE PASS")
    sys.exit(0)
print("SMOKE FAIL: expected 'Paris' in the reply")
sys.exit(1)
PY