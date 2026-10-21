#!/usr/bin/env python3
"""Long-prompt ladder: TTFT + prefill tok/s per prompt length.

Plain text prompts only (repeated words) via /v1/completions, so no chat
template / special-token handling is involved. Stops at the first failure and
prints whatever the server returned.
"""
import os
import json
import sys
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8000/v1/completions"
WORD = "the quick brown fox jumps over a lazy dog while seven wizards quietly vex "


def prompt_of(target_tokens: int) -> str:
    # ~1.15 tokens/word for this text; overshoot then trim by words
    words = WORD.split()
    n = max(1, int(target_tokens / 1.15))
    out = []
    while len(out) < n:
        out.extend(words)
    return " ".join(out[:n])


def run(target: int, max_tokens: int = 8):
    body = {
        "model": os.environ.get("SERVED_MODEL_NAME", "dsv41"),
        "prompt": prompt_of(target),
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    ttft = None
    usage = None
    nchunk = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            j = json.loads(payload)
            if j.get("usage"):
                usage = j["usage"]
            if j["choices"] and j["choices"][0].get("text"):
                nchunk += 1
                if ttft is None:
                    ttft = time.time() - t0
    total = time.time() - t0
    pt = (usage or {}).get("prompt_tokens")
    ct = (usage or {}).get("completion_tokens")
    return {
        "target": target,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "ttft_s": round(ttft, 3) if ttft else None,
        "total_s": round(total, 3),
        "prefill_tok_s": round(pt / ttft, 1) if (pt and ttft) else None,
        "decode_tok_s": round((ct - 1) / (total - ttft), 2)
        if (ct and ttft and total > ttft)
        else None,
    }


def main():
    targets = [int(x) for x in sys.argv[1:]] or [200, 250, 256, 257, 260, 300, 1024, 2048, 8192, 30000]
    for t in targets:
        try:
            print(json.dumps(run(t)), flush=True)
        except Exception as exc:  # noqa: BLE001
            body = ""
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    body = exc.read().decode("utf-8", "replace")[:2000]
                except Exception:
                    pass
            print(json.dumps({"target": t, "ERROR": repr(exc), "body": body}), flush=True)
            break


if __name__ == "__main__":
    main()