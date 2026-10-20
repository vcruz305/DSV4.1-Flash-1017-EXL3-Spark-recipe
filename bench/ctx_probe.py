#!/usr/bin/env python3
"""Long-context probe for one vLLM serve: retrieval, prefill and decode at depth in ONE request.

usage: ctx_probe.py <base_url> <prompt_tokens> <out_jsonl> [needle_depth=0.5] [decode_tokens=384]

Builds varied log-like filler (unique ids/numbers on every line, so it does not route like a
repeated sentence), hides a passphrase at the given depth, ends with a plain-text question,
and sends one greedy streamed /v1/completions request with ignore_eos. The first line carries a
per-run nonce, so with prefix caching on (the default) a repeated run still measures a full
prefill, not a cache hit. Prints and appends one JSON line:
prompt_tokens, ttft_s, prefill_tok_s, decode_tok_s (tokens after the first / time after the
first), found (passphrase in the first 64 generated tokens), and the head of the answer.
Plain-text prompts only (no chat template).
"""
import json
import os
import random
import sys
import time
import urllib.request

base, target, out = sys.argv[1].rstrip("/"), int(sys.argv[2]), sys.argv[3]
depth = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
ndec = int(sys.argv[5]) if len(sys.argv) > 5 else 384
MODEL = "dsv41"


def post(path, body, timeout):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ntok(text):
    return post("/tokenize", {"model": MODEL, "prompt": text}, 300)["count"]


rng = random.Random(target)
SVC = ["auth", "billing", "search", "ingest", "render", "queue", "cache", "gateway", "mailer", "audit"]
LVL = ["INFO", "WARN", "DEBUG", "ERROR"]
VERB = ["accepted", "retried", "flushed", "rejected", "compacted", "rotated", "migrated", "throttled"]


def line(i):
    return (f"2026-03-{1 + i % 28:02d}T{i % 24:02d}:{(i * 7) % 60:02d}:{(i * 13) % 60:02d}Z "
            f"{rng.choice(LVL)} {rng.choice(SVC)}-{rng.randrange(1, 64)} req={rng.randrange(16**8):08x} "
            f"{rng.choice(VERB)} {rng.randrange(1, 5000)} items in {rng.randrange(1, 900)} ms "
            f"(shard {rng.randrange(0, 256)}, user {rng.randrange(10**5, 10**6)})\n")


phrase = f"violet-harbor-{rng.randrange(1000, 9999)}-quartz"
needle = f"NOTE: the secret passphrase for this log is {phrase}. Remember it.\n"
question = ("\nEnd of log.\nQuestion: what is the secret passphrase mentioned in the NOTE line of the "
            "log above? Answer with the passphrase.\nAnswer: The secret passphrase is")

# Size: measure tokens per line on a sample, then fill, then trim with one exact count.
sample = "".join(line(i) for i in range(200))
per_line = ntok(sample) / 200.0
n = max(10, int((target - ntok(needle + question)) / per_line))
lines = [line(i) for i in range(n)]
lines.insert(int(n * depth), needle)
nonce = f"log export {os.urandom(8).hex()}\n"  # differs every run: no cached prefix to reuse
lines.insert(0, nonce)
text = "".join(lines) + question
count = ntok(text)
while count > target and len(lines) > 20:
    drop = max(1, int((count - target) / per_line))
    for _ in range(drop):
        lines.pop(-1 if lines[-1] != needle else -2)  # trim filler from the end, keep the needle
    text = "".join(lines) + question
    count = ntok(text)

body = {"model": MODEL, "prompt": text, "max_tokens": ndec, "temperature": 0, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True}}
req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
t0 = time.time()
t_first = None
pieces = []
usage = {}
with urllib.request.urlopen(req, timeout=7200) as r:
    for raw in r:
        raw = raw.strip()
        if not raw.startswith(b"data:"):
            continue
        data = raw[5:].strip()
        if data == b"[DONE]":
            break
        ev = json.loads(data)
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            if ch.get("text"):
                if t_first is None:
                    t_first = time.time()
                pieces.append(ch["text"])
t_end = time.time()
gen = usage.get("completion_tokens", 0)
ptoks = usage.get("prompt_tokens", count)
answer = "".join(pieces)
ttft = (t_first or t_end) - t0
rec = {
    "target": target, "prompt_tokens": ptoks, "needle_depth": depth, "ttft_s": round(ttft, 2),
    "prefill_tok_s": round(ptoks / ttft, 1) if ttft > 0 else None,
    "completion_tokens": gen,
    "decode_tok_s": round((gen - 1) / (t_end - t_first), 2) if t_first and gen > 1 and t_end > t_first else None,
    "found": phrase in "".join(answer.split())[:400] or phrase in answer[:400],
    "answer_head": answer[:120],
}
print(json.dumps(rec))
with open(out, "a") as f:
    f.write(json.dumps(rec) + "\n")
