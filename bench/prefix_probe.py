#!/usr/bin/env python3
"""Prefix-cache probe for one vLLM serve: what a long shared prefix costs the first time and after.

usage: prefix_probe.py <base_url> <prompt_tokens> <out_jsonl> [decode_tokens=128]

Builds one varied log-like document of about <prompt_tokens> tokens (unique ids on every line,
same generator as ctx_probe.py) with two passphrases, ALPHA at 30 % depth and BETA at 70 %, then
sends four greedy streamed /v1/completions requests (ignore_eos) in order:

  cold      document + question about ALPHA       (nothing cached: full prefill)
  new_q     the same document + question about BETA (shared prefix, different suffix)
  repeat    the cold request again, byte for byte
  next_turn cold request + its answer + a question about BETA (a multi-turn follow-up)

The document starts with a per-run nonce, so 'cold' is a full prefill even when an earlier run
left its document in the cache. Each request prints and appends one JSON line: prompt_tokens,
ttft_s, prefill_tok_s (prompt_tokens / ttft_s; on the warm rows this is an effective rate, not
the prefill kernel's), ttft_vs_cold, decode_tok_s (tokens after the first / time after the first),
found (the asked passphrase in the first answer tokens) and the answer head. Plain-text prompts
only (no chat template).
"""
import json
import os
import random
import sys
import time
import urllib.request

base, target, out = sys.argv[1].rstrip("/"), int(sys.argv[2]), sys.argv[3]
ndec = int(sys.argv[4]) if len(sys.argv) > 4 else 128
MODEL = os.environ.get("SERVED_MODEL_NAME", "dsv41")


def post(path, body, timeout):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ntok(text):
    return post("/tokenize", {"model": MODEL, "prompt": text}, 300)["count"]


rng = random.Random(target + 7)
SVC = ["auth", "billing", "search", "ingest", "render", "queue", "cache", "gateway", "mailer", "audit"]
LVL = ["INFO", "WARN", "DEBUG", "ERROR"]
VERB = ["accepted", "retried", "flushed", "rejected", "compacted", "rotated", "migrated", "throttled"]


def line(i):
    return (f"2026-03-{1 + i % 28:02d}T{i % 24:02d}:{(i * 7) % 60:02d}:{(i * 13) % 60:02d}Z "
            f"{rng.choice(LVL)} {rng.choice(SVC)}-{rng.randrange(1, 64)} req={rng.randrange(16**8):08x} "
            f"{rng.choice(VERB)} {rng.randrange(1, 5000)} items in {rng.randrange(1, 900)} ms "
            f"(shard {rng.randrange(0, 256)}, user {rng.randrange(10**5, 10**6)})\n")


phrase = {"ALPHA": f"amber-delta-{rng.randrange(1000, 9999)}-falcon",
          "BETA": f"cobalt-river-{rng.randrange(1000, 9999)}-lynx"}
needle = {k: f"NOTE {k}: the {k} passphrase for this log is {v}. Remember it.\n" for k, v in phrase.items()}


def question(k):
    return (f"\nEnd of log.\nQuestion: what is the {k} passphrase given on the NOTE {k} line of the log "
            f"above? Answer with the passphrase.\nAnswer: The {k} passphrase is")


sample = "".join(line(i) for i in range(200))
per_line = ntok(sample) / 200.0
n = max(20, int((target - ntok(needle["ALPHA"] + needle["BETA"] + question("ALPHA"))) / per_line))
lines = [line(i) for i in range(n)]
lines.insert(int(n * 0.7), needle["BETA"])
lines.insert(int(n * 0.3), needle["ALPHA"])
lines.insert(0, f"log export {os.urandom(8).hex()}\n")  # differs every run
keep = set(needle.values()) | {lines[0]}
doc = "".join(lines)
count = ntok(doc + question("ALPHA"))
while count > target and len(lines) > 40:
    drop = max(1, int((count - target) / per_line))
    i = len(lines) - 1
    while drop and i >= 0:
        if lines[i] not in keep:
            lines.pop(i)
            drop -= 1
        i -= 1
    doc = "".join(lines)
    count = ntok(doc + question("ALPHA"))


COLD = []


def run(label, prompt, asked):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": ndec, "temperature": 0, "ignore_eos": True,
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
    ptoks = usage.get("prompt_tokens", 0)
    answer = "".join(pieces)
    ttft = (t_first or t_end) - t0
    rec = {
        "request": label, "target": target, "prompt_tokens": ptoks, "ttft_s": round(ttft, 2),
        "prefill_tok_s": round(ptoks / ttft, 1) if ttft > 0 else None,
        "ttft_vs_cold": round(ttft / COLD[0], 4) if COLD else 1.0,
        "completion_tokens": gen,
        "decode_tok_s": round((gen - 1) / (t_end - t_first), 2) if t_first and gen > 1 and t_end > t_first else None,
        "asked": asked, "found": phrase[asked] in answer[:400],
        "answer_head": answer[:80],
    }
    if not COLD:
        COLD.append(ttft)
    print(json.dumps(rec), flush=True)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return answer


a1 = run("cold", doc + question("ALPHA"), "ALPHA")
run("new_q", doc + question("BETA"), "BETA")
run("repeat", doc + question("ALPHA"), "ALPHA")
follow = ("\nQuestion: and what is the BETA passphrase given on the NOTE BETA line of the log above? "
          "Answer with the passphrase.\nAnswer: The BETA passphrase is")
run("next_turn", doc + question("ALPHA") + a1 + follow, "BETA")
