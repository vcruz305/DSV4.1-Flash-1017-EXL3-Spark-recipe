#!/usr/bin/env python3
"""Parallel-request decode: N distinct greedy requests at once, per-request and total tok/s.

  python3 bench/conc_decode.py --conc 1,2,4 --tokens 256 --out conc.jsonl

Each request gets its own prompt (prose, code, structured, mixed, with a per-run nonce so
the prefix cache cannot share them), streamed via /v1/completions with ignore_eos. Per level:
TTFT and decode tok/s per request (tokens after the first / time after the first), and the
total generated tokens over the wall time of the whole level. With MAX_NUM_SEQS=1 (default)
the requests queue, so total tok/s stays at the single-request rate."""
import argparse, json, os, threading, time, urllib.request

PROMPTS = [
    "Write a detailed essay about the history of lighthouses and the people who kept them.",
    "Write a Python module that implements a thread-safe LRU cache with TTL expiry, with tests.",
    "Produce a JSON array of 40 fictional books, each with title, author, year, genre and a summary.",
    "Explain step by step how TCP congestion control works, then compare Reno, CUBIC and BBR.",
    "Write a Rust command-line tool that counts words per file in a directory tree, in full.",
    "Write a short story set on a night train crossing the Alps in winter.",
    "Write a SQL schema for a library system and twenty example queries with explanations.",
    "Describe how a compiler turns source code into machine code, phase by phase, with examples.",
]

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
ap.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "dsv41"))
ap.add_argument("--conc", default="1,2,4")
ap.add_argument("--tokens", type=int, default=256)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def one(i, nonce, res):
    body = {"model": a.model, "prompt": f"[run {nonce}-{i}]\n{PROMPTS[i % len(PROMPTS)]}\n\n",
            "max_tokens": a.tokens, "temperature": 0, "ignore_eos": True, "stream": True}
    req = urllib.request.Request(a.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); t1 = None; n = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                continue
            if json.loads(line[6:])["choices"][0].get("text"):
                n += 1
                t1 = t1 or time.time()
    t2 = time.time()
    res[i] = {"ttft_s": round(t1 - t0, 2), "chunks": n, "decode_tps": round((a.tokens - 1) / max(t2 - t1, 1e-9), 2),
              "wall_s": round(t2 - t0, 2)}


for c in [int(x) for x in a.conc.split(",")]:
    res = [None] * c
    nonce = f"{time.time():.0f}"
    t0 = time.time()
    th = [threading.Thread(target=one, args=(i, nonce, res)) for i in range(c)]
    [t.start() for t in th]; [t.join() for t in th]
    wall = time.time() - t0
    row = {"conc": c, "tokens": a.tokens, "wall_s": round(wall, 2), "total_tps": round(c * a.tokens / wall, 2),
           "per_request": res}
    print(json.dumps(row), flush=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(row) + "\n")
