#!/usr/bin/env python3
"""Decode speed on held-out Hermes requests (same session pick/trim as replay_hermes.py).

  python3 bench/hermes_decode.py --shard 1/2 --limit 6 --tokens 384 --out x.jsonl

Each request is sent twice, greedy, streamed: pass 1 fills the prefix cache, pass 2 is
timed (TTFT, then decode tok/s over the streamed tokens). Greedy keeps the generated
tokens identical across cold-set configurations, so pass-2 decode times compare."""
import argparse, glob, json, os, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.expanduser("~/.hermes/sessions"))
ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
ap.add_argument("--model", default="dsv41")
ap.add_argument("--max-chars", type=int, default=240000)
ap.add_argument("--tokens", type=int, default=384)
ap.add_argument("--shard", default="1/2")
ap.add_argument("--limit", type=int, default=6)
ap.add_argument("--out", required=True)
a = ap.parse_args()


def body_of(f):
    x = json.load(open(f))
    b = x.get("request", {}).get("body") or x.get("request") or x
    return x.get("session_id") or f, json.loads(b) if isinstance(b, str) else b


def trim(msgs, cap):
    head = [m for m in msgs[:1] if m.get("role") == "system"]
    tail, n = [], len(json.dumps(head))
    for m in reversed(msgs[len(head):]):
        s = len(json.dumps(m))
        if n + s > cap and tail:
            break
        tail.append(m)
        n += s
    tail.reverse()
    while tail and tail[0].get("role") in ("tool", "assistant"):
        tail.pop(0)
    return head + tail


best = {}
for f in sorted(glob.glob(os.path.join(a.dir, "request_dump_*.json"))):
    try:
        sid, b = body_of(f)
    except Exception:
        continue
    if b.get("messages"):
        best[sid] = b
k, n = map(int, a.shard.split("/"))
items = [x for i, x in enumerate(best.items()) if i % n == k][: a.limit or None]


def stream(p):
    req = urllib.request.Request(a.url, data=json.dumps(p).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                if dl.get("content") or dl.get("reasoning_content") or dl.get("reasoning") or dl.get("tool_calls"):
                    if ttft is None:
                        ttft = time.time()
                    chunks += 1
    t1 = time.time()
    gen = usage.get("completion_tokens", 0)
    dec = (gen - 1) / (t1 - ttft) if ttft and gen > 1 and t1 > ttft else None
    return {"ttft": round((ttft or t1) - t0, 2), "gen": gen, "decode_tps": dec and round(dec, 2),
            "prompt": usage.get("prompt_tokens"), "s": round(t1 - t0, 2)}


with open(a.out, "a") as log:
    for sid, b in items:
        p = {"model": a.model, "max_tokens": a.tokens, "temperature": 0.0, "stream": True,
             "stream_options": {"include_usage": True}, "ignore_eos": True,
             "messages": trim(b["messages"], a.max_chars - len(json.dumps(b.get("tools") or [])))}
        if b.get("tools"):
            p["tools"] = b["tools"]
        for ps in (1, 2):
            rec = {"sid": sid, "pass": ps, **stream(p)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
