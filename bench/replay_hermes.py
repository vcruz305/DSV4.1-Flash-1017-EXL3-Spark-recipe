#!/usr/bin/env python3
"""Replay saved Hermes request bodies (~/.hermes/sessions/request_dump_*.json) against the
server to collect expert-routing counts from real agent traffic. No tools are executed.

  python3 bench/replay_hermes.py [--max-chars 240000] [--tokens 512] [--limit N] [--out log.jsonl]

One request per Hermes session (its largest dump under --max-chars after trimming); long
conversations keep the system message and the most recent turns. Prefix caching is on in
the server, so each session is sent once."""
import argparse, glob, json, os, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.expanduser("~/.hermes/sessions"))
ap.add_argument("--url", default=os.environ.get("URL", "http://127.0.0.1:8000/v1/chat/completions"))
ap.add_argument("--model", default=os.environ.get("SERVED_NAME", "dsv41"))
ap.add_argument("--max-chars", type=int, default=240000)
ap.add_argument("--tokens", type=int, default=512)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--shard", default="0/1", help="k/n: send only sessions i with i %% n == k")
ap.add_argument("--out", default="replay_hermes.jsonl")
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
        tail.append(m); n += s
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
    if not b.get("messages"):
        continue
    best[sid] = b  # latest dump of each session
k, n = map(int, a.shard.split("/"))
items = [x for i, x in enumerate(best.items()) if i % n == k][: a.limit or None]
print(f"{len(items)} sessions", flush=True)

def post(payload):
    req = urllib.request.Request(a.url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)

with open(a.out, "a") as log:
    for i, (sid, b) in enumerate(items):
        msgs = trim(b["messages"], a.max_chars - len(json.dumps(b.get("tools") or [])))
        p = {"model": a.model, "messages": msgs, "max_tokens": a.tokens, "temperature": 0.6}
        if b.get("tools"):
            p["tools"] = b["tools"]
        t = time.time(); err = None
        try:
            r = post(p)
        except Exception as e:
            err = f"{type(e).__name__}: {getattr(e, 'read', lambda: b'')()[:300]!r}"
            p.pop("tools", None)
            try:
                r = post(p); err += " (retried without tools)"
            except Exception as e2:
                r = None; err += f" | {type(e2).__name__}"
        u = (r or {}).get("usage", {})
        rec = {"i": i, "sid": sid, "s": round(time.time() - t, 1), "prompt": u.get("prompt_tokens"),
               "completion": u.get("completion_tokens"), "err": err}
        log.write(json.dumps(rec) + "\n"); log.flush(); print(rec, flush=True)
