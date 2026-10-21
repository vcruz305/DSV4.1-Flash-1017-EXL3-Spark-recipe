#!/usr/bin/env python3
"""Does a prefix-cache hit change decode? One long prompt sent three times: cold, then cached twice.

usage: prefix_accept.py <prompt_tokens> <decode_tokens> <out_jsonl> [base_url=http://127.0.0.1:8000]

Builds varied log-like text (a per-run nonce on the first line, a passphrase halfway, a question
that asks for the passphrase and a summary), then sends the same greedy streamed
/v1/completions request (ignore_eos) three times. For each request it reads the server's
/metrics before and after and prints one JSON line: ttft_s, decode_tok_s, tok_per_step
(1 + accepted draft tokens / drafts), ms_per_step (decode time / drafts), cache_hit (prefix-cache
hits / queries over the request), found, and the head of the answer. Run it on an idle server:
the counters are server-wide.
"""
import os
import json, os, random, sys, time, urllib.request
target = int(sys.argv[1]); ndec = int(sys.argv[2]); out = sys.argv[3]
base = (sys.argv[4] if len(sys.argv) > 4 else "http://127.0.0.1:8000").rstrip("/")
def post(path, body, t):
    r = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=t) as f: return json.loads(f.read())
def metrics():
    with urllib.request.urlopen(base + "/metrics", timeout=10) as f: txt = f.read().decode()
    v = {}
    for l in txt.splitlines():
        for k in ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_drafts_total", "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"):
            if l.startswith(k): v[k] = v.get(k, 0) + float(l.split()[-1])
    return v
rng = random.Random(target + 11)
SVC = ["auth", "billing", "search", "ingest", "render", "queue", "cache", "gateway", "mailer", "audit"]
def line(i):
    return (f"2026-03-{1 + i % 28:02d}T{i % 24:02d}:{(i * 7) % 60:02d}:{(i * 13) % 60:02d}Z "
            f"{rng.choice(['INFO','WARN','DEBUG','ERROR'])} {rng.choice(SVC)}-{rng.randrange(1, 64)} req={rng.randrange(16**8):08x} "
            f"{rng.choice(['accepted','retried','flushed','rejected'])} {rng.randrange(1, 5000)} items in {rng.randrange(1, 900)} ms "
            f"(shard {rng.randrange(0, 256)}, user {rng.randrange(10**5, 10**6)})\n")
ph = f"silver-meadow-{rng.randrange(1000, 9999)}-heron"
q = "\nEnd of log.\nQuestion: what is the secret passphrase on the NOTE line above? Answer with the passphrase, then summarize the log in a few paragraphs.\nAnswer: The secret passphrase is"
per = post("/tokenize", {"model": os.environ.get("SERVED_MODEL_NAME", "dsv41"), "prompt": "".join(line(i) for i in range(200))}, 300)["count"] / 200
n = int((target - 200) / per)
lines = [f"log export {os.urandom(8).hex()}\n"] + [line(i) for i in range(n)]
lines.insert(n // 2, f"NOTE: the secret passphrase for this log is {ph}. Remember it.\n")
text = "".join(lines) + q
while post("/tokenize", {"model": os.environ.get("SERVED_MODEL_NAME", "dsv41"), "prompt": text}, 300)["count"] > target:
    del lines[-40:-1]; text = "".join(lines) + q
for label in ("cold", "cached_1", "cached_2"):
    m0 = metrics(); t0 = time.time(); tf = None; parts = []; usage = {}
    body = {"model": os.environ.get("SERVED_MODEL_NAME", "dsv41"), "prompt": text, "max_tokens": ndec, "temperature": 0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    r = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=7200) as f:
        for raw in f:
            raw = raw.strip()
            if not raw.startswith(b"data:") or raw[5:].strip() == b"[DONE]": continue
            ev = json.loads(raw[5:])
            if ev.get("usage"): usage = ev["usage"]
            for ch in ev.get("choices") or []:
                if ch.get("text"):
                    tf = tf or time.time(); parts.append(ch["text"])
    te = time.time(); m1 = metrics(); d = {k: m1[k] - m0.get(k, 0) for k in m1}
    acc, dr = d["vllm:spec_decode_num_accepted_tokens_total"], d["vllm:spec_decode_num_drafts_total"]
    hit = d["vllm:prefix_cache_hits_total"] / d["vllm:prefix_cache_queries_total"] if d["vllm:prefix_cache_queries_total"] else None
    gen = usage.get("completion_tokens", 0); a = "".join(parts)
    rec = {"request": label, "prompt_tokens": usage.get("prompt_tokens"), "ttft_s": round(tf - t0, 2),
           "decode_tok_s": round((gen - 1) / (te - tf), 2), "tok_per_step": round(1 + acc / dr, 2) if dr else None,
           "ms_per_step": round((te - tf) * 1000 / dr, 1) if dr else None, "cache_hit": round(hit, 4) if hit is not None else None,
           "found": ph in a[:200], "head": a[:70]}
    print(json.dumps(rec, ensure_ascii=True), flush=True)
    open(out, "a").write(json.dumps(rec) + "\n")
