#!/usr/bin/env python3
"""Phase 7 robust bench: rank configs on ms/step (kernel speed) and tokens/step
(acceptance) separately, over >= 3 unique prompts per class.

Per request we read the engine's own spec-decode counters (vllm:spec_decode_*)
before/after, so:
  steps        = d(num_drafts)                      # one draft = one verify step
  tokens/step  = 1 + d(accepted)/d(drafts)
  ms/step      = decode_wall_ms / steps             # kernel+memory speed
  tok/s        = (completion_tokens-1)/decode_wall  # what the user sees

Usage: bench7.py <label> [--reps 1] [--tokens 512]
Writes JSONL to stdout and a summary line per class.
"""
import json
import os
import re
import statistics
import sys
import time
import urllib.request

_PORT = os.environ.get("API_PORT", "8000")
URL = f"http://127.0.0.1:{_PORT}/v1/completions"
METRICS = f"http://127.0.0.1:{_PORT}/metrics"

P = {
    "prose": [
        "The history of the printing press begins in the fifteenth century, when a goldsmith in Mainz "
        "began casting individual metal letters and arranging them in a wooden frame. ",
        "For most of the nineteenth century the river carried barges of coal down to the mills, and the "
        "towpaths were kept clear by an act of parliament that nobody has ever formally repealed. ",
        "She had spent the better part of the afternoon arguing with the parish council about the "
        "hedgerow, and by the time the light failed she had convinced exactly one of them. ",
    ],
    "code": [
        "def merge_intervals(intervals):\n    \"\"\"Merge overlapping intervals and return them sorted.\"\"\"\n",
        "class LRUCache:\n    def __init__(self, capacity):\n        self.cap = capacity\n        self.d = {}\n",
        "async def fetch_all(urls, session, limit=8):\n    sem = asyncio.Semaphore(limit)\n",
    ],
    "structured": [
        '{"order_id": "A-1932", "customer": {"name": "Ada Lovelace", "tier": "gold"}, '
        '"items": [{"sku": "X-1", "qty": 2, "price": 19.5}, {"sku": "Y-7", "qty": 1, "price": 4.0}], '
        '"shipping": {"country": "GB", "postcode": "SW1A 1AA"}, "total": 43.0}\n'
        "# Validate this payload and list every field that violates the schema: ",
        '{"host": "spark-01", "role": "worker", "labels": {"zone": "eu-west-1a", "pool": "gpu"}, '
        '"alloc": {"cpu": 8, "mem_gib": 64, "gpus": [{"idx": 0, "type": "GB10"}]}}\n'
        "# Rewrite this record as a flat table row and name each column: ",
        "| layer | moe_ms | ats | resident |\n|---|---:|---|---|\n"
        "| 38 | 27.3 | yes | no |\n# Continue this table for layers 39 and 40, keeping the units: ",
    ],
}

KEYS = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_accepted_tokens",
        "vllm:spec_decode_num_draft_tokens")


def scrape():
    out = {}
    try:
        with urllib.request.urlopen(METRICS, timeout=30) as r:
            for line in r.read().decode("utf-8", "replace").splitlines():
                m = re.match(r"^(vllm:spec_decode_num_\w+?)_total\{[^}]*\}\s+(\S+)$", line)
                if m:
                    out[m.group(1)] = float(m.group(2))
    except Exception as exc:  # noqa: BLE001
        out["_err"] = repr(exc)
    return out


def gen(prompt, max_tokens):
    body = {"model": os.environ.get("SERVED_MODEL_NAME", "dsv41"), "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    ntok = 0
    usage = None
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
                if ttft is None:
                    ttft = time.time() - t0
                ntok += 1
    total = time.time() - t0
    ct = (usage or {}).get("completion_tokens") or ntok
    dec = total - (ttft or 0.0)
    return ct, dec


def main():
    lab = sys.argv[1] if len(sys.argv) > 1 else "cfg"
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 1
    ntok = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 512
    rows = []
    for name, plist in P.items():
        for rep in range(reps):
            for i, p in enumerate(plist):
                b = scrape()
                ct, dec = gen(p, ntok)
                a = scrape()
                d = {k: a.get(k, 0.0) - b.get(k, 0.0) for k in KEYS}
                steps = d[KEYS[0]]
                row = {"label": lab, "cls": name, "prompt": i, "rep": rep,
                       "completion_tokens": ct, "decode_s": round(dec, 3),
                       "tok_s": round((ct - 1) / dec, 2) if dec > 0 else None,
                       "steps": steps,
                       "ms_step": round(dec * 1000.0 / steps, 1) if steps else None,
                       "tok_step": round(1.0 + d[KEYS[1]] / steps, 3) if steps else None,
                       "draft_tok_step": round(d[KEYS[2]] / steps, 3) if steps else None}
                rows.append(row)
                print(json.dumps(row), flush=True)
    summ = {}
    for name in P:
        rs = [r for r in rows if r["cls"] == name]
        summ[name] = {
            "tok_s_med": round(statistics.median([r["tok_s"] for r in rs]), 2),
            "ms_step_mean": round(statistics.mean([r["ms_step"] for r in rs]), 1),
            "ms_step_med": round(statistics.median([r["ms_step"] for r in rs]), 1),
            "tok_step_mean": round(statistics.mean([r["tok_step"] for r in rs]), 3),
        }
    mean_tps = round(statistics.mean([summ[n]["tok_s_med"] for n in P]), 2)
    mean_ms = round(statistics.mean([summ[n]["ms_step_mean"] for n in P]), 1)
    mean_ts = round(statistics.mean([summ[n]["tok_step_mean"] for n in P]), 3)
    print(json.dumps({"label": lab, "SUMMARY": summ, "mean_tok_s": mean_tps,
                      "mean_ms_step": mean_ms, "mean_tok_step": mean_ts}), flush=True)


if __name__ == "__main__":
    main()