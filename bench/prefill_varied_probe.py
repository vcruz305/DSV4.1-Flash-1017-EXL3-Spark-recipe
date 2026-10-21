#!/usr/bin/env python3
"""Prefill TTFT probe with VARIED text (python stdlib source), plain /v1/completions.

Usage: python3 prefill_varied_probe.py 2048,8192,16384,30000
Prints one line per length: target, prompt_tokens, TTFT s, prefill tok/s, MemAvailable min (GiB).
"""
import os
import glob, json, os, sys, time, threading, urllib.request

URL = "http://127.0.0.1:8000/v1/completions"


def corpus():
    files = sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))
    txt = []
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                t = fh.read()
        except OSError:
            continue
        if t.isascii():
            txt.append(t)
        if sum(map(len, txt)) > 2_500_000:
            break
    return "\n".join(txt)


def memavail():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1048576
    return -1.0


def run(text, target, offset):
    # ~3.3 chars/token for python source; generous, then report real prompt_tokens
    chunk = text[offset: offset + int(target * 3.3)]
    body = {"model": os.environ.get("SERVED_MODEL_NAME", "dsv41"), "prompt": chunk, "max_tokens": 4, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    lo = [memavail()]
    stop = [False]

    def watch():
        while not stop[0]:
            lo[0] = min(lo[0], memavail())
            time.sleep(0.1)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    t0 = time.time()
    ttft = None
    ptoks = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if ttft is None and d.get("choices"):
                ttft = time.time() - t0
            if d.get("usage"):
                ptoks = d["usage"].get("prompt_tokens")
    stop[0] = True
    th.join()
    return ptoks, ttft, lo[0], int(target * 3.3)


def main():
    lengths = [int(x) for x in sys.argv[1].split(",")]
    text = corpus()
    off = int.from_bytes(os.urandom(3), "big") % 200_000
    for n in lengths:
        ptoks, ttft, lo, used = run(text, n, off)
        off += used
        print(f"target={n} prompt_tokens={ptoks} ttft={ttft:.2f}s prefill={ptoks / ttft:.0f} tok/s "
              f"memavail_min={lo:.2f}GiB", flush=True)


if __name__ == "__main__":
    main()
