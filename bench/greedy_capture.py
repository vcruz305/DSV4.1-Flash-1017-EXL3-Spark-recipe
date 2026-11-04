#!/usr/bin/env python3
"""Greedy (temperature 0) completions for a fixed prompt set, saved as JSON.

Note: with DSpark speculative decoding + async scheduling, two greedy runs on the same
server already differ on some prompts (measured), so compare by similarity, not equality.
  python3 bench/greedy_capture.py out.json [--tokens 200] [--url http://127.0.0.1:8000]
"""
import argparse, json, urllib.request

PROMPTS = [
    "Write a Python function that merges overlapping intervals, with a docstring and tests.",
    "Explain how a CPU cache hierarchy works, from registers to DRAM, in plain language.",
    "List the planets of the solar system as a JSON array of objects with name and order.",
    "Translate to French and then summarize in one sentence: The committee postponed the vote "
    "because two members were travelling and the quorum rules require seven present.",
    "def quicksort(xs):\n    \"\"\"Sort a list of ints.\"\"\"\n",
    "A train leaves at 09:40 and travels 245 km at 70 km/h. When does it arrive? Show steps.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    a = ap.parse_args()
    model = json.load(urllib.request.urlopen(a.url + "/v1/models"))["data"][0]["id"]
    res = []
    for p in PROMPTS:
        body = {"model": model, "messages": [{"role": "user", "content": p}],
                "max_tokens": a.tokens, "temperature": 0, "logprobs": False}
        req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        m = r["choices"][0]["message"]
        res.append({"reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
                    "content": m.get("content") or ""})
    json.dump(res, open(a.out, "w"), indent=1)
    print(a.out, [len(x["reasoning"]) + len(x["content"]) for x in res])


if __name__ == "__main__":
    main()
