#!/usr/bin/env python3
"""GSM8K through the running server's chat API (greedy), flex-scored.

  python3 bench/gsm8k_api.py --data gsm8k-test.jsonl --limit 200 --out gsm.jsonl
  python3 bench/gsm8k_api.py --score gsm.jsonl

--data: one {"id", "question", "ref"} per line (ref = the number after '####' in the
openai/gsm8k answer). Scoring: the last bolded number wins, else the last number in the text,
compared exactly with ref (the flex rule used for the other Spark quality sets)."""
import argparse, json, os, re, sys, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
ap.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "dsv41"))
ap.add_argument("--data"); ap.add_argument("--out"); ap.add_argument("--score")
ap.add_argument("--limit", type=int, default=200)
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--think", type=int, default=0)
ap.add_argument("--conc", type=int, default=1)
a = ap.parse_args()


def final(text):
    b = re.findall(r"\*\*\\?\$?(-?\d[\d,]*\.?\d*)\\?\$?\*\*", text or "")
    if b:
        return b[-1].replace(",", "").rstrip(".")
    n = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    return n[-1].replace(",", "").rstrip(".") if n else None


def score(path):
    rows = [json.loads(l) for l in open(path)]
    ok = sum(final(r["text"]) == r["ref"] for r in rows)
    trunc = sum(r.get("finish") == "length" for r in rows)
    print(f"gsm8k {ok}/{len(rows)} = {ok / max(len(rows), 1):.1%} (truncated {trunc})")


if a.score:
    score(a.score); sys.exit(0)

probs = [json.loads(l) for l in open(a.data)][: a.limit]
done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
todo = [p for p in probs if p["id"] not in done]
lock = threading.Lock()


def run(p):
    body = {"model": a.model, "temperature": 0, "max_tokens": a.max_tokens,
            "chat_template_kwargs": {"enable_thinking": bool(a.think)},
            "messages": [{"role": "user", "content": p["question"] +
                          "\n\nSolve step by step, then give the final answer as a bold number, like **42**."}]}
    t0 = time.time()
    req = urllib.request.Request(a.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    c = r["choices"][0]
    row = {"id": p["id"], "ref": p["ref"], "text": c["message"].get("content") or "", "finish": c.get("finish_reason"),
           "gen": r["usage"]["completion_tokens"], "s": round(time.time() - t0, 2)}
    with lock, open(a.out, "a") as fh:
        fh.write(json.dumps(row) + "\n")


it = iter(todo)


def worker():
    for p in it:
        run(p)


th = [threading.Thread(target=worker) for _ in range(a.conc)]
[t.start() for t in th]; [t.join() for t in th]
score(a.out)
