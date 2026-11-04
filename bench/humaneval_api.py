#!/usr/bin/env python3
"""HumanEval pass@1 through the served OpenAI API (greedy, thinking off by default).

  python3 bench/humaneval_api.py --data HumanEval.jsonl --out gen.jsonl [--n 164] [--conc 4] [--think]
  python3 bench/humaneval_api.py --score gen.jsonl      # runs the tests in python:3.12-slim, --network none

--data is the 164-problem HumanEval jsonl (github.com/openai/human-eval, data/HumanEval.jsonl.gz, gunzipped).
Generation only talks to the server; the model's code is executed only by --score, inside a throwaway
container without network (sudo -n docker), one problem per process with a 15 s timeout."""
import argparse, concurrent.futures as cf, json, os, re, subprocess, sys, tempfile, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--data"); ap.add_argument("--out"); ap.add_argument("--score")
ap.add_argument("--n", type=int, default=164); ap.add_argument("--conc", type=int, default=4)
ap.add_argument("--max-tokens", type=int, default=1024); ap.add_argument("--think", action="store_true")
ap.add_argument("--url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000") + "/v1/chat/completions")
ap.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "dsv41"))
a = ap.parse_args()


def program(p, reply):
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", reply, re.S)
    code = max(blocks, key=len) if blocks else reply
    if f"def {p['entry_point']}" not in code:
        code = p["prompt"] + code
    return code + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})\n"


RUNNER = r'''
import json, subprocess, sys
res = {}
for l in open("/w/progs.jsonl"):
    j = json.loads(l)
    open("/tmp/p.py", "w").write(j["src"])
    try:
        res[j["task_id"]] = subprocess.run([sys.executable, "/tmp/p.py"], capture_output=True, timeout=15).returncode == 0
    except subprocess.TimeoutExpired:
        res[j["task_id"]] = False
json.dump(res, open("/w/res.json", "w"))
'''

if a.score:
    rows = [json.loads(l) for l in open(a.score)]
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(a.score))) as w:
        with open(os.path.join(w, "progs.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps({"task_id": r["task_id"], "src": r["program"]}) + "\n")
        open(os.path.join(w, "run.py"), "w").write(RUNNER)
        os.chmod(w, 0o777)
        subprocess.run(["sudo", "-n", "docker", "run", "--rm", "--network", "none", "--memory", "2g", "--pids-limit", "256",
                        "-v", f"{w}:/w", "python:3.12-slim", "python", "/w/run.py"], check=True)
        res = json.load(open(os.path.join(w, "res.json")))
    ok = sum(res.values())
    print(json.dumps({"file": a.score, "n": len(res), "pass": ok, "pass@1": round(ok / len(res), 4),
                      "failed": sorted(k for k, v in res.items() if not v)}))
    sys.exit(0)

probs = [json.loads(l) for l in open(a.data)][: a.n]


def ask(p):
    body = {"model": a.model, "temperature": 0, "max_tokens": a.max_tokens,
            "chat_template_kwargs": {"enable_thinking": bool(a.think)},
            "messages": [{"role": "user", "content": "Complete this Python function. Reply with the full function in one "
                          "```python code block.\n\n```python\n" + p["prompt"] + "```"}]}
    t = time.time()
    req = urllib.request.Request(a.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    j = json.load(urllib.request.urlopen(req, timeout=1800))
    m = j["choices"][0]["message"]
    reply = m.get("content") or ""
    return {"task_id": p["task_id"], "reply": reply, "finish": j["choices"][0].get("finish_reason"),
            "gen": j["usage"]["completion_tokens"], "s": round(time.time() - t, 2), "program": program(p, reply)}


t0 = time.time()
with cf.ThreadPoolExecutor(a.conc) as ex, open(a.out, "w") as f:
    for i, r in enumerate(ex.map(ask, probs)):
        f.write(json.dumps(r) + "\n"); f.flush()
        if i % 20 == 0:
            print(f"{i + 1}/{len(probs)} {time.time() - t0:.0f} s", flush=True)
print(f"done {len(probs)} in {time.time() - t0:.0f} s -> {a.out}")
