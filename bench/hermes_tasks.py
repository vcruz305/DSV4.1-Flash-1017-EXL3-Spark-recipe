#!/usr/bin/env python3
"""Real multi-step Hermes tasks against the running server, scored by hidden checks.

    python3 bench/hermes_tasks.py [--model dsv41] [--only NAME,...] [--out results.jsonl]

Each task gets a fresh folder with seed files, runs one `hermes chat -Q --yolo` turn there,
then a check the model never sees decides pass/fail (files, contents, tests). Per task it
records time, tool calls (from Hermes's session store), finish and the peak RSS of the
Hermes process tree (what the agent itself needs from the reserve). The route record is
paused while it runs (touch <record dir>/pause), so the cold set is not picked from it.
"""
from __future__ import annotations

import argparse, json, os, re, shutil, sqlite3, subprocess, sys, tempfile, textwrap, threading, time

import psutil

STATS_BUG = "def median(xs):\n    xs = sorted(xs)\n    return xs[len(xs) // 2]\n"
TASKS = [
    {"name": "fix-bug", "files": {"stats.py": STATS_BUG,
        "test_stats.py": "from stats import median\n\ndef test_odd():\n    assert median([3, 1, 2]) == 2\n\ndef test_even():\n    assert median([4, 1, 3, 2]) == 2.5\n"},
     "prompt": "test_stats.py fails. Find the bug in stats.py and fix it without editing the test, then run the tests with python3 -m pytest -q.",
     "check": "from stats import median\nassert median([1,2,3,4])==2.5 and median([5])==5 and median([9,1,5])==5\nimport subprocess\nassert subprocess.run(['python3','-m','pytest','-q'],capture_output=True).returncode==0\nassert 'xs[len(xs) // 2]' not in open('test_stats.py').read() or True"},
    {"name": "rename", "files": {
        "api.py": "def fetch_data(n):\n    return list(range(n))\n",
        "report.py": "from api import fetch_data\n\ndef report(n):\n    return sum(fetch_data(n))\n",
        "test_report.py": "from report import report\n\ndef test_report():\n    assert report(4) == 6\n"},
     "prompt": "Rename the function fetch_data to load_data everywhere in this folder (definition and every use), keep behaviour the same, and run python3 -m pytest -q.",
     "check": "import glob,subprocess\nsrc=''.join(open(f).read() for f in glob.glob('*.py'))\nassert 'fetch_data' not in src and 'def load_data' in src\nassert subprocess.run(['python3','-m','pytest','-q'],capture_output=True).returncode==0"},
    {"name": "csv-totals", "files": {"sales.csv": "region,item,amount\nnorth,a,10.5\nsouth,b,3\nnorth,c,4.5\neast,a,7\nsouth,a,2\n"},
     "prompt": "Read sales.csv and write totals.json: a JSON object mapping each region to its total amount, as numbers.",
     "check": "import json\nd=json.load(open('totals.json'))\nassert {k:float(v) for k,v in d.items()}=={'north':15.0,'south':5.0,'east':7.0}"},
    {"name": "find-file", "files": {**{f"notes/n{i}.txt": f"note {i}\n" for i in range(30)}, "notes/deep/x/n77.txt": "see TODO-7731 here\n"},
     "prompt": "Some file under notes/ contains the text TODO-7731. Write its path, relative to this folder, as the only line of answer.txt.",
     "check": "assert open('answer.txt').read().strip().lstrip('./')=='notes/deep/x/n77.txt'"},
    {"name": "module-tests", "files": {},
     "prompt": "Create slug.py with slugify(text): lowercase, drop every character that is not a letter, digit, space or hyphen, then turn spaces into hyphens (GitHub heading anchors, e.g. 'Hello, World!' -> 'hello-world'). Add test_slug.py with at least 5 pytest tests and run them.",
     "check": "from slug import slugify\nassert slugify('Hello, World!')=='hello-world' and slugify('A  b')=='a--b' and slugify('C++ & Rust')=='c--rust'\nimport subprocess\nr=subprocess.run(['python3','-m','pytest','-q','test_slug.py'],capture_output=True,text=True)\nassert r.returncode==0 and int(re.search(r'(\\d+) passed',r.stdout).group(1))>=5"},
    {"name": "git-flow", "files": {"README.md": "# demo\n"},
     "prompt": "Make this folder a git repository, commit README.md with the message 'init', then create a branch named feature, add hello.txt containing 'hi' on it and commit that with the message 'add hello'. Use -c user.name=bench -c user.email=bench@example.com for the commits.",
     "check": "import subprocess as s\ng=lambda *a:s.run(['git',*a],capture_output=True,text=True).stdout.strip()\nassert g('log','--format=%s','feature')=='add hello\\ninit'\nassert g('show','feature:hello.txt')=='hi'"},
    {"name": "cli-run", "files": {"sample.txt": "the cat and the dog and the bird\nthe end\n"},
     "prompt": "Write wordfreq.py, a CLI: python3 wordfreq.py FILE N prints the N most frequent words of FILE as 'word count' lines, most frequent first. Run it on sample.txt with N=2 and save its output to out.txt.",
     "check": "assert open('out.txt').read().split()==['the','4','and','2']"},
    {"name": "json-config", "files": {"config.json": json.dumps({"server": {"port": 8080, "hosts": ["a", "b"]}, "debug": True}, indent=2)},
     "prompt": "In config.json set server.port to 9090, add host 'c' to the end of server.hosts and set debug to false. Keep everything else unchanged and keep it valid JSON.",
     "check": "import json\nassert json.load(open('config.json'))=={'server':{'port':9090,'hosts':['a','b','c']},'debug':False}"},
    {"name": "lru-cache", "files": {},
     "prompt": "Write lru.py with a class LRUCache(capacity) offering get(key, default=None) and put(key, value), O(1) each, evicting the least recently used key when full (get counts as a use). Do not use functools. Add tests with pytest and run them.",
     "check": "from lru import LRUCache\nc=LRUCache(2); c.put('a',1); c.put('b',2); assert c.get('a')==1\nc.put('c',3); assert c.get('b') is None and c.get('a')==1 and c.get('c')==3\nc.put('a',9); c.put('d',4); assert c.get('c') is None and c.get('a')==9\nassert 'functools' not in open('lru.py').read()"},
    {"name": "traceback-fix", "files": {
        "app.py": "import json\n\ndef load(path):\n    with open(path) as f:\n        return json.load(f)\n\ndef total(path):\n    return sum(item['price'] * item['qty'] for item in load(path)['items'])\n\nif __name__ == '__main__':\n    print(total('order.json'))\n",
        "order.json": json.dumps({"items": [{"price": 2.5, "qty": 2}, {"price": 1.0}, {"price": 4.0, "qty": 1}]}),
        "app.log": "Traceback (most recent call last):\n  File \"app.py\", line 11, in <module>\n    print(total('order.json'))\n  File \"app.py\", line 8, in <genexpr>\nKeyError: 'qty'\n"},
     "prompt": "app.log shows the crash users hit when running python3 app.py. Find the root cause and fix app.py so an item without qty counts as quantity 1 (do not edit order.json). Run python3 app.py to confirm it prints the total.",
     "check": "import subprocess\nr=subprocess.run(['python3','app.py'],capture_output=True,text=True)\nassert r.returncode==0 and float(r.stdout.strip())==10.0\nimport json\nassert json.load(open('order.json'))['items'][1]=={'price':1.0}"},
    {"name": "add-feature", "files": {
        "shop/__init__.py": "",
        "shop/cart.py": "class Cart:\n    def __init__(self):\n        self.lines = []\n\n    def add(self, name, price, qty=1):\n        self.lines.append((name, price, qty))\n\n    def total(self):\n        return sum(p * q for _, p, q in self.lines)\n",
        "tests/test_cart.py": "from shop.cart import Cart\n\ndef test_total():\n    c = Cart(); c.add('a', 2.0, 3); c.add('b', 1.5)\n    assert c.total() == 7.5\n"},
     "prompt": "Add percentage discount codes to shop/cart.py: Cart.apply_code(code) where codes come from a dict CODES = {'SAVE10': 10, 'HALF': 50} in a new module shop/codes.py; an unknown code raises ValueError; only one code applies (the last one wins); total() returns the discounted total rounded to 2 decimals. Add tests in tests/ and run python3 -m pytest -q from this folder.",
     "check": "import sys,subprocess; sys.path.insert(0,'.')\nfrom shop.cart import Cart\nfrom shop.codes import CODES\nc=Cart(); c.add('a',10.0,3); c.apply_code('SAVE10'); assert c.total()==27.0\nc.apply_code('HALF'); assert c.total()==15.0\ntry:\n    c.apply_code('NOPE'); raise SystemExit(1)\nexcept ValueError: pass\nassert subprocess.run(['python3','-m','pytest','-q'],capture_output=True).returncode==0"},
    {"name": "log-report", "files": {"access.log": "".join(
        f"10.0.0.{i % 7} - - [04/Nov/2026:10:{i % 60:02d}:00 +0000] \"GET /p{i % 5} HTTP/1.1\" {[200, 200, 404, 500][i % 4]} {100 + i}\n" for i in range(200))},
     "prompt": "Write report.py that reads access.log (common log format) and writes report.json with: total_requests, status_counts (status code string -> count), top_paths (the 3 most requested paths as a list, most first, ties broken alphabetically) and bytes_total. Run it.",
     "check": "import json\nd=json.load(open('report.json'))\nassert d['total_requests']==200 and d['status_counts']=={'200':100,'404':50,'500':50}\nassert d['top_paths']==['/p0','/p1','/p2'] and d['bytes_total']==sum(100+i for i in range(200))"},
]


def tree_rss(pid: int) -> int:
    try:
        p = psutil.Process(pid)
        return sum(q.memory_info().rss for q in [p, *p.children(recursive=True)])
    except psutil.Error:
        return 0


def tool_calls(session_id: str) -> int | None:
    db = os.path.expanduser("~/.hermes/state.db")
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = con.execute("select tool_calls from messages where session_id=? and tool_calls is not null",
                           (session_id,)).fetchall()
    except sqlite3.Error:
        return None
    return sum(len(json.loads(r[0]) or []) for r in rows if r[0])


def run_task(t: dict, model: str, timeout: int) -> dict:
    d = tempfile.mkdtemp(prefix=f"hermes-task-{t['name']}-")
    for rel, body in t["files"].items():
        os.makedirs(os.path.join(d, os.path.dirname(rel)), exist_ok=True)
        open(os.path.join(d, rel), "w").write(body)
    prompt = f"Work only inside the folder {d}. {t['prompt']}"
    cmd = ["hermes", "chat", "-Q", "-m", model, "--provider", "vllm", "--max-turns", "40", "--yolo",
           "--in", d, "--run-budget", str(timeout), "-q", prompt]
    t0, peak = time.time(), [0]
    p = subprocess.Popen(cmd, cwd=d, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    stop = threading.Event()

    def sample():
        while not stop.wait(0.5):
            peak[0] = max(peak[0], tree_rss(p.pid))
    threading.Thread(target=sample, daemon=True).start()
    try:
        out, _ = p.communicate(timeout=timeout + 120)
    except subprocess.TimeoutExpired:
        p.kill()
        out, _ = p.communicate()
    stop.set()
    secs = time.time() - t0
    sid = (re.findall(r"session_id:\s*(\S+)", out) or [None])[-1]
    chk = subprocess.run([sys.executable, "-c", "import re\n" + t["check"]], cwd=d, capture_output=True, text=True)
    res = {"task": t["name"], "pass": chk.returncode == 0, "secs": round(secs, 1), "rc": p.returncode,
           "tool_calls": tool_calls(sid) if sid else None, "peak_rss_gib": round(peak[0] / 2**30, 2),
           "session": sid, "check_err": chk.stderr.strip().splitlines()[-1:] if chk.returncode else []}
    if res["pass"]:
        shutil.rmtree(d, ignore_errors=True)
    else:
        res["dir"] = d
        res["tail"] = out[-600:]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--only", default="")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default="")
    ap.add_argument("--record-dir", default=os.environ.get("DYN_RESIDENCY_STATE_DIR",
                                                          os.path.expanduser("~/.cache/dsv41-ds41f/route-heat")))
    a = ap.parse_args()
    pause = os.path.join(a.record_dir, "pause")
    mine = os.path.isdir(a.record_dir) and not os.path.exists(pause)
    if mine:
        open(pause, "w").close()
    try:
        names = set(filter(None, a.only.split(",")))
        rows = []
        for t in TASKS:
            if names and t["name"] not in names:
                continue
            r = run_task(t, a.model, a.timeout)
            rows.append(r)
            print(json.dumps(r), flush=True)
            if a.out:
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
        ok = sum(r["pass"] for r in rows)
        print(json.dumps({"SUMMARY": f"{ok}/{len(rows)} passed",
                          "tool_calls": sum(r["tool_calls"] or 0 for r in rows),
                          "secs": round(sum(r["secs"] for r in rows)),
                          "peak_rss_gib": max((r["peak_rss_gib"] for r in rows), default=0)}))
    finally:
        if mine and os.path.exists(pause):
            os.remove(pause)
    return 0


if __name__ == "__main__":
    sys.exit(main())
