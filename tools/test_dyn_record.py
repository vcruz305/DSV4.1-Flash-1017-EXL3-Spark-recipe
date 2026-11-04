#!/usr/bin/env python3
"""Checks of the route record (scripts/py/dyn_residency.py) without an engine:
paused-at-boot traffic never reaches the file, and old counts fade so recent use wins.
Run: python3 tools/test_dyn_record.py"""
import importlib.util
import json
import os
import sys
import tempfile
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E = 4  # experts in the fake pack


class T:  # the bits of a torch tensor the module touches
    def __init__(self, a): self.a = np.asarray(a, dtype="float64")
    def cpu(self): return self
    def numpy(self): return self.a
    def sum(self): return types.SimpleNamespace(item=lambda: float(self.a.sum()))


def load_module(state, half_life="14"):
    os.environ["DYN_RESIDENCY_STATE"] = state
    os.environ["DYN_RESIDENCY_HALF_LIFE_DAYS"] = half_life
    spec = importlib.util.spec_from_file_location("dynres_t", os.path.join(HERE, "..", "scripts", "py", "dyn_residency.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def counters(n):
    st = {k: T(np.array(n, dtype="float64")) for k in ("counts", "calls", "rows")}
    sys.modules["vllm_exl3"] = types.SimpleNamespace(route_stats=types.SimpleNamespace(_STATE={"l0": st}))
    sys.modules["vllm_exl3.route_stats"] = sys.modules["vllm_exl3"].route_stats
    return st


def write(state, counts, age_days):
    meta = {"boots": 1, "saved_unix": time.time() - age_days * 86400}
    c = np.asarray(counts, dtype="float64")
    np.savez(state, counts_0=c, calls_0=c, rows_0=c, meta=np.array(json.dumps(meta)))


def boot(m, warmup):
    st = counters(warmup)
    m._S["rec"] = m._load_record((E,))
    m._take_base()
    return st


def test_paused_from_boot():
    with tempfile.TemporaryDirectory() as d:
        state = os.path.join(d, "rec.npz")
        write(state, [100, 0, 0, 0], 0)
        open(os.path.join(d, "pause"), "w").close()
        m = load_module(state)
        st = boot(m, [7, 7, 0, 0])                     # warm-up routes: never counted
        st["counts"].a += [0, 50, 0, 0]                 # benchmark while paused, before the saver starts
        m._pause_check(st, os.path.join(d, "pause"))
        m._save_record(st)
        assert np.allclose(np.load(state)["counts_0"], [100, 0, 0, 0], atol=0.01), "paused routes leaked"
        os.remove(os.path.join(d, "pause"))
        m._pause_check(st, os.path.join(d, "pause"))
        st["counts"].a += [0, 0, 5, 0]                  # real traffic after resume
        m._save_record(st)
        assert np.allclose(np.load(state)["counts_0"], [100, 0, 5, 0], atol=0.01), np.load(state)["counts_0"]


def test_recent_beats_stale():
    with tempfile.TemporaryDirectory() as d:
        state = os.path.join(d, "rec.npz")
        for hl, stale_wins in (("14", False), ("0", True)):
            write(state, [100, 0, 0, 0], 28)            # last month: expert 0
            m = load_module(state, hl)
            st = boot(m, [0, 0, 0, 0])
            st["counts"].a += [0, 0, 0, 40]             # this week: expert 3
            m._save_record(st)
            c = np.load(state)["counts_0"]
            assert np.isclose(c[0], 25.0 if hl == "14" else 100.0, rtol=1e-3), c
            assert (c[0] > c[3]) == stale_wins, (hl, c)


if __name__ == "__main__":
    for f in (test_paused_from_boot, test_recent_beats_stale):
        f()
        print("ok", f.__name__)
