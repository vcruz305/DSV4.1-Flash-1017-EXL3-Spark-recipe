"""Opt-in dynamic cold-expert residency (DYN_RESIDENCY=1), runs inside the vLLM worker.

The image's MoE kernels (exl3_moe_mixedk, K-grouped) find every routed expert through a
per-layer int64 device table, ``layer._exl3_ptrs["{gate,up,down}_trellis"][expert]``; the
fat-prefill path reads ``layer._exl3_inners[expert][proj].trellis``. A swap promotes a
cold (ATS-aliased) expert C into the CUDA arena slot of a resident expert H with the same
trellis shapes, and re-points H at its own bytes in the (64-byte re-laid) checkpoint
shard. Only pointers and one slot copy change, so the math is untouched and CUDA graphs
stay valid (they read the table from device memory). Swaps happen between steps, after a
device synchronize, at most DYN_RESIDENCY_MAX_SWAPS per tick.

Heat = decayed route counts from the image's route_stats module (needs ROUTE_STATS=1).
Knobs: DYN_RESIDENCY_EVERY (execute_model calls per tick, 64), _MAX_SWAPS (8; 0 =
monitor only), _RATIO (2.0) and _MIN (32 routes): promote C over H only if
heat[C] > RATIO * heat[H] + MIN; _DECAY (0.9 per tick); _VERIFY=1 checks bytes per swap;
_TEST_SWAPS=n forces n random same-shape swaps per tick (correctness testing only).
Log: $VLLM_EXL3_ROUTE_STATS_DIR/dynres.jsonl, one line per tick.

Permanent record: with DYN_RESIDENCY_STATE=<file.npz> the cumulative route counts of every
boot are added to that file (loaded at the first tick, saved every _SAVE_S=300 s, same
layout as a route_stats dump, so packs/1.55bpw-rexl3/tools/plan_reserve.py --npz reads it). A saver thread writes it
when the counts changed and SAVE_S passed, or at once when <record dir>/flush.now appears
(serve.sh touches it before a restart, so idle-time and last-minutes counts are not lost).
While <record dir>/pause exists the traffic is not added to the record (benchmarks, eval replays);
it still counts for this boot's swap decisions. Each save also writes <record>.status.json
(routes, boots, cold-route share of the last 64 ticks) for dashboards. Swaps then rank
experts by heat = record + this boot's counts + _SHORT (1.0) * decayed recent counts.
Old traffic fades: at each boot the record's counts are scaled by 0.5 ** (days since its last
save / _HALF_LIFE_DAYS (14); 0 keeps them all), so the cold set follows a change of use.
MAX_SWAPS=0 keeps recording without swapping.
"""
import json
import logging
import os
import random
import threading
from collections import deque
import time

log = logging.getLogger("vllm.dyn_residency")


def _f(name, default):
    return type(default)(os.environ.get("DYN_RESIDENCY_" + name, default))


EVERY = _f("EVERY", 64)
MAX_SWAPS = _f("MAX_SWAPS", 8)
RATIO = _f("RATIO", 2.0)
MIN = _f("MIN", 32.0)
DECAY = _f("DECAY", 0.9)
VERIFY = _f("VERIFY", 0)
TEST_SWAPS = _f("TEST_SWAPS", 0)
STATE = os.environ.get("DYN_RESIDENCY_STATE", "")
SAVE_S = _f("SAVE_S", 300.0)
SHORT = _f("SHORT", 1.0)
HALF_LIFE_DAYS = _f("HALF_LIFE_DAYS", 14.0)
OUT = os.path.join(os.environ.get("VLLM_EXL3_ROUTE_STATS_DIR", "/tmp"), "dynres.jsonl")
PROJ = {"gate": "w1", "up": "w3", "down": "w2"}

_S = {"calls": 0, "layers": None, "last": None, "off": False, "ticks": 0, "swaps": 0, "t0": time.time(),
      "rec": None, "saved": time.time(), "excl": None, "pause0": None, "base": None,
      "recent": deque(maxlen=64)}


class _Layer:
    def __init__(self, layer, key, row, cold):
        import numpy as np

        self.layer, self.key, self.row = layer, key, row
        self.cold = set(cold)
        inn = layer._exl3_inners
        self.shape = [tuple(tuple(inn[e][p].trellis.shape) for p in PROJ) for e in range(len(inn))]
        self.heat = np.zeros(len(inn))
        self.short = np.zeros(len(inn))
        pk = getattr(layer, "_exl3_packed_id", None)
        self.pid = pk.cpu().tolist() if pk is not None else None


def _discover(model):
    from vllm_exl3 import ats_alias, route_stats

    out = []
    for m in model.modules():
        if not (getattr(m, "_exl3_ptrs", None) and getattr(m, "_exl3_inners", None)):
            continue
        if getattr(m, "_exl3_gateless", False):
            continue
        key, row = ats_alias._layer_key(m), route_stats._row_of(m)
        if key is None or row is None or not key.startswith("layers."):
            continue
        al = ats_alias.layer_aliases(m) or {}
        cold = set.intersection(*(set(al.get(p, {})) for p in PROJ)) if al else set()
        out.append(_Layer(m, key, row, cold))
    log.info("dyn_residency: %d MoE layers, %d cold experts", len(out), sum(len(l.cold) for l in out))
    return out


_IDX = {}
_HDR = {}


def _entry(d, k):
    from vllm_exl3 import ats_alias

    if not _IDX:
        _IDX.update(ats_alias._pack_index())
    path = os.path.join(d, _IDX[k])
    hdr = _HDR.get(path)
    if hdr is None:
        with open(path, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            hdr = _HDR[path] = (8 + n, json.loads(fh.read(n)))
    base, h = hdr
    b, e = h[k]["data_offsets"]
    return path, base + b, e - b, tuple(h[k]["shape"])


def _shard_view(ls, e, p, like):
    from vllm_exl3 import ats_alias

    d = os.environ.get("VLLM_EXL3_MODEL_DIR") or os.environ.get("VLLM_ENGRAM_MODEL_DIR")
    k = f"{ls.key}.ffn.experts.{e}.{PROJ[p]}.trellis"
    path, off, nbytes, shape = _entry(d, k)
    if tuple(shape) != tuple(like.shape):
        raise RuntimeError(f"dyn_residency: {k} shard shape {shape} != {tuple(like.shape)}")
    return ats_alias._ats_view(path, off, nbytes, shape, like.dtype, like.device)


def _set_ptr(ls, p, e, value):
    ls.layer._exl3_ptrs[p + "_trellis"][e] = value
    packed = getattr(ls.layer, "_exl3_ptrs_packed", None)
    if packed is not None and ls.pid is not None:
        packed[p + "_trellis"][ls.pid[e]] = value


def _swap(ls, c, h):
    import torch
    from vllm_exl3.exl3 import make_linear_exl3

    inn = ls.layer._exl3_inners
    for p in PROJ:
        ic, ih = inn[c][p], inn[h][p]
        slot = ih.trellis
        hview = _shard_view(ls, h, p, slot)
        if VERIFY and not torch.equal(hview, slot):
            raise RuntimeError(f"dyn_residency: shard bytes != resident {ls.key} e{h} {p}")
        _set_ptr(ls, p, h, hview.data_ptr())
        inn[h][p] = make_linear_exl3(hview, ih.suh, ih.svh, ih.mcg_tensor, ih.mul1_tensor,
                                     out_dtype=ih.out_dtype)
        slot.copy_(ic.trellis)
        if VERIFY and not torch.equal(slot, ic.trellis):
            raise RuntimeError(f"dyn_residency: slot copy mismatch {ls.key} e{c} {p}")
        _set_ptr(ls, p, c, slot.data_ptr())
        inn[c][p] = make_linear_exl3(slot, ic.suh, ic.svh, ic.mcg_tensor, ic.mul1_tensor,
                                     out_dtype=ic.out_dtype)
    ls.cold.discard(c)
    ls.cold.add(h)


def _plan(layers):
    props = []
    for ls in layers:
        if not ls.cold:
            continue
        res = sorted((e for e in range(len(ls.heat)) if e not in ls.cold), key=lambda e: ls.heat[e])
        used = set()
        for c in sorted(ls.cold, key=lambda e: -ls.heat[e]):
            if TEST_SWAPS:
                pool = [e for e in res if ls.shape[e] == ls.shape[c] and e not in used]
                h = random.choice(pool) if pool else None
            else:
                h = next((e for e in res if ls.shape[e] == ls.shape[c] and e not in used), None)
                if h is None or ls.heat[c] <= RATIO * ls.heat[h] + MIN:
                    continue
            if h is None:
                continue
            used.add(h)
            props.append((ls.heat[c] - ls.heat[h], ls, c, h))
            if TEST_SWAPS and len(props) >= TEST_SWAPS:
                return props
    if TEST_SWAPS:
        return props[:TEST_SWAPS]
    return sorted(props, key=lambda x: -x[0])[:MAX_SWAPS]


def _load_record(shape):
    import numpy as np

    rec = {"counts": np.zeros(shape), "calls": None, "rows": None, "boots": 0}
    if STATE and os.path.exists(STATE):
        z = np.load(STATE)
        if z["counts_0"].shape == shape:
            meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
            # older traffic fades so the cold set follows a change of use; files without a save
            # time (older builds) are taken as they are
            age_days = max(0.0, time.time() - meta["saved_unix"]) / 86400 if "saved_unix" in meta else 0.0
            keep = 0.5 ** (age_days / HALF_LIFE_DAYS) if HALF_LIFE_DAYS > 0 else 1.0
            rec.update(counts=keep * z["counts_0"].astype("float64"), calls=keep * z["calls_0"].astype("float64"),
                       rows=keep * z["rows_0"].astype("float64"), boots=meta.get("boots", 0))
            if keep < 1.0:
                log.info("dyn_residency: record last saved %.1f days ago, old counts x%.3f (half-life %.0f d)",
                         age_days, keep, HALF_LIFE_DAYS)
        else:
            log.warning("dyn_residency: record %s has shape %s, expected %s; starting a new one",
                        STATE, z["counts_0"].shape, shape)
    rec["boots"] += 1
    log.info("dyn_residency: record %s, %d routes over %d boots", STATE or "(off)",
             int(rec["counts"].sum()), rec["boots"])
    return rec


def _snap(st):
    b = _S["base"] or (0, 0, 0)
    return [st[k].cpu().numpy().astype("float64") - b[i] for i, k in enumerate(("counts", "calls", "rows"))]


def _take_base():
    # vLLM's profile run and CUDA-graph capture route dummy tokens before the first real step;
    # whatever the counters hold at the first execute_model is warm-up, not traffic
    try:
        from vllm_exl3 import route_stats

        st = next(iter(route_stats._STATE.values()), None) if route_stats._STATE else None
        _S["base"] = (tuple(st[k].cpu().numpy().astype("float64") for k in ("counts", "calls", "rows"))
                      if st is not None else (0, 0, 0))
        if st is not None:
            log.info("dyn_residency: %d warm-up routes left out", int(_S["base"][0].sum()))
        if STATE and os.path.exists(os.path.join(os.path.dirname(STATE), "pause")):
            # paused from the first step: the saver only starts at the first tick, EVERY calls later
            _S["pause0"] = [0.0, 0.0, 0.0]
            log.info("dyn_residency: record paused from the first step")
    except Exception:
        _S["base"] = (0, 0, 0)


def _pause_check(st, pause):
    if os.path.exists(pause):
        if _S["pause0"] is None:
            _S["pause0"] = _snap(st)
            log.info("dyn_residency: record paused (%s)", pause)
    elif _S["pause0"] is not None:
        now = _snap(st)
        ex = _S["excl"] or [0, 0, 0]
        _S["excl"] = [ex[i] + now[i] - _S["pause0"][i] for i in range(3)]
        _S["pause0"] = None
        log.info("dyn_residency: record resumed; %d paused routes left out", int(_S["excl"][0].sum()))


def _host_owned(path):
    # the container runs as root: hand each file to the owner of its (host-mounted) folder,
    # or host-side tools (auto_cold, plan_reserve, a restore) cannot replace it
    try:
        d = os.stat(os.path.dirname(path) or ".")
        os.chown(path, d.st_uid, d.st_gid)
    except OSError:
        pass


def _save_record(st):
    import numpy as np

    r = _S["rec"]
    now = _S["pause0"] or _snap(st)
    ex = _S["excl"] or [0, 0, 0]
    c = r["counts"] + now[0] - ex[0]
    calls = now[1] - ex[1] + (r["calls"] if r["calls"] is not None else 0)
    rows = now[2] - ex[2] + (r["rows"] if r["rows"] is not None else 0)
    meta = {"boots": r["boots"], "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "saved_unix": time.time(),
            "half_life_days": HALF_LIFE_DAYS,
            "note": "dyn_residency permanent record: cumulative token-routes per expert"}
    tmp = STATE + ".tmp.npz"
    np.savez(tmp, counts_0=c, calls_0=calls, rows_0=rows, meta=np.array(json.dumps(meta)))
    os.replace(tmp, STATE)
    _host_owned(STATE)
    _S["saved"] = time.time()
    rt = sum(x[0] for x in _S["recent"])
    status = {"time": time.time(), "routes": int(c.sum()), "boots": r["boots"],
              "record_bytes": os.path.getsize(STATE), "paused": _S["pause0"] is not None,
              "recent_routes": int(rt), "recent_cold_share": (sum(x[1] for x in _S["recent"]) / rt) if rt else None,
              "swaps_total": _S["swaps"], "max_swaps": MAX_SWAPS}
    with open(STATE[:-4] + ".status.json.tmp", "w") as fh:
        json.dump(status, fh)
    os.replace(STATE[:-4] + ".status.json.tmp", STATE[:-4] + ".status.json")
    _host_owned(STATE[:-4] + ".status.json")


def _saver(st):
    flush = os.path.join(os.path.dirname(STATE), "flush.now")
    pause = os.path.join(os.path.dirname(STATE), "pause")
    last = None
    while True:
        time.sleep(2.0)
        try:
            _pause_check(st, pause)
            asked = os.path.exists(flush)
            if not asked and time.time() - _S["saved"] <= SAVE_S:
                continue
            n = float(st["counts"].sum().item())
            if n != last:
                _save_record(st)
                last = n
                log.info("dyn_residency: record saved (%s)", "flush" if asked else "periodic")
            else:
                _S["saved"] = time.time()
            if asked:
                os.remove(flush)
        except Exception as exc:
            log.warning("dyn_residency: record save failed: %r", exc)


def tick(runner):
    if _S["off"]:
        return
    _S["calls"] += 1
    if _S["calls"] % EVERY:
        return
    try:
        import torch
        from vllm_exl3 import route_stats

        if torch.cuda.is_current_stream_capturing() or not route_stats._STATE:
            return
        if _S["layers"] is None:
            m = runner.get_model() if hasattr(runner, "get_model") else runner.model
            _S["layers"] = _discover(m)
        layers = _S["layers"]
        st = next(iter(route_stats._STATE.values()))
        c = st["counts"].cpu().numpy().astype("float64") - _S["base"][0]
        if _S["rec"] is None:
            _S["rec"] = _load_record(c.shape)
            if STATE:
                threading.Thread(target=_saver, args=(st,), name="dynres-saver", daemon=True).start()
        delta = c if _S["last"] is None else c - _S["last"]
        _S["last"] = c
        tot = cold = 0.0
        for ls in layers:
            d = delta[ls.row]
            tot += d.sum()
            cold += sum(d[e] for e in ls.cold)
            ls.short = DECAY * ls.short + d
            ls.heat = _S["rec"]["counts"][ls.row] + c[ls.row] + SHORT * ls.short
        _S["recent"].append((tot, cold))
        props = _plan(layers) if (MAX_SWAPS or TEST_SWAPS) else []
        t = time.time()
        if props:
            torch.cuda.synchronize()
            for _, ls, ce, he in props:
                _swap(ls, ce, he)
            torch.cuda.synchronize()
        _S["ticks"] += 1
        _S["swaps"] += len(props)
        rec = {"t": round(time.time() - _S["t0"], 1), "tick": _S["ticks"], "routes": int(tot),
               "cold_share": round(cold / tot, 5) if tot else None, "swaps": len(props),
               "swaps_total": _S["swaps"], "swap_ms": round((time.time() - t) * 1e3, 1)}
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception as exc:
        _S["off"] = True
        log.exception("dyn_residency disabled after error: %r", exc)


def patch(mod):
    cls = mod.GPUModelRunner
    orig = cls.execute_model

    def execute_model(self, *a, **k):
        if _S["base"] is None:
            _take_base()
        tick(self)
        return orig(self, *a, **k)

    cls.execute_model = execute_model
    print(f"dyn_residency: patched {mod.__name__}.GPUModelRunner", flush=True)
    log.info("dyn_residency: armed every=%d max_swaps=%d ratio=%.1f min=%.0f test=%d",
             EVERY, MAX_SWAPS, RATIO, MIN, TEST_SWAPS)
