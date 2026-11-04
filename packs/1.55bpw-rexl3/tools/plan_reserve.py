#!/usr/bin/env python3
"""Size the cold-expert alias from the memory you want left free (RESERVE_GIB).

    python3 packs/1.55bpw-rexl3/tools/plan_reserve.py --reserve-gib 12 [--npz route_stats.npz] [--build]

Before a boot it reads MemAvailable and predicts MemAvailable at ready from a
measured boot of the same pack (packs/1.55bpw-rexl3/configs/reserve-calibration.json):

    ready  = avail_now - (cal.preload_avail - cal.ready_min) - (KVB - cal.kvb)
    cold   = cal.cold_gib + max(0, reserve - ready)

Each extra GiB of expert trellis moved into the alias side file is one GiB the
engine no longer copies into CUDA memory; those experts are read from the mapped
file (page cache, then NVMe) when routed. The coldest experts by a route
histogram (ROUTE_STATS=1 serve, then touch <run>/ctr/route-stats/dump.now) are
chosen first. Without --npz the permanent route record of DYN_RESIDENCY=1 serves
($DYN_RESIDENCY_STATE_DIR/route-heat-<model>.npz, all traffic seen so far) is used
when it exists. Without a histogram only the shipped selection is possible.

MemAvailable counts page cache as free, so it is not a guarantee: the reserve
is only protected if nothing else grows the page cache. Prints the plan; with
--build it writes the side file and $ALIAS_DIR/selection.json (pass that as
COLD_SELECTION=... to serve.sh). Refuses plans over --max-cold-gib.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

GIB = 2**30
HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(HOME))  # repo root: ats_build_alias.py and the
# shipped cold-alias membership are shared with the 1.62 pack and live there


def mem_available_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    raise RuntimeError("no MemAvailable in /proc/meminfo")


def engine_running(name: str) -> bool:
    r = subprocess.run(
        ["sudo", "-n", "docker", "inspect", "-f", "{{.State.Running}}", name],
        capture_output=True, text=True,
    )
    return r.stdout.strip() == "true"


def tenant_peak_hint(reserve_gib: float) -> None:
    """What the tenants actually used: the reserve cgroup's peak (tenant_scope.sh / reserve.slice)."""
    uid = os.getuid()
    cg = os.environ.get("RESERVE_CGROUP",
                        f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/reserve.slice")
    try:
        peak = int(open(os.path.join(cg, "memory.peak"), encoding="utf-8").read())
    except (OSError, ValueError):
        return
    need = max(4.0, 1.5 * peak / GIB)
    print(f"  tenants in {os.path.basename(cg)} peaked at {peak / GIB:.2f} GiB since the slice started "
          f"(its peak resets with it); 1.5x headroom would need RESERVE_GIB >= {need:.0f}"
          + (f", so {reserve_gib:.0f} leaves {reserve_gib - need:.0f} GiB unused" if reserve_gib > need + 1 else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reserve-gib", type=float, default=float(os.environ.get("RESERVE_GIB", "0")))
    ap.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "dsv41-1.62bpw-v3"))
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", ""))
    ap.add_argument("--kvb", type=int, default=int(os.environ.get("KVB", str(GIB))))
    ap.add_argument("--calib", default=os.path.join(HOME, "configs", "reserve-calibration.json"))
    ap.add_argument("--npz", default="", help="route histogram from a ROUTE_STATS=1 serve")
    ap.add_argument("--out", default=os.environ.get("ALIAS_DIR", ""))
    ap.add_argument("--max-cold-gib", type=float, default=15.0)
    ap.add_argument("--avail-gib", type=float, default=None, help="override MemAvailable (testing)")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if not a.npz:
        rec = os.path.join(os.environ.get("DYN_RESIDENCY_STATE_DIR")
                           or os.path.expanduser("~/.cache/dsv41-ds41f/route-heat"),
                           f"route-heat-{a.model_name}.npz")
        if os.path.exists(rec):
            a.npz = rec
            print(f"route histogram: permanent record {rec}")

    cal_all = json.load(open(a.calib))
    if a.model_name not in cal_all:
        print(f"no calibration for {a.model_name} in {a.calib}; boot once with the shipped "
              f"selection and record preload/ready MemAvailable", file=sys.stderr)
        return 1
    cal = cal_all[a.model_name]

    if a.avail_gib is None and engine_running(os.environ.get("CONTAINER", "dsv41-tp1")):
        print("engine container is running: MemAvailable now does not describe a fresh boot. "
              "Stop it first (bash scripts/guard.sh stop) or pass --avail-gib.", file=sys.stderr)
        return 1
    avail = a.avail_gib if a.avail_gib is not None else mem_available_gib()
    used = cal["preload_avail_gib"] - cal["ready_min_gib"]
    kv_delta = (a.kvb - cal["kvb"]) / GIB
    ready = avail - used - kv_delta
    deficit = max(0.0, a.reserve_gib - ready)
    cold = cal["cold_gib"] + deficit

    plan = {
        "model": a.model_name, "reserve_gib": a.reserve_gib, "mem_available_gib": round(avail, 2),
        "engine_use_at_calibration_gib": round(used, 2), "kv_delta_gib": round(kv_delta, 2),
        "predicted_ready_gib": round(ready, 2), "cold_gib": round(cold, 3),
        "extra_cold_gib": round(deficit, 3), "calibration": cal,
    }
    print(f"{avail:.1f} GiB available - {used:.1f} engine (measured, KV {cal['kvb'] / GIB:.2f} GiB)"
          f" - {kv_delta:+.2f} KV change = {ready:.1f} GiB free at ready with the shipped cold set")
    print(f"reserve {a.reserve_gib:.1f} GiB -> cold experts {cold:.2f} GiB "
          f"({cal['cold_gib']:.2f} shipped + {deficit:.2f} extra)")
    rows = sorted(cal.get("measured", []), key=lambda r: (abs(r["cold_gib"] - cold), r["reserve_used_gib"]))
    for r in rows[:1] + [x for x in rows[1:2] if x["cold_gib"] == rows[0]["cold_gib"]]:
        print(f"  measured at {r['cold_gib']:.1f} GiB cold, {r['reserve_used_gib']} GiB of the reserve in use: "
              f"decode {r['decode_tps']} t/s, prefill 2K/7K {r['prefill_2k']}/{r['prefill_7k']} t/s")
    print("  (free memory at ready is mostly the cold experts' page cache: using the reserve "
          "moves them to NVMe, see the in-use row)")
    tenant_peak_hint(a.reserve_gib)
    if cold > a.max_cold_gib:
        print(f"REFUSED: {cold:.1f} GiB cold > --max-cold-gib {a.max_cold_gib}; lower the reserve, "
              f"the KV pool (KVB) or MAXLEN", file=sys.stderr)
        return 2
    if deficit > 0 and not a.npz:
        print("REFUSED: a bigger cold set needs a route histogram (--npz); serve once with "
              "ROUTE_STATS=1", file=sys.stderr)
        return 2
    if not a.build:
        print(json.dumps(plan, indent=1))
        return 0
    if not a.out or not a.model_dir:
        print("--build needs --out/ALIAS_DIR and --model-dir/MODEL_DIR", file=sys.stderr)
        return 1
    cmd = [sys.executable, os.path.join(ROOT, "tools", "ats_build_alias.py"), "--model-dir", a.model_dir,
           "--out", a.out, "--embed", "--align", "64"]
    if deficit > 0:
        cmd += ["--select-npz", a.npz, "--budget-gib", f"{cold:.4f}"]
    else:
        cmd += ["--experts-json", os.path.join(ROOT, "configs", "cold-alias-selection.json")]
    subprocess.run(cmd, check=True)
    man = json.load(open(os.path.join(a.out, "manifest.json")))
    with open(os.path.join(a.out, "selection.json"), "w") as f:
        json.dump(man["experts"], f)
    with open(os.path.join(a.out, "plan.json"), "w") as f:
        json.dump(plan, f, indent=1)
    print(f"serve with: ALIAS_DIR={a.out} COLD_SELECTION={a.out}/selection.json bash packs/1.55bpw-rexl3/scripts/serve.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
