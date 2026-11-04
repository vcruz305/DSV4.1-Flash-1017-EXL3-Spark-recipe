#!/usr/bin/env python3
"""Re-pick the SSD cold set from the permanent route record before a boot (serve.sh AUTO_COLD=1).

  python3 tools/auto_cold.py --alias-dir DIR --model-dir DIR --record FILE [--min-gain 0.002]

Same byte budget as the current side file (its plan.json cold_gib, else its size). The coldest
experts by the record are chosen (ats_build_alias --select-only). Both sets are scored by the
expected bytes one decode step reads from cold experts, sum of q_e * bytes_e with
q_e = 1 - (1 - g_e)**7 (MTP layers included). The picker's coldest-q-first fill is the
optimum of exactly that for a byte budget (fractional knapsack: cost q_e*bytes_e, weight
bytes_e), so this is the metric it is built for. The side file is rebuilt only when the
re-pick cuts it by more than --min-gain (default 5 %), because a rebuild writes the whole file
(~18 GiB, ~25 s). The cold experts touched per step and the cold-route share are printed too;
they are different metrics and can move the other way (experts range 4.2-16.9 MiB). The new file is built next to
the old one and swapped in with a rename, so a failed build leaves the old set serving."""
import argparse, json, os, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--alias-dir", required=True)
ap.add_argument("--model-dir", required=True)
ap.add_argument("--record", required=True)
ap.add_argument("--min-gain", type=float, default=float(os.environ.get("AUTO_COLD_MIN_GAIN", "0.05")))
a = ap.parse_args()


def scores(sel, z, nbytes, rows_per_step=7):
    """(expected MiB read per step, cold experts touched per step, cold-route share)."""
    c = z["counts_0"]
    tok = z["rows_0"] if "rows_0" in z.files else z["calls_0"]
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    off = int(meta.get("mtp_offset", 40))
    mib = touched = cold = 0.0
    for k, v in sel.items():
        kind, i = k.split(".")
        r = int(i) + (off if kind == "mtp" else 0)
        g = c[r, v] / max(float(tok[r]), 1.0)
        q = 1 - (1 - g) ** rows_per_step
        touched += float(q.sum())
        mib += float(sum(qq * nbytes.get((k, e), 0) for qq, e in zip(q, v))) / 2**20
        if kind == "layers":
            cold += float(c[r, v].sum())
    return mib, touched, cold / max(float(c[:off].sum()), 1.0)


def done(decision, **kw):
    """One line for dashboards: <record dir>/auto_cold-last.json."""
    try:
        out = os.path.join(os.path.dirname(a.record), "auto_cold-last.json")
        with open(out + ".tmp", "w") as fh:
            json.dump({"time": time.time(), "decision": decision, "alias_dir": a.alias_dir, **kw}, fh)
        os.replace(out + ".tmp", out)
    except OSError:
        pass
    sys.exit(0)


if not os.path.exists(a.record):
    print(f"auto_cold: no record at {a.record} yet; keeping {a.alias_dir}")
    done("no-record")
z = np.load(a.record)
c = z["counts_0"]
man = json.load(open(os.path.join(a.alias_dir, "manifest.json")))
plan = os.path.join(a.alias_dir, "plan.json")
gib = json.load(open(plan)).get("cold_gib") if os.path.exists(plan) else None
gib = float(os.environ.get("AUTO_COLD_GIB") or gib or man["bytes"] / 2**30)
tmp = a.alias_dir.rstrip("/") + ".auto"
shutil.rmtree(tmp, ignore_errors=True)
base = [sys.executable, os.path.join(HERE, "ats_build_alias.py"), "--model-dir", a.model_dir,
        "--out", tmp, "--embed", "--align", "64", "--select-npz", a.record, "--budget-gib", f"{gib:.4f}"]
subprocess.run(base + ["--select-only"], check=True, stdout=subprocess.DEVNULL)
new = json.load(open(os.path.join(tmp, "selection.json")))
sys.path.insert(0, HERE)
from ats_build_alias import _scan_expert_bytes  # noqa: E402
wm = json.load(open(os.path.join(a.model_dir, "model.safetensors.index.json")))["weight_map"]
nbytes = {k: v[0] for k, v in _scan_expert_bytes(a.model_dir, wm).items()}
(cur, cur_t, cur_sh), (nxt, nxt_t, nxt_sh) = scores(man["experts"], z, nbytes), scores(new, z, nbytes)
gain = (cur - nxt) / max(cur, 1e-9)
kw = dict(routes=int(c.sum()), gib=round(gib, 2), mib_now=round(cur, 2), mib_new=round(nxt, 2),
          touched_now=round(cur_t, 3), touched_new=round(nxt_t, 3), gain=round(gain, 4),
          share_now=round(cur_sh, 5), share_new=round(nxt_sh, 5), min_gain=a.min_gain)
print(f"auto_cold: record {int(c.sum())} routes; cold reads per step {cur:.1f} MiB now, {nxt:.1f} re-picked "
      f"({gain:+.1%}); experts touched {cur_t:.2f} -> {nxt_t:.2f}; cold-route share {cur_sh:.3%} -> {nxt_sh:.3%} "
      f"({gib:.2f} GiB)")
if gain <= a.min_gain:
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"auto_cold: gain {gain:.1%} <= {a.min_gain:.1%}; keeping the current side file")
    done("kept", **kw)
need = int(gib * 2**30 * 1.05) + (2 << 30)
free = shutil.disk_usage(os.path.dirname(tmp) or ".").free
if free < need:
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"auto_cold: only {free / 2**30:.1f} GiB free beside {a.alias_dir}, a rebuild needs "
          f"{need / 2**30:.1f}; keeping the current side file")
    done("no-disk-space", free_gib=round(free / 2**30, 1), **kw)
subprocess.run(base, check=True, stdout=subprocess.DEVNULL)
m2 = json.load(open(os.path.join(tmp, "manifest.json")))
json.dump(m2["experts"], open(os.path.join(tmp, "selection.json"), "w"))
if os.path.exists(plan):
    shutil.copy(plan, os.path.join(tmp, "plan.json"))
old = a.alias_dir.rstrip("/") + ".prev"
shutil.rmtree(old, ignore_errors=True)
os.rename(a.alias_dir, old)
os.rename(tmp, a.alias_dir)
shutil.rmtree(old, ignore_errors=True)
print(f"auto_cold: rebuilt {a.alias_dir} from the record")
done("rebuilt", **kw)
