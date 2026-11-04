#!/usr/bin/env python3
"""Share of expert routes that land in a cold set, on a held-out route histogram.

  python3 packs/1.55bpw-rexl3/tools/cold_share_eval.py --eval after.npz [--minus before.npz] sel1.json [sel2.json ...]
  python3 packs/1.55bpw-rexl3/tools/cold_share_eval.py --diff after.npz before.npz out.npz   # write a delta histogram

Lower is better: each cold route is a read through the ATS alias (SSD when evicted)."""
import json, sys
import numpy as np


def load(p):
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def delta(a, b):
    out = dict(a)
    for k in ("counts_0", "calls_0", "rows_0"):
        out[k] = a[k].astype("float64") - (b[k].astype("float64") if b is not None else 0)
    return out


def main(argv):
    if argv[:1] == ["--diff"]:
        a, b, o = argv[1:4]
        np.savez(o, **delta(load(a), load(b)))
        print("wrote", o)
        return 0
    ev = argv[argv.index("--eval") + 1]
    mi = argv[argv.index("--minus") + 1] if "--minus" in argv else None
    sels = [x for i, x in enumerate(argv) if x.endswith(".json")]
    d = delta(load(ev), load(mi) if mi else None)
    c = d["counts_0"]
    off = json.loads(str(d["meta"])).get("mtp_offset", 40) if "meta" in d else 40
    main_tot = c[:off].sum()
    print(f"eval routes (main layers): {int(main_tot)}  tokens/row: {int(d['rows_0'][0])}")
    for s in sels:
        sel = json.load(open(s))
        cold = 0.0
        n = 0
        for lay, eids in sel.items():
            if not lay.startswith("layers."):
                continue
            r = int(lay.split(".")[1])
            cold += c[r, eids].sum()
            n += len(eids)
        print(f"{s}: {n} cold experts (main layers), cold share {cold / main_tot:.4%}")
    return 0


sys.exit(main(sys.argv[1:]))
