#!/usr/bin/env python3
"""Write the REXL3 overlay's model.safetensors.index.json: the pack's index with every tensor
that was transcoded (head-bf16 / nonrouted-fp8 side files) pointed at its side file and the EXL3
parts it replaces (trellis/suh/svh/mul1) dropped.
Usage: rexl3_overlay_index.py <pack dir> <overlay dir>"""
import json
import os
import struct
import sys

SIDE = ("head-bf16.safetensors", "nonrouted-fp8.safetensors")


def keys(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return [k for k in json.loads(fh.read(n)) if k != "__metadata__"]


src, dst = sys.argv[1], sys.argv[2]
idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
wm = idx["weight_map"]
new = {k: f for f in SIDE if os.path.exists(os.path.join(dst, f)) for k in keys(os.path.join(dst, f))}
gone = {k.rsplit(".", 1)[0] for k in new}
out = {k: v for k, v in wm.items() if k.rsplit(".", 1)[0] not in gone}
out.update(new)
idx["weight_map"] = dict(sorted(out.items()))
tmp = os.path.join(dst, "model.safetensors.index.json.tmp")
with open(tmp, "w") as fh:
    json.dump(idx, fh, indent=2)
os.replace(tmp, os.path.join(dst, "model.safetensors.index.json"))
print(f"index: {len(wm) - len(out) + len(new)} tensors dropped, {len(new)} from side files")
