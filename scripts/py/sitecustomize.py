"""Mounted with PACK_FILTER=1 or DYN_RESIDENCY=1: load only the tensors the pack's index lists
(and with DYN_RESIDENCY=1 arm dyn_residency.py in the worker).

vLLM streams every tensor of every shard, not just the index's. Two packs need that filtered:
- shards re-laid by exllamav3's util/align_safetensors.py carry "__align_pad__.<shard>.<n>"
  U8 filler tensors (they keep each file contiguous), which no model parameter matches;
- a hard-link overlay that replaces some tensors through extra files plus its own
  model.safetensors.index.json still has the replaced originals inside the shared shards.
Runs the image's own sitecustomize first.
"""
import os
import sys

_ORIG = "/usr/lib/python3.12/sitecustomize.py"
if os.path.isfile(_ORIG):
    exec(compile(open(_ORIG).read(), _ORIG, "exec"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recipe_hooks  # noqa: E402,F401
