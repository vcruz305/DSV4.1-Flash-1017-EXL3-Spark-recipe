"""Recipe import hooks: loader filter (PACK_FILTER) and dyn_residency (DYN_RESIDENCY=1).

Installed from zz_recipe_hooks.pth (site-packages) so that every interpreter gets them,
including vLLM's spawned EngineCore, whose environment drops PYTHONPATH (and with it
/recipe-py/sitecustomize.py). Idempotent.
"""
import json
import importlib.abc
import importlib.util
import os
import sys

_PAD = "__align_pad__."
_DIR = os.environ.get("VLLM_EXL3_MODEL_DIR", "")
try:
    with open(os.path.join(_DIR, "model.safetensors.index.json")) as _f:
        _KEEP = frozenset(json.load(_f)["weight_map"])
except Exception:
    _KEEP = None
_TARGETS = {"vllm.model_executor.model_loader.default_loader": "_patch"}
if os.environ.get("DYN_RESIDENCY", "0") == "1":
    _TARGETS["vllm.v1.worker.gpu_model_runner"] = "_patch_runner"
    _TARGETS["vllm.v1.worker.gpu.model_runner"] = "_patch_runner"


def _patch_runner(mod):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dyn_residency

    dyn_residency.patch(mod)


def _patch(mod):
    cls = mod.DefaultModelLoader
    orig = cls._get_weights_iterator

    def _get_weights_iterator(self, *args, **kwargs):
        return ((n, t) for n, t in orig(self, *args, **kwargs)
                if not n.startswith(_PAD) and (_KEEP is None or n in _KEEP))

    cls._get_weights_iterator = _get_weights_iterator


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _TARGETS:
            return None
        fn = globals()[_TARGETS.pop(name)]
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            if _TARGETS:
                sys.meta_path.insert(0, self)
        loader = spec.loader
        exec_module = loader.exec_module

        def _exec(module):
            exec_module(module)
            fn(module)

        loader.exec_module = _exec
        return spec


if not getattr(sys, "_recipe_hooks", False):
    sys._recipe_hooks = True
    sys.meta_path.insert(0, _Finder())
