"""Dequantize REXL3's EXL3 head (head.trellis/suh/svh/mul1) to BF16 head.weight [vocab, hidden] for vLLM,
whose DSV4.1 ParallelLMHead takes no quant_config."""
import sys, json, torch
from safetensors import safe_open
from safetensors.torch import save_file
from exllamav3.modules.quant.exl3 import LinearEXL3
src, out = sys.argv[1], sys.argv[2]
idx = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
t = {}
for k in ("head.trellis", "head.suh", "head.svh", "head.mul1"):
    with safe_open(f"{src}/{idx[k]}", "pt", device="cuda") as f: t[k.split(".")[1]] = f.get_tensor(k)
print({k: (tuple(v.shape), v.dtype) for k, v in t.items()})
tr = t["trellis"]; inf, outf = tr.shape[0] * 16, tr.shape[1] * 16
lin = LinearEXL3(config=None, in_features=inf, out_features=outf, suh=t["suh"], svh=t["svh"], trellis=tr, mul1=t["mul1"])
w = lin.get_weight_tensor()  # [in, out] half
print("w", tuple(w.shape), w.dtype, float(w.abs().max()), bool(torch.isfinite(w).all()))
save_file({"head.weight": w.t().contiguous().to(torch.bfloat16).cpu()}, out, metadata={"format": "pt"})
print("saved", out)
