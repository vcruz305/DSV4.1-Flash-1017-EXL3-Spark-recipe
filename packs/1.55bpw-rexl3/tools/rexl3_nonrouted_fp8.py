"""Transcode REXL3's EXL3 non-routed linears (layers.N.attn.wq_b/wo_b, ffn.shared_experts.w1/w2/w3)
to the block-32 FP8 (E4M3 weight + E8M0 scale per 32x32 block) layout the pack's other non-routed
weights (and mtp.0.*) already use, so the vLLM V4.1 model's FP8-only paths can load them.
Lossy: FP8 re-quantization on top of the EXL3 reconstruction (error printed per tensor)."""
import sys, json, struct, torch
from safetensors import safe_open
from safetensors.torch import save_file
from exllamav3.modules.quant.exl3 import LinearEXL3
src, out = sys.argv[1], sys.argv[2]
idx = json.load(open(f"{src}/model.safetensors.index.json"))["weight_map"]
def get(k):
    with safe_open(f"{src}/{idx[k]}", "pt", device="cuda") as f: return f.get_tensor(k)
# convention check on a native FP8 tensor of the same pack
w8, s8 = get("mtp.0.attn.wo_b.weight"), get("mtp.0.attn.wo_b.scale")
o, i = w8.shape
bmax = w8.float().abs().view(o // 32, 32, i // 32, 32).amax(dim=(1, 3))
print("native block max |fp8|: min %.1f max %.1f, scale range 2^%d..2^%d" % (bmax.min(), bmax.max(),
      int(torch.log2(s8.float()).min()), int(torch.log2(s8.float()).max())), flush=True)
res, errs = {}, []
names = ["attn.wq_b", "attn.wo_b", "ffn.shared_experts.w1", "ffn.shared_experts.w2", "ffn.shared_experts.w3"]
for L in range(40):
    for n in names:
        p = f"layers.{L}.{n}"
        tr = get(p + ".trellis")
        lin = LinearEXL3(config=None, in_features=tr.shape[0] * 16, out_features=tr.shape[1] * 16,
                         suh=get(p + ".suh"), svh=get(p + ".svh"), trellis=tr, mul1=get(p + ".mul1"))
        W = lin.get_weight_tensor().t().float().contiguous()  # [out, in]
        o, i = W.shape
        B = W.view(o // 32, 32, i // 32, 32)
        amax = B.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-30)
        scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
        q = (B / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        deq = q.float() * scale
        errs.append(float((deq - B).norm() / B.norm()))
        res[p + ".weight"] = q.view(o, i).cpu()
        res[p + ".scale"] = scale.view(o // 32, i // 32).to(torch.float8_e8m0fnu).cpu()
        del lin, W, B, q, deq
    print(f"layer {L}: rel err max {max(errs[-5:]):.4f}", flush=True)
print("all: rel err mean %.4f max %.4f" % (sum(errs) / len(errs), max(errs)))
save_file(res, out, metadata={"format": "pt"})
print("saved", out, len(res))
