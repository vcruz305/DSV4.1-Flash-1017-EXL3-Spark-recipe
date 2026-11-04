#!/usr/bin/env python3
"""hf-overrides for the REXL3 1.55bpw pack in this image (experimental).

The V4.1 model here takes only block-32 FP8 for attn.wo_b (and FP8 for the other
non-routed linears), so REXL3's EXL3 wq_b/wo_b/shared experts are transcoded to that
layout first (packs/1.55bpw-rexl3/tools/rexl3_nonrouted_fp8.py, run with the exllamav3 fork) and the EXL3 head to BF16
(packs/1.55bpw-rexl3/tools/rexl3_head_bf16.py), in a hard-link overlay of the pack. What is left in EXL3 is the
routed experts, which REXL3 stores with the mul1 codebook.
Usage: HF_OVERRIDES_JSON="$(python3 packs/1.55bpw-rexl3/tools/rexl3_overrides.py)"
"""
import json
q = {"quant_method": "exl3", "bits": 2, "codebook": "mul1", "scope": "deepseek_v41_routed_experts",
     "non_routed_quantization": {"quant_method": "deepseek_v41_fp8", "weight_block_size": [32, 32],
                                 "is_checkpoint_fp8_serialized": True, "activation_scheme": "dynamic"},
     "mtp_experts": "exl3"}
print(json.dumps({"quantization_config": q}, separators=(",", ":")))
