#!/bin/bash
# In-container serve script. This is the measured invocation; every flag below
# was in the run that produced the numbers in README.md.
#
# Do not add --enforce-eager (that is the 74.4 ms/token no-draft path) and do not
# raise --max-num-batched-tokens past 2112 (8192 does not fit; see docs/ENGINEERING_NOTES.md).
set -euo pipefail

MODEL_DIR="${VLLM_EXL3_MODEL_DIR:-/models/dsv41-1.62bpw-v3}"
MAXLEN="${MAXLEN:-524288}"
KVB="${KVB:-1073741824}"
# The shim trims the RoPE cos/sin cache to the served length at load. Pin the trim to
# MAXLEN explicitly so it can never fall back to a shorter default than --max-model-len.
export VLLM_DS41_ROPE_TRIM_MAX_POS="${VLLM_DS41_ROPE_TRIM_MAX_POS:-$MAXLEN}"
MNBT="${MNBT:-2112}"
LPT="${LPT:-2047}"
MTP_K="${MTP_K:-5}"
CAPTURE_SIZES="${CAPTURE_SIZES:-1,2,3,4,5,6,7}"
CG_MODE="${CG_MODE:-FULL_DECODE_ONLY}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
# Prefix caching (default on) reuses the KV of a repeated prompt prefix (multi-turn chat,
# agents, follow-up questions on one long document); PREFIX_CACHING=0 turns it off.
# See README "Prefix caching".
PCF=--enable-prefix-caching
[ "${PREFIX_CACHING:-1}" = "0" ] && PCF=--no-enable-prefix-caching

echo "dsv41 serve: model=$MODEL_DIR maxlen=$MAXLEN kv=$KVB mnbt=$MNBT lpt=$LPT mtp_k=$MTP_K graphs=$CG_MODE prefix_caching=${PREFIX_CACHING:-1}"

exec vllm serve "$MODEL_DIR" \
    --served-model-name dsv41 \
    --host 0.0.0.0 --port 8000 \
    --tokenizer-mode deepseek_v41 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --distributed-executor-backend uni \
    --quantization exl3_dsv41 \
    --block-size 256 \
    $PCF \
    --enable-chunked-prefill \
    --max-model-len "$MAXLEN" \
    --max-num-seqs 1 \
    --max-parallel-prefills 1 \
    --max-num-batched-tokens "$MNBT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv-cache-memory-bytes "$KVB" \
    --generation-config vllm \
    --engram-config '{"table_memory":"disk","disk_resident_scales":false,"projection_tp":true}' \
    --attention-backend tilelang \
    --linear-backend tilelang \
    --moe-backend tilelang \
    --jit-monitor-mode warn \
    --compilation-config "{\"cudagraph_mode\":\"$CG_MODE\",\"custom_ops\":[\"all\"],\"cudagraph_capture_sizes\":[$CAPTURE_SIZES]}" \
    --max-cudagraph-capture-size "${CAPTURE_SIZES##*,}" \
    --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$MTP_K,\"draft_tensor_parallel_size\":1,\"attention_backend\":\"tilelang\",\"draft_sample_method\":\"greedy\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":true,\"adaptive_verification_row_cost\":0.01}" \
    --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --long-prefill-token-threshold "$LPT" \
    --async-scheduling \
    --no-scheduler-reserve-full-isl \
    --no-enable-flashinfer-autotune \
    --language-model-only \
    --mm-processor-cache-gb 0 \
    --hf-overrides '{"quantization_config":{"quant_method":"exl3","bits":2,"codebook":"mcg","scope":"deepseek_v41_routed_experts","non_routed_quantization":{"quant_method":"deepseek_v41_fp8","weight_block_size":[32,32],"is_checkpoint_fp8_serialized":true,"activation_scheme":"dynamic"},"mtp_experts":"exl3"}}'