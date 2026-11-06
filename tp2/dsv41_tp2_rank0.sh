#!/usr/bin/env bash
# self-healing ray install
if ! python3 -c "import ray" 2>/dev/null; then
  echo 151.101.64.223 pypi.org >> /etc/hosts
  echo 151.101.64.223 files.pythonhosted.org >> /etc/hosts
  python3 -m pip install --no-cache-dir -q ray==2.59.0 ||     python3 -m pip install --no-cache-dir -q ray
fi
python3 -c "import ray" 2>/dev/null || { echo "[boot] ray install FAILED"; exit 3; }

# rank0 head (e98b): ray head on the fabric; wait for worker; vllm serve TP2.
source /root/launch/dsv41_tp2_rank_common.sh
export SPARKNET_ROCE_GID_INDEX=5
export VLLM_HOST_IP=10.99.200.15
echo "10.99.200.12 gx10-5b84" >> /etc/hosts
export NCCL_IB_GID_INDEX=5
ray stop --force 2>/dev/null
ray start --head --port=29501 --node-ip-address=10.99.200.15 --include-dashboard=false 2>&1
echo "[rank0] ray head rc=$?; waiting for worker..."
JOINED=0
for i in $(seq 1 60); do
  if ray status 2>/dev/null | grep -q "2.0 GPU"; then JOINED=1; echo "[rank0] poll $i: BOTH GPUs registered"; break; fi
  echo "[rank0] poll $i: not yet"
  sleep 10
done
if [ "$JOINED" != 1 ]; then echo "[rank0] WARNING: proceeding without confirmed worker"; fi
echo "[rank0] launching vllm serve TP2"
exec vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 --port 8000 \
  --tokenizer-mode deepseek_v41 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --distributed-executor-backend ray \
  --quantization exl3_dsv41 \
  --block-size 256 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-model-len "$MAXLEN" \
  --max-num-seqs 1 \
  --max-parallel-prefills 1 \
  --max-num-batched-tokens "$MNBT" \
  --gpu-memory-utilization 0.90 \
  --kv-cache-memory-bytes "$KVB" \
  --generation-config vllm --override-generation-config "{$gen_json}" \
  --engram-config '{"table_memory":"disk","disk_resident_scales":false,"projection_tp":true}' \
  --attention-backend tilelang \
  --linear-backend tilelang \
  --moe-backend tilelang \
  --jit-monitor-mode warn \
  --compilation-config "{\"cudagraph_mode\":\"$CG_MODE\",\"custom_ops\":[\"all\"],\"cudagraph_capture_sizes\":[$CAPTURE_SIZES]}" \
  --max-cudagraph-capture-size "${CAPTURE_SIZES##*,}" \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$MTP_K,\"draft_tensor_parallel_size\":2,\"attention_backend\":\"tilelang\",\"draft_sample_method\":\"greedy\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":true,\"adaptive_verification_row_cost\":0.01}" \
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
