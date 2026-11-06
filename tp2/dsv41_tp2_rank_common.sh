#!/usr/bin/env bash
# DSV4.1 2.50bpw TP2 in-container common env: fabric + engine.
# Recipe engine env (measured) + this rig's fabric env (vendor-recipe pattern).

# ---- fabric (this rig: ETH=enp1s0f0np0, RoCE=rocep1s0f0 only; second twin unwired) ----

# ---- engine (the recipe's measured set, verbatim) ----
export VLLM_EXL3_MIXEDK_MODE=unified
export VLLM_DS41_MARKOV_NVFP4=1
export VLLM_EXL3_L2_PREFETCH=0
export VLLM_DS41_KERNEL_BACKEND=tilelang
export VLLM_DS41_ATTENTION_COMPUTE=auto
export VLLM_DS41_HEAD_DTYPE=float32
export B12X_AUTOTUNE=0
export NCCL_BUFFSIZE=2097152
export NCCL_CUMEM_ENABLE=0
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_P2P_LEVEL=SYS
export NCCL_PROTO="LL,LL128,Simple"
export TORCH_NCCL_USE_COMM_NONBLOCKING=0
export NCCL_DEBUG=INFO
export VLLM_LOGGING_LEVEL=INFO

# ---- serve knobs (recipe defaults; TP2 re-measures these) ----
MODEL_DIR=/models/dsv41-2.50bpw
MAXLEN=${MAXLEN:-524288}
KVB=${KVB:-1073741824}
export VLLM_DS41_ROPE_TRIM_MAX_POS="$MAXLEN"
MNBT=${MNBT:-2112}
LPT=${LPT:-2047}
MTP_K=${MTP_K:-5}
CAPTURE_SIZES=${CAPTURE_SIZES:-1,2,3,4,5,6,7}
CG_MODE=${CG_MODE:-FULL_DECODE_ONLY}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-dsv41}
DEFAULT_MAX_TOKENS=${DEFAULT_MAX_TOKENS:-32768}
DEFAULT_TEMPERATURE=${DEFAULT_TEMPERATURE:-0.6}
DEFAULT_TOP_P=${DEFAULT_TOP_P:-0.95}

gen_json=""
[ "$DEFAULT_MAX_TOKENS" != "0" ] && gen_json="\"max_new_tokens\": $DEFAULT_MAX_TOKENS"
[ -n "$DEFAULT_TEMPERATURE" ] && gen_json="${gen_json:+$gen_json,}\"temperature\": $DEFAULT_TEMPERATURE"
[ -n "$DEFAULT_TOP_P" ] && gen_json="${gen_json:+$gen_json,}\"top_p\": $DEFAULT_TOP_P"
