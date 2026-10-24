#!/usr/bin/env bash
# Host-side TP2 pair launcher for the DSV4.1 2.50bpw pack.
# Env injection follows the vendor two-Spark recipe (launch_qwen38.sh): ALL fabric
# vars ride on docker -e so every process in the container (ray actors included)
# inherits them — vllm's ray env whitelist cannot strip them.
# Usage: launch_dsv41_tp2.sh rank0   (on e98b)
#        launch_dsv41_tp2.sh rank1   (on 5b84)
# Requires: docker access; image dsv41-1017-exl3:r6w built.
set -eu
ROLE=${1:?rank0|rank1}
IMAGE=${IMAGE:-dsv41-1017-exl3:r6w}
CONTAINER=dsv41-tp2-$ROLE
ETH_IF=enp1s0f0np0
IB_IF=rocep1s0f0
HEAD_IP=10.99.200.15
MASTER_PORT=29501
HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
case "$ROLE" in
  rank0) INNER=dsv41_tp2_rank0.sh ;;
  rank1) INNER=dsv41_tp2_rank1.sh ;;
esac
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" \
  --privileged --gpus all --ipc host --network host \
  --shm-size=16g \
  --ulimit core=0 \
  --ulimit nofile=1048576:1048576 --ulimit memlock=-1 --ulimit stack=67108864 \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  -e HF_TOKEN \
  -e MN_IF_NAME=$ETH_IF \
  -e UCX_NET_DEVICES=$ETH_IF \
  -e NCCL_SOCKET_IFNAME=$ETH_IF \
  -e NCCL_IB_HCA=$IB_IF \
  -e NCCL_IB_DISABLE=0 \
  -e OMPI_MCA_btl_tcp_if_include=$ETH_IF \
  -e GLOO_SOCKET_IFNAME=$ETH_IF \
  -e TP_SOCKET_IFNAME=$ETH_IF \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e HEAD_IP=$HEAD_IP \
  -e MASTER_PORT=$MASTER_PORT \
  -e VLLM_EXL3_MODEL_DIR=/models/dsv41-2.50bpw -e MAXLEN=1048576 -e KVB=2147483648 \
  -e VLLM_EXL3_TRIM_AFTER_LAYER=1 \
  -e VLLM_EXL3_MOE_KERNEL=exllamav3 \
  -e VLLM_EXL3_V41_NATIVE_MOE=0 \
  -e VLLM_EXL3_MIXEDK_LEGACY=1 \
  -e VLLM_DS41_ROPE_TRIM=1 \
  -e DSV41_DSML_STOP=1 \
  -e DSV41_DSML_PARAM_REPAIR=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900 \
  -v /home/victor/models:/models:ro \
  -v /home/victor/.cache/huggingface/hub:/hfhub:ro \
  -v /home/victor/.cache/dsv41-ds41f:/cache:rw \
  -v /home/victor/dsv41_tp2_launch:/root/launch:ro \
  --entrypoint /bin/bash \
  "$IMAGE" "/root/launch/$INNER"
echo "launched $CONTAINER"
