# Two-Spark TP2 deployment (DSV4.1-Flash-1017 DERISKED EXL3 2.50bpw)

This directory adds a measured two-Spark (2× DGX Spark, GB10) TP2 deployment
of the `vcruz305/DSV4.1-Flash-1017-DERISKED-EXL3-2.50bpw` pack on top of the
single-Spark recipe in this repository. Everything below was measured on
the live deployment on 2026-10-22.

## Results

| measurement | value |
|---|---|
| sixcat standard (vendor policy `deepseek-v4-flash-dspark`, thinking on) | **89.17% (107/120)** |
| sixcat speed, single-stream best-sustained (p90) | **64.00 tok/s** (p50 63.09 / p95 64.21 / max 64.88) |
| serving-knee aggregate @ C=1 | 58.45 tok/s |
| bench7 hard-class mean (this repo's bench) | 44.15 tok/s at 91.8 ms/step, 4.258 accepted tok/step |
| long-context needle probes | found at 480K and 950K depth |
| prefill at depth | 1404 tok/s (480K) / 1273 tok/s (950K) |
| decode at depth | 50.15 tok/s (480K) / 46.22 tok/s (950K) |
| served context ceiling | **1048576** (the model's rated YaRN ceiling), verified |

## What the two-Spark deployment changes

- **TP2 sharding**: the pack's 15 shards plus the 2 head shards split across
  both boxes over the RoCEv2 fabric; Ray as the distributed executor.
- **Context ceiling**: `MAXLEN=1048576`, `KVB=2147483648` (the single-Spark
  recipe's 32K/512K caps are one-box memory caps, not model limits).
- **The trellis factorized vocab head ships in the pack** (`head.suh/svh/
  trellis/mul1`; there is no dense embed): it MUST load through the shim's
  head install. The shim silently skips that install unless
  `VLLM_EXL3_MODEL_DIR` is set — the boot then completes and serves a
  headless model that emits one identical placeholder token forever, with
  no error anywhere. All ten shim-gating variables ride on docker `-e`.
- **Per-rank RoCE GID indices are host-specific** and mandatory: index 5 on
  the rank-0 box, index 3 on the rank-1 box. A wrong index stalls silently
  inside NCCL init. `NCCL_SOCKET_IFNAME` is never pinned.
- **`VLLM_HOST_IP` must be pinned to the fabric IP per host**, and the peer
  hostname must resolve via /etc/hosts on both sides; without them the
  distributed stack deadlocks in `shm_broadcast.wait_until_ready`.
- **Ray self-install per boot**: the launcher's `rm -f` destroys the
  writable layer each launch; both rank scripts self-heal the ray install
  before `exec vllm serve`.
- The single-Spark recipe's cold-expert side file is NOT used at TP2: the
  full pack is sharded. Note the fleet limit — 2×121 GB boxes cannot hold
  the 386 GB pack resident, so expert streaming remains load-bearing.

## Files

- `launch_dsv41_tp2.sh` — host-side launcher; run `rank0` on the first box
  and `rank1` on the second. All fabric/shim env rides on docker `-e`.
- `dsv41_tp2_rank0.sh` — ray head: cluster start, join check, serve flags.
- `dsv41_tp2_rank1.sh` — ray worker: self-healing join loop.
- `dsv41_tp2_rank_common.sh` — shared NCCL tuning and serve-knob defaults.

## What was measured and rejected

- **Alias-subsystem kill**: void on this fleet (242 GB total vs a 386 GB
  pack); expert streaming cannot be removed at TP2.
- **MTP k re-sweep**: rejected — per-position draft acceptance is ~85%,
  so the draft quality, not k, is the ceiling; k=5 stands.
- **No further engine-side headroom**: the 91.8 ms step is compute-bound at
  this pack's quantization.

The full campaign record, including the raw sixcat receipts, lives in
`REXL3/REXL3-dev` under `status/tp2-dflash/dsv41_spark_tp2/`.
