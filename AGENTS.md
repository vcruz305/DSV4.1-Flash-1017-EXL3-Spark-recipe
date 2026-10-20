# DSV4.1-Flash-1017 EXL3 serving recipe (single Spark)

Serve vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw on ONE NVIDIA DGX Spark
using the derived vLLM image this repo builds. Read the README first: the
measured flag set, the cold-expert alias and the memory guard are not optional.

Do NOT:
- drop the cold-expert alias (`VLLM_EXL3_ATS_ALIAS=0`) and expect a boot: the
  pack does not fit resident on 121.7 GiB
- build the exllamav3 fork without `MOE_TILESIZE_K=16`: you get the 32-row tile,
  ~34 % slower per step (190.6 vs 142.0 ms), and none of the README numbers apply
- raise `--max-num-batched-tokens` to 8192 without re-measuring: it OOMs
- enable `B12X_AUTOTUNE` as a "helpful" default (hours of bring-up); prefix caching
  is on by default and measured (README "Prefix caching"), do not turn it off by default
- run the container without `seccomp=unconfined --security-opt label=disable
  --ulimit memlock=-1`: the Engram reader's io_uring init fails
- serve without `scripts/guard.sh` armed (0.5 GiB MemAvailable floor, 64 MiB
  swap-growth allowance)
- print or commit `HF_TOKEN`

Exact commands: see README "Quick start". Verify a serve with
`bash scripts/smoke.sh` (must contain "Paris").
