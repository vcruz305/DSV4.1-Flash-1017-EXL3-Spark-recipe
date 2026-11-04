#!/bin/bash
# Run a non-engine program (Hermes, tools, dashboards) inside the memory the planner reserved.
#
#   RESERVE_GIB=12 bash packs/1.55bpw-rexl3/scripts/tenant_scope.sh <unit-name> <command...>
#
# The engine cannot be capped from its own cgroup: its CUDA allocations are not charged
# to the container's memcg (measured: memory.current ~4.5 GiB with ~100 GiB allocated).
# So the other tenants get the cap instead: MemoryHigh at 90 % of the reserve (reclaim
# starts), MemoryMax at the reserve (OOM inside the scope, not in the engine), no swap.
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: RESERVE_GIB=<n> $0 <unit-name> <command...>" >&2; exit 1; }
R="${RESERVE_GIB:?set RESERVE_GIB}"
unit="$1"; shift
max=$(python3 -c "print(int($R*2**30))")
high=$(python3 -c "print(int($R*0.9*2**30))")
exec systemd-run --user --scope --unit "$unit" -p MemoryHigh="$high" -p MemoryMax="$max" -p MemorySwapMax=0 "$@"
