#!/usr/bin/env bash
# self-healing ray install
if ! python3 -c "import ray" 2>/dev/null; then
  echo 151.101.64.223 pypi.org >> /etc/hosts
  echo 151.101.64.223 files.pythonhosted.org >> /etc/hosts
  python3 -m pip install --no-cache-dir -q ray==2.59.0 ||     python3 -m pip install --no-cache-dir -q ray
fi
python3 -c "import ray" 2>/dev/null || { echo "[boot] ray install FAILED"; exit 3; }

# rank1 worker (5b84): join the head's ray cluster over the fabric; self-healing.
source /root/launch/dsv41_tp2_rank_common.sh
while true; do
  ray stop --force 2>/dev/null
  ray start --address="10.99.200.15:29501" --node-ip-address=10.99.200.12 --block 2>&1
  echo "[rank1] ray block exited rc=$?; retrying in 15s"
  sleep 15
done
