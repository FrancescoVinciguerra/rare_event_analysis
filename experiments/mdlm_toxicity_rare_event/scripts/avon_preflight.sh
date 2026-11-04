#!/usr/bin/env bash

set -euo pipefail

run() {
  printf '\n$'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

echo "Read-only Avon preflight: no module loads, Python, downloads, or jobs."
run hostname -f
run id
run pwd -P
run readlink -f "${HOME}"
run stat -c '%A %U %G %n' "${HOME}"
run df -hT "${HOME}"
run findmnt -rn -T "${HOME}"
run findmnt -rn -T /gpfs/home/stats/stuhvh
if command -v quota >/dev/null 2>&1; then
  run quota -s
fi
run sinfo -o '%P|%a|%l|%D|%G|%C'
run scontrol show partition
run scontrol show config
if command -v sacctmgr >/dev/null 2>&1; then
  run sacctmgr --readonly show assoc user="${USER}" format=Cluster,Account,User,Partition,QOS
  run sacctmgr --readonly show qos format=Name,MaxWall,MaxTRESPU,Flags
else
  echo "sacctmgr is not available in PATH; record this rather than guessing."
fi
run module spider Python
run module spider CUDA
run module spider cuDNN
run module spider PyTorch
