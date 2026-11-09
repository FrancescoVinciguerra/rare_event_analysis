#!/usr/bin/env bash

set -euo pipefail

avon_assert_experiment_package() {
  python - <<'PY'
import os
from pathlib import Path

import diffusion_toxicity_smc_experiment as package

expected_root = (Path(os.environ["EXPERIMENT_DIR"]) / "src").resolve()
package_file = Path(package.__file__).resolve()
if expected_root not in package_file.parents:
    raise RuntimeError(
        "wrong diffusion_toxicity_smc_experiment import: "
        f"expected a module below {expected_root}, got {package_file}"
    )
print(f"experiment_package_file={package_file}")
PY
}

avon_activate_environment() {
  : "${SLURM_JOB_ID:?This script must run inside a SLURM allocation}"
  : "${PROJECT_DIR:?Set PROJECT_DIR}"
  : "${VENV_DIR:?Set VENV_DIR}"
  : "${HF_HOME:?Set HF_HOME}"
  : "${AVON_MODULES:?Set AVON_MODULES from the verified module spider output}"
  EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_DIR}/mdlm}"

  if ! type module >/dev/null 2>&1; then
    echo "The environment-modules command is unavailable; stop without guessing an initialization path." >&2
    exit 1
  fi
  module purge
  read -r -a avon_module_list <<< "${AVON_MODULES}"
  module load "${avon_module_list[@]}"
  source "${VENV_DIR}/bin/activate"
  cd "${PROJECT_DIR}"

  export EXPERIMENT_DIR
  # `python -m` normally prepends the current working directory.  The Avon
  # repository also contains a historical package with the same import name,
  # so make the selected experiment source tree exclusive and fail closed if
  # import resolution ever changes again.
  export PYTHONSAFEPATH=1
  export PYTHONPATH="${EXPERIMENT_DIR}/src:${PROJECT_DIR}/shared:${PROJECT_DIR}"
  export HF_HOME
  export HF_HUB_CACHE="${HF_HOME}/hub"
  export TOKENIZERS_PARALLELISM=false
  export PYTHONUNBUFFERED=1
  avon_assert_experiment_package
}

avon_print_diagnostics() {
  echo "hostname=$(hostname -f)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "project_dir=$(pwd -P)"
  echo "experiment_dir=${EXPERIMENT_DIR}"
  echo "hf_home=${HF_HOME}"
  echo "modules=${AVON_MODULES}"
  nvidia-smi
  python - <<'PY'
import json
import platform
import torch

payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}
if torch.cuda.is_available():
    payload.update({
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    })
print(json.dumps(payload, sort_keys=True))
PY
}

avon_print_cpu_diagnostics() {
  echo "hostname=$(hostname -f)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "project_dir=$(pwd -P)"
  echo "experiment_dir=${EXPERIMENT_DIR}"
  echo "hf_home=${HF_HOME}"
  echo "modules=${AVON_MODULES}"
  python - <<'PY'
import json
import platform
import torch

print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}, sort_keys=True))
PY
}
