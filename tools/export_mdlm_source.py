"""Build the standalone MDLM estimator source tree as a ZIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/mdlm_toxicity_rare_event"
PACKAGE = "diffusion_toxicity_smc_experiment"
SOURCE = EXPERIMENT / "src"
CONFIG = EXPERIMENT / "configs/mdlm_owt_unconditional_estimation_100tok_trajectory_mh.yaml"
MODULES = (
    "__init__", "ancestral", "configuration", "estimate", "evaluation", "exploratory_kernel",
    "exploratory_population", "pipeline", "population_checkpoints", "potentials", "runtime", "smc",
)

PYPROJECT = """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "mdlm-rollback-estimator"
version = "0.1.0"
requires-python = ">=3.11"
readme = "README.md"
dependencies = ["numpy>=1.26", "PyYAML>=6", "torch>=2.6", "transformers==4.57.6"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
diffusion_toxicity_smc_experiment = ["estimator.json"]
"""


def export(output: Path) -> None:
    sys.path.insert(0, str(SOURCE))
    from diffusion_toxicity_smc_experiment.configuration import load_yaml

    readme = (EXPERIMENT / "README.md").read_text(encoding="utf-8").split("## Repository layout", 1)[0].rstrip()
    readme += """

## Layout

- `src/diffusion_toxicity_smc_experiment/estimate.py` is the entry point.
- `estimator.json` pins the model revisions and supplies the estimator configuration.
- `pipeline.py` sets up the models and computes the reverse-tilt estimate.
- `exploratory_kernel.py` implements tempering and RollBack–MH.
"""
    sources = {
        f"src/{PACKAGE}/{name}.py": (SOURCE / PACKAGE / f"{name}.py").read_bytes() for name in MODULES
    }
    sources.update({
        f"src/rea_llms/{name}": (ROOT / "rea_llms" / name).read_bytes()
        for name in ("__init__.py", "statistics.py")
    })
    sources.update({
        f"src/{PACKAGE}/estimator.json": (json.dumps(load_yaml(CONFIG), indent=2) + "\n").encode(),
        "README.md": readme.encode(),
        "pyproject.toml": PYPROJECT.encode(),
        ".gitignore": b".venv/\n__pycache__/\n*.egg-info/\nbuild/\ndist/\n",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, contents in sorted(sources.items()):
            archive.writestr(f"mdlm_rollback/{name}", contents)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/mdlm_rollback_source.zip")
    args = parser.parse_args()
    export(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
