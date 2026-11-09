"""Download, load, and verify all pinned benchmark artifacts on a compute node."""

from __future__ import annotations

import argparse
import gc
import os
import platform
from pathlib import Path

import torch

from .configuration import decoder_id, evaluator_id, load_yaml, validate_benchmark_config
from .evaluation import ToxicityEvaluator
from .runtime import load_evaluator_runtime, load_generator_runtime
from .storage import atomic_write_json, initialize_manifest


def _runtime_checks() -> dict[str, object]:
    if platform.python_version_tuple()[:2] != ("3", "11"):
        raise RuntimeError(f"Python 3.11 is required, found {platform.python_version()}")
    if not torch.__version__.startswith("2.6.0"):
        raise RuntimeError(f"PyTorch 2.6.0 is required, found {torch.__version__}")
    if torch.version.cuda is None or not torch.version.cuda.startswith("11.8"):
        raise RuntimeError(
            "the cu118 PyTorch wheel is required, "
            f"found CUDA runtime {torch.version.cuda}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA on this allocated node")
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability < (7, 0):
        raise RuntimeError(f"unsupported GPU compute capability: {capability}")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(capability),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def prefetch(config_path: Path, output_root: Path) -> Path:
    config = load_yaml(config_path)
    validate_benchmark_config(config)
    runtime = _runtime_checks()
    torch.cuda.reset_peak_memory_stats()

    generator = load_generator_runtime(config, local_files_only=False)
    generator_info = {
        "implementation": config["decoder"]["model_implementation"],
        "official_checkpoint": config["decoder"]["checkpoint"],
        "parameters": sum(parameter.numel() for parameter in generator.model.parameters()),
        "dtype": str(next(generator.model.parameters()).dtype),
    }
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    evaluator_runtime = load_evaluator_runtime(config, local_files_only=False)
    numerical_batching = config["evaluation"]["numerical_batching"]
    evaluator = ToxicityEvaluator(
        evaluator_runtime.classifier,
        evaluator_runtime.classifier_tokenizer,
        evaluator_runtime.device,
        positive_label_index=int(config["evaluation"]["classifier"]["positive_label_index"]),
        batch_size=1,
        padding=str(numerical_batching["padding"]),
        pad_final_batch_to_batch_size=bool(
            numerical_batching["pad_final_batch_to_batch_size"]
        ),
    )
    probe = evaluator.score_texts(["pipeline compatibility probe"], max_length=32)[0]
    classifier_info = {
        "classifier": config["evaluation"]["classifier"],
        "parameters": sum(
            parameter.numel() for parameter in evaluator_runtime.classifier.parameters()
        ),
        "dtype": str(next(evaluator_runtime.classifier.parameters()).dtype),
        "probe_logits_finite": all(torch.isfinite(torch.tensor(probe["logits"])).tolist()),
    }
    artifact_dir = (
        output_root
        / "mdlm_owt"
        / "prefetch"
        / decoder_id(config)
        / evaluator_id(config)
    )
    initialize_manifest(
        artifact_dir,
        {
            "kind": "avon_model_prefetch_v1",
            "decoder_id": decoder_id(config),
            "evaluator_id": evaluator_id(config),
            "generator_specification": config["decoder"],
            "evaluator_specification": config["evaluation"],
        },
    )
    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    verification_path = artifact_dir / "verifications" / f"job_{job_id}.json"
    if verification_path.exists():
        raise FileExistsError(f"refusing to overwrite {verification_path}")
    atomic_write_json(
        verification_path,
        {
            "runtime": runtime,
            "generator": generator_info,
            "toxicity_evaluator": classifier_info,
            "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
    )
    return artifact_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(prefetch(args.config, args.output_root))


if __name__ == "__main__":
    main()
