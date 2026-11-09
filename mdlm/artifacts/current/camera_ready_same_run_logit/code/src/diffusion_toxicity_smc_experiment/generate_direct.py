"""Generate persistent IID unconditional samples from the native MDLM decoder."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch

from .ancestral import NativeMDLMSampler
from .configuration import (
    decoder_id,
    decoder_manifest,
    evaluator_id,
    load_yaml,
    validate_benchmark_config,
)
from .runtime import load_generator_runtime
from .storage import (
    atomic_write_json,
    atomic_write_jsonl,
    initialize_manifest,
    missing_indices,
    scan_direct_generations,
    stable_sample_id,
)


def direct_run_directory(config: dict[str, Any], output_root: Path | None = None) -> Path:
    root = output_root or Path(config.get("outputs", {}).get("root", "outputs"))
    return root / "mdlm_owt" / "direct" / decoder_id(config)


def _shard_seed(*, base_seed: int, decoder: str, sample_indices: list[int]) -> int:
    payload = f"{base_seed}:{decoder}:{','.join(map(str, sample_indices))}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:15], 16)


def _autocast_metadata(device: torch.device) -> dict[str, Any]:
    try:
        enabled = bool(torch.is_autocast_enabled(device.type))
    except TypeError:  # Compatibility with older Torch APIs.
        enabled = bool(torch.is_autocast_enabled())
    if not enabled:
        return {"enabled": False, "dtype": None}
    try:
        dtype = torch.get_autocast_dtype(device.type)
    except (AttributeError, TypeError):
        dtype = torch.get_autocast_gpu_dtype()
    return {"enabled": True, "dtype": str(dtype)}


def _prefetched_classifier_dtype(
    config: dict[str, Any], output_root: Path
) -> tuple[str | None, str | None]:
    prefetch_decoder_dir = (
        output_root
        / "mdlm_owt"
        / "prefetch"
        / decoder_id(config)
    )
    verification_dir = prefetch_decoder_dir / evaluator_id(config) / "verifications"
    candidates = sorted(
        verification_dir.glob("job_*.json"),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    expected_classifier = config["evaluation"]["classifier"]
    fallback_candidates = sorted(
        prefetch_decoder_dir.glob("*/verifications/job_*.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )
    for source in reversed(candidates):
        payload = json.loads(source.read_text(encoding="utf-8"))
        dtype = payload.get("toxicity_evaluator", {}).get("dtype")
        if dtype is not None:
            return str(dtype), str(source)
    # Evaluator IDs include numerical batching.  A batching-only update does not
    # invalidate the already verified classifier checkpoint or its dtype.
    for source in fallback_candidates:
        payload = json.loads(source.read_text(encoding="utf-8"))
        evaluator = payload.get("toxicity_evaluator", {})
        if evaluator.get("classifier") == expected_classifier and evaluator.get("dtype"):
            return str(evaluator["dtype"]), str(source)
    return None, None


def _performance_artifact_path(run_dir: Path) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if job_id:
        suffix = f"job_{job_id}"
        if task_id is not None:
            suffix += f"_task_{task_id}"
    else:
        suffix = dt.datetime.now(dt.timezone.utc).strftime("manual_%Y%m%dT%H%M%S%fZ")
    return run_dir / "performance" / f"{suffix}.json"


def generate_missing(
    config: dict[str, Any],
    *,
    target_samples: int,
    output_root: Path | None,
    task_index: int,
    num_tasks: int,
    allow_network: bool,
) -> Path:
    total_started = time.perf_counter()
    validate_benchmark_config(config)
    decoder = config["decoder"]
    if decoder["generation_mode"] != "unconditional":
        raise ValueError("the current direct benchmark requires unconditional generation")
    identifier = decoder_id(config)
    run_dir = direct_run_directory(config, output_root)
    initialize_manifest(
        run_dir,
        {
            "decoder_id": identifier,
            "decoder_manifest": decoder_manifest(config),
            "conditioning_mode": "unconditional",
            "storage_schema": "unconditional_direct_generation_shards_v1",
        },
    )
    inventory = scan_direct_generations(run_dir, identifier)
    reused_samples = sum(index < target_samples for index in inventory.sample_indices)
    pending = missing_indices(
        inventory,
        target=target_samples,
        task_index=task_index,
        num_tasks=num_tasks,
    )
    if not pending:
        print(
            json.dumps(
                {
                    "kind": "direct_generation_resume_noop_v1",
                    "decoder_id": identifier,
                    "target_samples": target_samples,
                    "completion_samples_reused": reused_samples,
                    "completion_samples_generated": 0,
                    "total_seconds": time.perf_counter() - total_started,
                },
                sort_keys=True,
            )
        )
        return run_dir

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    runtime = load_generator_runtime(config, local_files_only=not allow_network)
    direct = config.get("direct", {})
    shard_size = int(direct.get("shard_size", 32))
    microbatch_size = int(direct.get("generation_microbatch_size", 4))
    if shard_size <= 0 or microbatch_size <= 0:
        raise ValueError("shard_size and generation_microbatch_size must be positive")
    sampler = NativeMDLMSampler(
        runtime.model,
        mask_token_id=int(decoder["mask_token_id"]),
        predictor=str(decoder["predictor"]),
        noise_epsilon=float(decoder["noise_schedule"]["epsilon"]),
        sampling_epsilon=float(decoder["sampling_epsilon"]),
        final_noise_removal=bool(decoder["final_noise_removal"]),
        final_noise_removal_method=str(decoder["final_noise_removal_method"]),
    )
    fixed_prefix = torch.tensor(
        decoder["fixed_prefix_token_ids"], dtype=torch.long, device=runtime.device
    )
    if fixed_prefix.numel() != 0:
        raise ValueError("unconditional generation cannot have fixed prefix tokens")
    base_seed = int(direct.get("base_seed", 8128))
    sampling_seconds = 0.0
    decoding_seconds = 0.0
    serialization_seconds = 0.0
    total_nfe = 0
    generated_samples = 0

    for offset in range(0, len(pending), shard_size):
        indices = pending[offset : offset + shard_size]
        seed = _shard_seed(
            base_seed=base_seed, decoder=identifier, sample_indices=indices
        )
        generator = torch.Generator(device=runtime.device.type).manual_seed(seed)
        sampler.reset_nfe()
        sampling_started = time.perf_counter()
        terminal_batches = []
        for batch_start in range(0, len(indices), microbatch_size):
            batch_count = min(microbatch_size, len(indices) - batch_start)
            terminal_batches.append(
                sampler.sample(
                    batch_size=batch_count,
                    sequence_length=int(decoder["sequence_length"]),
                    num_steps=int(decoder["denoising_steps"]),
                    fixed_prefix_token_ids=fixed_prefix,
                    device=runtime.device,
                    generator=generator,
                ).cpu()
            )
        sampling_elapsed = time.perf_counter() - sampling_started
        sampling_seconds += sampling_elapsed
        terminal = torch.cat(terminal_batches, dim=0)
        if bool((terminal == int(decoder["mask_token_id"])).any().item()):
            raise RuntimeError("unconditional direct generation ended with [MASK]")
        decoding_started = time.perf_counter()
        decoded = runtime.tokenizer.batch_decode(
            terminal.tolist(), skip_special_tokens=True
        )
        decoding_elapsed = time.perf_counter() - decoding_started
        decoding_seconds += decoding_elapsed
        elapsed = sampling_elapsed + decoding_elapsed
        total_nfe += sampler.nfe
        generated_samples += len(indices)
        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        rows = []
        for position, (sample_index, tokens, text) in enumerate(
            zip(indices, terminal, decoded)
        ):
            token_ids = [int(value) for value in tokens.tolist()]
            rows.append(
                {
                    "sample_id": stable_sample_id(
                        decoder_id=identifier,
                        sample_index=sample_index,
                        shard_seed=seed,
                    ),
                    "sample_index": sample_index,
                    "conditioning_mode": "unconditional",
                    "shard_seed": seed,
                    "position_in_shard": position,
                    "token_ids": token_ids,
                    "completion_token_ids": token_ids,
                    "decoded_text": text,
                    "checkpoint_revision": decoder["checkpoint"]["revision"],
                    "tokenizer_revision": decoder["tokenizer"]["revision"],
                    "decoder_id": identifier,
                    "generated_at_utc": timestamp,
                    "nfe_for_shard": sampler.nfe,
                    "generation_runtime_seconds_for_shard": elapsed,
                    "peak_cuda_memory_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated(runtime.device))
                        if runtime.device.type == "cuda"
                        else 0
                    ),
                }
            )
        shard = run_dir / "generations" / (
            f"shard_{indices[0]:08d}_{indices[-1]:08d}.jsonl"
        )
        serialization_started = time.perf_counter()
        atomic_write_jsonl(shard, rows)
        serialization_seconds += time.perf_counter() - serialization_started

    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    final_inventory = scan_direct_generations(run_dir, identifier)
    target_indices = {index for index in final_inventory.sample_indices if index < target_samples}
    if target_indices != set(range(target_samples)):
        raise RuntimeError(
            "direct generation did not produce exactly one sample for every requested index"
        )
    residual_mask_count = sum(
        int(token_id == int(decoder["mask_token_id"]))
        for row in final_inventory.rows
        if int(row["sample_index"]) < target_samples
        for token_id in row["completion_token_ids"]
    )
    if residual_mask_count:
        raise RuntimeError("persisted direct generations contain residual [MASK] tokens")

    output_root_resolved = output_root or Path(config.get("outputs", {}).get("root", "outputs"))
    classifier_dtype, classifier_dtype_source = _prefetched_classifier_dtype(
        config, output_root_resolved
    )
    num_steps = int(decoder["denoising_steps"])
    particle_steps = generated_samples * num_steps
    terminal_particle_evaluations = (
        generated_samples if bool(decoder["final_noise_removal"]) else 0
    )
    total_seconds = time.perf_counter() - total_started
    performance = {
        "kind": "direct_generation_performance_v1",
        "decoder_id": identifier,
        "conditioning_mode": decoder["generation_mode"],
        "initial_state": f"[MASK]^{int(decoder['sequence_length'])}",
        "denoising_steps": num_steps,
        "terminal_sampling": decoder["final_noise_removal_method"],
        "model_dtype_actual": str(next(runtime.model.parameters()).dtype),
        "classifier_dtype_actual": classifier_dtype,
        "classifier_dtype_source": classifier_dtype_source,
        "autocast": _autocast_metadata(runtime.device),
        "generation_microbatch_size_effective": microbatch_size,
        "model_load_seconds": runtime.model_load_seconds,
        "tokenizer_load_seconds": runtime.tokenizer_load_seconds,
        "sampling_seconds": sampling_seconds,
        "decoding_seconds": decoding_seconds,
        "serialization_seconds": serialization_seconds,
        "total_seconds": total_seconds,
        "completion_samples_reused": reused_samples,
        "completion_samples_generated": generated_samples,
        "completion_samples_target": target_samples,
        "completion_samples_validated_after": len(target_indices),
        "unique_sample_ids_after": len(
            {
                row["sample_id"]
                for row in final_inventory.rows
                if int(row["sample_index"]) < target_samples
            }
        ),
        "nfe_model_forward_calls": total_nfe,
        "particle_steps_processed": particle_steps,
        "terminal_particle_evaluations": terminal_particle_evaluations,
        "completions_per_sampling_second": generated_samples / sampling_seconds,
        "particle_steps_per_second": particle_steps / sampling_seconds,
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(runtime.device))
            if runtime.device.type == "cuda"
            else 0
        ),
        "peak_cuda_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(runtime.device))
            if runtime.device.type == "cuda"
            else 0
        ),
        "residual_mask_token_count": residual_mask_count,
        "nonfinite_detected": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(runtime.device)
            if runtime.device.type == "cuda"
            else None
        ),
    }
    numeric_metrics = [
        value
        for value in performance.values()
        if isinstance(value, float)
    ]
    if not all(math.isfinite(value) for value in numeric_metrics):
        performance["nonfinite_detected"] = True
        raise FloatingPointError("non-finite direct-generation performance metric")
    performance_path = _performance_artifact_path(run_dir)
    if performance_path.exists():
        raise FileExistsError(f"refusing to overwrite {performance_path}")
    atomic_write_json(performance_path, performance)
    print(f"performance_artifact={performance_path}")
    print(f"direct_generation_performance={json.dumps(performance, sort_keys=True)}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-samples", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    target = args.target_samples
    if target is None:
        configured_target = config["direct"].get("target_samples")
        if configured_target is None:
            parser.error("--target-samples is required when the config has no smoke target")
        target = int(configured_target)
    task_index = args.task_index
    if task_index is None:
        task_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    run_dir = generate_missing(
        config,
        target_samples=target,
        output_root=args.output_root,
        task_index=task_index,
        num_tasks=args.num_tasks,
        allow_network=args.allow_network,
    )
    print(run_dir)
    if torch.cuda.is_available():
        print(f"peak_cuda_memory_allocated_bytes={torch.cuda.max_memory_allocated()}")


if __name__ == "__main__":
    main()
