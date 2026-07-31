from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from rea_llms.generation import (
    GENERATION_ENGINE_VERSION,
    generate_fixed_length_completions,
    prompt_token_ids,
)
from rea_llms.models import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT,
    DEFAULT_TOKENIZER_NAME,
    load_model_and_tokenizer,
    resolve_device,
    set_seed,
)
from rea_llms.observables import create_observable
from rea_llms.official_direct_data import (
    boxplot_statistics,
    rule_of_three_upper_bound,
    wilson_interval,
)
from rea_llms.scoring import (
    ARI_CONVENTION_CURRENT,
    ARI_CONVENTIONS,
    ari_decoding_description,
    score_completion_tokens,
    validate_ari_convention,
)


CURRENT_DIRECT_SCHEMA_VERSION = 1
DEFAULT_THRESHOLDS = (6.0, 7.0, 8.0, 10.0)


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def prompt_sha256(prompt: str) -> str:
    return _sha256_bytes(prompt.encode("utf-8"))


@dataclass(frozen=True)
class CurrentDirectConfig:
    num_samples: int = 200_000
    batch_size: int = 256
    max_new_tokens: int = 100
    seed: int = 91_000
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    ari_convention: str = ARI_CONVENTION_CURRENT

    def __post_init__(self) -> None:
        if self.num_samples <= 0 or self.batch_size <= 0 or self.max_new_tokens <= 0:
            raise ValueError("sample, batch, and token counts must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.thresholds or any(
            not math.isfinite(float(value)) for value in self.thresholds
        ):
            raise ValueError("thresholds must be a non-empty finite sequence")
        validate_ari_convention(self.ari_convention)


@dataclass
class RunningARIStatistics:
    thresholds: tuple[float, ...]
    count: int = 0
    mean: float = 0.0
    sum_squared_deviations: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def __post_init__(self) -> None:
        self.tail_counts = {float(threshold): 0 for threshold in self.thresholds}

    def update(self, values: np.ndarray) -> None:
        batch = np.asarray(values, dtype=np.float64)
        if batch.ndim != 1 or not np.isfinite(batch).all():
            raise ValueError("ARI batches must be finite one-dimensional arrays")
        if batch.size == 0:
            return
        batch_count = int(batch.size)
        batch_mean = float(batch.mean())
        batch_m2 = float(np.sum((batch - batch_mean) ** 2))
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.sum_squared_deviations += (
            batch_m2 + delta * delta * self.count * batch_count / total
        )
        self.mean += delta * batch_count / total
        self.count = total
        self.minimum = min(self.minimum, float(batch.min()))
        self.maximum = max(self.maximum, float(batch.max()))
        for threshold in self.thresholds:
            self.tail_counts[float(threshold)] += int(np.count_nonzero(batch >= threshold))

    def variance_population(self) -> float:
        return self.sum_squared_deviations / self.count if self.count else math.nan

    def tail_statistics(self) -> list[dict[str, float | int | None]]:
        if self.count <= 0:
            raise ValueError("tail statistics require at least one observation")
        rows: list[dict[str, float | int | None]] = []
        for threshold in self.thresholds:
            count = self.tail_counts[float(threshold)]
            lower, upper = wilson_interval(count, self.count)
            rows.append(
                {
                    "threshold": float(threshold),
                    "count": count,
                    "empirical_probability": count / self.count,
                    "wilson_95_lower": lower,
                    "wilson_95_upper": upper,
                    "rule_of_three_upper": rule_of_three_upper_bound(
                        count, self.count
                    ),
                }
            )
        return rows


def requested_metadata(
    *,
    model_name: str,
    tokenizer_name: str,
    prompt: str,
    config: CurrentDirectConfig,
    device_type: str,
) -> dict[str, Any]:
    """Metadata known before loading a model, used for strict cache reuse."""

    metadata = {
        "schema_version": CURRENT_DIRECT_SCHEMA_VERSION,
        "reference_type": (
            "current_pipeline_iid_direct"
            if config.ari_convention == ARI_CONVENTION_CURRENT
            else "official_public_iid_direct"
        ),
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "prompt": prompt,
        "prompt_sha256": prompt_sha256(prompt),
        "max_new_tokens": config.max_new_tokens,
        "fixed_completion_length": True,
        "eos_stopping": False,
        "temperature": 1.0,
        "top_k": None,
        "top_p": None,
        "generation_engine_version": GENERATION_ENGINE_VERSION,
        "decoding_convention": ari_decoding_description(config.ari_convention),
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": "tokenizer default; not passed explicitly",
        "observable": "textstat automated_readability_index",
        "observable_scope": (
            "exact prompt string plus separately decoded completion tokens"
            if config.ari_convention == ARI_CONVENTION_CURRENT
            else "exact prompt string plus character-sliced completion from full-sequence decoding"
        ),
        "ari_representation": "raw",
        "ari_cap": None,
        "num_samples": config.num_samples,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "thresholds": [float(value) for value in config.thresholds],
        "device_type": device_type,
    }
    # Legacy current-reference fingerprints predate the explicit field. Omitting
    # it for that convention preserves exact reuse of those completed archives.
    if config.ari_convention != ARI_CONVENTION_CURRENT:
        metadata["ari_convention"] = config.ari_convention
    return metadata


def complete_metadata(
    requested: dict[str, Any],
    *,
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
) -> dict[str, Any]:
    metadata = dict(requested)
    metadata.update(
        {
            "model_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
            "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
            "model_parameter_dtype": str(next(model.parameters()).dtype),
            "prompt_token_ids": [int(value) for value in prompt_ids.detach().cpu().tolist()],
            "prompt_token_count": int(prompt_ids.numel()),
            "tokenizer_clean_up_tokenization_spaces": getattr(
                tokenizer,
                "clean_up_tokenization_spaces",
                None,
            ),
            "textstat_version": _package_version("textstat"),
            "transformers_version": _package_version("transformers"),
            "torch_version": torch.__version__,
        }
    )
    return metadata


def metadata_fingerprint(metadata: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(metadata).encode("utf-8"))


def _summary_is_reusable(
    summary: dict[str, Any],
    requested: dict[str, Any],
    ari_path: Path,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    metadata = summary.get("metadata")
    if summary.get("complete") is not True or not isinstance(metadata, dict):
        reasons.append("summary is not marked complete or lacks metadata")
        return False, reasons
    if summary.get("metadata_sha256") != metadata_fingerprint(metadata):
        reasons.append("metadata SHA-256 mismatch")
    for key, value in requested.items():
        if metadata.get(key) != value:
            reasons.append(f"metadata mismatch: {key}")
    if not ari_path.is_file():
        reasons.append("direct_ari.npz is missing")
    elif summary.get("ari_file_sha256") != _sha256_file(ari_path):
        reasons.append("direct_ari.npz SHA-256 mismatch")
    else:
        try:
            with np.load(ari_path, allow_pickle=False) as archive:
                values = archive["raw_ari"]
                if values.shape != (int(requested["num_samples"]),):
                    reasons.append("direct ARI array has the wrong shape")
        except Exception as exc:
            reasons.append(f"direct ARI array is unreadable: {exc}")
    return not reasons, reasons


def load_reusable_summary(
    output_dir: Path,
    requested: dict[str, Any],
) -> dict[str, Any] | None:
    summary_path = output_dir / "summary.json"
    ari_path = output_dir / "direct_ari.npz"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    reusable, _ = _summary_is_reusable(summary, requested, ari_path)
    return summary if reusable else None


def load_scientifically_compatible_summary(
    summary_path: Path,
    *,
    model_name: str,
    tokenizer_name: str,
    prompt: str,
    max_new_tokens: int,
    thresholds: Sequence[float],
    ari_convention: str = ARI_CONVENTION_CURRENT,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Load a complete direct reference matching the active SMC distribution."""

    path = Path(summary_path)
    if not path.is_file():
        return None, ["current-pipeline direct summary is absent"]
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [f"current-pipeline direct summary is unreadable: {exc}"]
    metadata = summary.get("metadata")
    reasons: list[str] = []
    if summary.get("complete") is not True or not isinstance(metadata, dict):
        return None, ["current-pipeline direct summary is incomplete"]
    if summary.get("metadata_sha256") != metadata_fingerprint(metadata):
        reasons.append("current-pipeline direct metadata SHA-256 mismatch")
    validate_ari_convention(ari_convention)
    expected = {
        "reference_type": (
            "current_pipeline_iid_direct"
            if ari_convention == ARI_CONVENTION_CURRENT
            else "official_public_iid_direct"
        ),
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "prompt": prompt,
        "prompt_sha256": prompt_sha256(prompt),
        "max_new_tokens": int(max_new_tokens),
        "fixed_completion_length": True,
        "eos_stopping": False,
        "temperature": 1.0,
        "top_k": None,
        "top_p": None,
        "generation_engine_version": GENERATION_ENGINE_VERSION,
        "decoding_convention": ari_decoding_description(ari_convention),
        "observable": "textstat automated_readability_index",
        "observable_scope": (
            "exact prompt string plus separately decoded completion tokens"
            if ari_convention == ARI_CONVENTION_CURRENT
            else "exact prompt string plus character-sliced completion from full-sequence decoding"
        ),
        "ari_representation": "raw",
        "ari_cap": None,
    }
    if ari_convention != ARI_CONVENTION_CURRENT:
        expected["ari_convention"] = ari_convention
    for key, value in expected.items():
        if metadata.get(key) != value:
            reasons.append(f"current-pipeline direct metadata mismatch: {key}")
    ari_path = path.parent / str(summary.get("ari_file", "direct_ari.npz"))
    if not ari_path.is_file():
        reasons.append("current-pipeline direct ARI archive is absent")
    elif summary.get("ari_file_sha256") != _sha256_file(ari_path):
        reasons.append("current-pipeline direct ARI archive SHA-256 mismatch")
    statistics = summary.get("statistics")
    if not isinstance(statistics, dict) or int(statistics.get("count", 0)) <= 0:
        reasons.append("current-pipeline direct statistics are absent")
    else:
        available = {
            float(row["threshold"])
            for row in statistics.get("tails", [])
            if isinstance(row, dict) and "threshold" in row
        }
        missing = [float(value) for value in thresholds if float(value) not in available]
        if missing:
            reasons.append(f"current-pipeline direct thresholds are absent: {missing}")
    return (summary if not reasons else None), reasons


def sample_current_direct_reference(
    model,
    tokenizer,
    observable,
    *,
    prompt: str,
    config: CurrentDirectConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, RunningARIStatistics]:
    """Generate the requested IID sample count in bounded GPU batches."""

    set_seed(config.seed)
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(config.seed)
    prompt_ids = prompt_token_ids(tokenizer, prompt, device)
    values = np.empty(config.num_samples, dtype=np.float32)
    running = RunningARIStatistics(config.thresholds)
    offset = 0
    with torch.inference_mode():
        while offset < config.num_samples:
            count = min(config.batch_size, config.num_samples - offset)
            tokens = generate_fixed_length_completions(
                model,
                prompt_ids,
                num_sequences=count,
                max_new_tokens=config.max_new_tokens,
                batch_size=count,
                generator=generator,
            )
            _, raw_ari, _ = score_completion_tokens(
                tokenizer,
                observable,
                prompt=prompt,
                prompt_ids=prompt_ids,
                completions=tokens,
            )
            batch_values = raw_ari.detach().cpu().numpy().astype(np.float32, copy=False)
            values[offset : offset + count] = batch_values
            running.update(batch_values)
            offset += count
            if progress_callback is not None:
                progress_callback(offset, config.num_samples)
    return values, running


def _write_npz_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, raw_ari=np.asarray(values, dtype=np.float32))
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_summary(
    values: np.ndarray,
    running: RunningARIStatistics,
    metadata: dict[str, Any],
    *,
    ari_file_sha256: str,
) -> dict[str, Any]:
    if running.count != values.size:
        raise ValueError("running statistics do not cover the stored ARI array")
    return {
        "schema_version": CURRENT_DIRECT_SCHEMA_VERSION,
        "complete": True,
        "metadata": metadata,
        "metadata_sha256": metadata_fingerprint(metadata),
        "ari_file": "direct_ari.npz",
        "ari_file_sha256": ari_file_sha256,
        "statistics": {
            "count": running.count,
            "mean": running.mean,
            "variance_population": running.variance_population(),
            "standard_deviation_population": math.sqrt(running.variance_population()),
            "min": running.minimum,
            "max": running.maximum,
            "boxplot": boxplot_statistics(np.asarray(values, dtype=np.float64)),
            "tails": running.tail_statistics(),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one current-pipeline IID direct reference.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=91_000)
    parser.add_argument("--thresholds", default="6,7,8,10")
    parser.add_argument(
        "--ari-convention",
        choices=ARI_CONVENTIONS,
        default=ARI_CONVENTION_CURRENT,
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    thresholds = tuple(float(item.strip()) for item in args.thresholds.split(",") if item.strip())
    config = CurrentDirectConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        thresholds=thresholds,
        ari_convention=args.ari_convention,
    )
    device = resolve_device(args.device)
    requested = requested_metadata(
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        prompt=args.prompt,
        config=config,
        device_type=device.type,
    )
    output_dir = Path(args.output_dir)
    existing = load_reusable_summary(output_dir, requested)
    if existing is not None and args.skip_completed:
        print(f"reused complete current direct reference: {output_dir.resolve()}")
        return
    if args.skip_completed and (output_dir / "summary.json").exists():
        raise RuntimeError(
            "existing current direct reference is incomplete or metadata-incompatible; "
            "use --no-skip-completed to replace it explicitly"
        )

    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        args.tokenizer_name,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    prompt_ids = prompt_token_ids(tokenizer, args.prompt, next(model.parameters()).device)
    metadata = complete_metadata(
        requested,
        model=model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
    )
    observable = create_observable(
        "ari",
        mode="raw",
        cap=15.0,
        backend="exact_cached",
        convention=config.ari_convention,
        cache_enabled=False,
    )
    progress = tqdm(total=config.num_samples, desc="Current direct reference", unit="sample")
    previous = 0

    def update(done: int, total: int) -> None:
        nonlocal previous
        progress.update(done - previous)
        previous = done

    try:
        values, running = sample_current_direct_reference(
            model,
            tokenizer,
            observable,
            prompt=args.prompt,
            config=config,
            progress_callback=update,
        )
    finally:
        progress.close()
    ari_path = output_dir / "direct_ari.npz"
    _write_npz_atomic(ari_path, values)
    summary = build_summary(
        values,
        running,
        metadata,
        ari_file_sha256=_sha256_file(ari_path),
    )
    _write_json_atomic(output_dir / "summary.json", summary)
    print(f"current direct reference: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
