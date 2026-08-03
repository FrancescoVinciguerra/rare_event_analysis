from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rea_llms.scoring import (
    ARI_CONVENTION_CURRENT,
    ARI_CONVENTIONS,
)


AUDIT_CATEGORIES = (
    "exactly_reusable",
    "comparable_but_not_cache_equivalent",
    "incompatible",
    "unverifiable",
)

SCIENTIFIC_METADATA_KEYS = (
    "algorithm",
    "mutation_kernel",
    "model_name",
    "tokenizer_name",
    "prompt",
    "max_new_tokens",
    "proposal_temperature",
    "generation_temperature",
    "fixed_completion_length",
    "ari_mode",
    "ari_cap",
    "final_evaluation",
    "observable_backend",
    "ari_convention",
    "lambda_final",
    "resampling",
    "resampling_method",
    "resample_final",
)

COMPUTATIONAL_METADATA_KEYS = (
    "algorithm_version",
    "lambdas",
    "num_particles",
    "mcmc_steps_per_level",
    "ess_threshold",
    "suffix_mode",
    "suffix_sizes",
    "suffix_rhos",
    "require_global_irreducibility",
    "batch_size",
    "seed_group",
    "component_index",
    "seed",
)

REQUIRED_METADATA_KEYS = tuple(
    key
    for key in SCIENTIFIC_METADATA_KEYS + COMPUTATIONAL_METADATA_KEYS
    if key != "ari_convention"
)


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def prompt_sha256(prompt: str) -> str:
    """Hash the exact UTF-8 prompt without any normalization."""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def infer_ari_convention(metadata: Mapping[str, Any]) -> str | None:
    """Infer only conventions whose legacy provenance is known by this repository."""

    explicit = metadata.get("ari_convention")
    if explicit in ARI_CONVENTIONS:
        return str(explicit)
    reference_type = metadata.get("reference_type")
    if reference_type == "current_pipeline_iid_direct":
        return ARI_CONVENTION_CURRENT
    if reference_type == "official_public_iid_direct":
        return "official_public"
    if metadata.get("algorithm") in {
        "suffix_direct_mh_smc",
        "official_annealed_tps",
    } and metadata.get("observable_backend") in {
        "exact",
        "exact_cached",
        "textstat",
    }:
        return ARI_CONVENTION_CURRENT
    return None


@dataclass(frozen=True)
class ArchiveFacts:
    path: Path
    metadata: dict[str, Any] | None
    keys: tuple[str, ...]
    raw_ari: np.ndarray | None
    final_log_weights: np.ndarray | None
    log_normalizing_constant: float | None
    has_phi: bool
    has_completions: bool
    runtime_seconds: float | None = None
    generated_token_count: int | None = None
    read_error: str | None = None


@dataclass(frozen=True)
class ArchiveAuditRecord:
    facts: ArchiveFacts
    classification: str
    reasons: tuple[str, ...]
    expected_run_id: str | None = None
    selected_for_mixture: bool = False

    def to_row(self) -> dict[str, Any]:
        metadata = self.facts.metadata or {}
        prompt = metadata.get("prompt")
        raw_count = None if self.facts.raw_ari is None else int(self.facts.raw_ari.size)
        weight_count = (
            None
            if self.facts.final_log_weights is None
            else int(self.facts.final_log_weights.size)
        )

        def encoded(key: str) -> str | None:
            value = metadata.get(key)
            if value is None:
                return None
            return canonical_json(value) if isinstance(value, (list, dict)) else str(value)

        return {
            "path": str(self.facts.path),
            "run_id": self.facts.path.stem,
            "expected_run_id": self.expected_run_id,
            "classification": self.classification,
            "selected_for_mixture": self.selected_for_mixture,
            "reasons": "; ".join(self.reasons),
            "archive_format": self.facts.path.suffix.lower(),
            "archive_keys": canonical_json(list(self.facts.keys)),
            "read_error": self.facts.read_error,
            "algorithm": metadata.get("algorithm"),
            "code_version": metadata.get("algorithm_version"),
            "git_commit": metadata.get("repository_commit"),
            "model_name": metadata.get("model_name"),
            "model_revision": metadata.get("model_revision"),
            "tokenizer_name": metadata.get("tokenizer_name"),
            "tokenizer_revision": metadata.get("tokenizer_revision"),
            "prompt": prompt,
            "prompt_repr": repr(prompt) if isinstance(prompt, str) else None,
            "prompt_sha256": prompt_sha256(prompt) if isinstance(prompt, str) else None,
            "prompt_token_ids": encoded("prompt_token_ids"),
            "prompt_token_count": metadata.get("prompt_token_count"),
            "max_new_tokens": metadata.get("max_new_tokens"),
            "fixed_completion_length": metadata.get("fixed_completion_length"),
            "eos_policy": metadata.get("eos_policy"),
            "pad_token_id": metadata.get("pad_token_id"),
            "generation_temperature": metadata.get("generation_temperature"),
            "top_k": metadata.get("top_k"),
            "top_p": metadata.get("top_p"),
            "observable_backend": metadata.get("observable_backend"),
            "ari_convention": infer_ari_convention(metadata),
            "ari_mode": metadata.get("ari_mode"),
            "ari_cap": metadata.get("ari_cap"),
            "lambda_final": metadata.get("lambda_final"),
            "lambda_schedule": encoded("lambdas"),
            "num_particles": metadata.get("num_particles"),
            "mcmc_steps_per_level": metadata.get("mcmc_steps_per_level"),
            "ess_threshold": metadata.get("ess_threshold"),
            "suffix_mode": metadata.get("suffix_mode"),
            "suffix_sizes": encoded("suffix_sizes"),
            "suffix_rhos": encoded("suffix_rhos"),
            "seed": metadata.get("seed"),
            "seed_group": metadata.get("seed_group"),
            "batch_size": metadata.get("batch_size"),
            "has_raw_ari": self.facts.raw_ari is not None,
            "raw_ari_count": raw_count,
            "has_final_smc_weights": self.facts.final_log_weights is not None,
            "final_weight_count": weight_count,
            "has_log_z": self.facts.log_normalizing_constant is not None,
            "log_z_smc": self.facts.log_normalizing_constant,
            "has_completions": self.facts.has_completions,
            "runtime_seconds": self.facts.runtime_seconds,
            "generated_token_count": self.facts.generated_token_count,
            "resample_final": metadata.get("resample_final"),
            "run_complete": _archive_is_complete(self.facts),
        }


def discover_component_archives(root: Path) -> list[Path]:
    """Find candidate run archives without interpreting their filenames."""

    if not root.exists():
        return []
    supported = {".npz", ".json", ".jsonl", ".parquet"}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in supported
    )


def _metadata_from_json_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("metadata"), dict):
        return dict(payload["metadata"])
    if isinstance(payload.get("metadata_json"), str):
        value = json.loads(payload["metadata_json"])
        return dict(value) if isinstance(value, dict) else None
    if isinstance(payload.get("config"), dict):
        metadata = dict(payload["config"])
        for key in (
            "algorithm",
            "algorithm_version",
            "repository_commit",
            "seed",
            "seed_group",
        ):
            if key in payload:
                metadata[key] = payload[key]
        return metadata
    return None


def read_archive_facts(path: Path) -> ArchiveFacts:
    """Read only internally stored metadata and indispensable estimator arrays."""

    try:
        suffix = path.suffix.lower()
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                keys = tuple(sorted(archive.files))
                metadata = None
                if "metadata_json" in archive:
                    parsed = json.loads(str(archive["metadata_json"].item()))
                    metadata = dict(parsed) if isinstance(parsed, dict) else None
                raw = (
                    np.asarray(archive["raw_ari"], dtype=np.float64)
                    if "raw_ari" in archive
                    else None
                )
                weights = (
                    np.asarray(archive["final_log_weights"], dtype=np.float64)
                    if "final_log_weights" in archive
                    else None
                )
                log_z = (
                    float(np.asarray(archive["log_normalizing_constant"]).item())
                    if "log_normalizing_constant" in archive
                    else None
                )
                runtime = (
                    float(np.asarray(archive["runtime_seconds"]).item())
                    if "runtime_seconds" in archive
                    else None
                )
                generated_token_count = (
                    int(np.asarray(archive["generated_token_count"]).item())
                    if "generated_token_count" in archive
                    else None
                )
                has_completions = any(
                    key in archive
                    for key in ("completion", "completions", "completion_token_ids")
                )
                return ArchiveFacts(
                    path=path,
                    metadata=metadata,
                    keys=keys,
                    raw_ari=raw,
                    final_log_weights=weights,
                    log_normalizing_constant=log_z,
                    has_phi="phi" in archive,
                    has_completions=has_completions,
                    runtime_seconds=runtime,
                    generated_token_count=generated_token_count,
                )
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            keys = tuple(sorted(payload)) if isinstance(payload, dict) else ()
            metadata = _metadata_from_json_payload(payload)
            raw = None
            weights = None
            log_z = None
            runtime = None
            if isinstance(payload, dict):
                if isinstance(payload.get("raw_ari"), list):
                    raw = np.asarray(payload["raw_ari"], dtype=np.float64)
                if isinstance(payload.get("final_log_weights"), list):
                    weights = np.asarray(payload["final_log_weights"], dtype=np.float64)
                if payload.get("log_normalizing_constant") is not None:
                    log_z = float(payload["log_normalizing_constant"])
                runtime_value = payload.get("runtime_total_seconds", payload.get("runtime_seconds"))
                if runtime_value is not None:
                    runtime = float(runtime_value)
            return ArchiveFacts(
                path=path,
                metadata=metadata,
                keys=keys,
                raw_ari=raw,
                final_log_weights=weights,
                log_normalizing_constant=log_z,
                has_phi=isinstance(payload, dict) and "phi" in payload,
                has_completions=isinstance(payload, dict)
                and any(key in payload for key in ("completion", "completions", "completion_token_ids")),
                runtime_seconds=runtime,
            )
        return ArchiveFacts(
            path=path,
            metadata=None,
            keys=(),
            raw_ari=None,
            final_log_weights=None,
            log_normalizing_constant=None,
            has_phi=False,
            has_completions=False,
            read_error=f"metadata reader is not available for {suffix}",
        )
    except Exception as exc:
        return ArchiveFacts(
            path=path,
            metadata=None,
            keys=(),
            raw_ari=None,
            final_log_weights=None,
            log_normalizing_constant=None,
            has_phi=False,
            has_completions=False,
            read_error=f"{type(exc).__name__}: {exc}",
        )


def _archive_is_complete(facts: ArchiveFacts) -> bool:
    metadata = facts.metadata
    if metadata is None or facts.read_error is not None:
        return False
    raw = facts.raw_ari
    weights = facts.final_log_weights
    log_z = facts.log_normalizing_constant
    if raw is None or weights is None or log_z is None:
        return False
    expected_count = metadata.get("num_particles")
    if not isinstance(expected_count, int) or expected_count <= 0:
        return False
    if raw.shape != (expected_count,) or weights.shape != raw.shape:
        return False
    if not np.isfinite(raw).all():
        return False
    valid_weights = (~np.isnan(weights)) & (~np.isposinf(weights))
    if not valid_weights.all() or not np.isfinite(weights).any():
        return False
    return math.isfinite(log_z) and metadata.get("resample_final") is True


def _mismatched_keys(
    metadata: Mapping[str, Any],
    expected: Mapping[str, Any],
    keys: Sequence[str],
) -> list[str]:
    def value(source: Mapping[str, Any], key: str) -> Any:
        return infer_ari_convention(source) if key == "ari_convention" else source.get(key)

    return [key for key in keys if value(metadata, key) != value(expected, key)]


def classify_component_archives(
    paths: Sequence[Path],
    expected_metadata_by_run_id: Mapping[str, Mapping[str, Any]],
) -> list[ArchiveAuditRecord]:
    """Classify archives and select only unique, exact production matches."""

    expected_items = [
        (str(run_id), dict(metadata))
        for run_id, metadata in expected_metadata_by_run_id.items()
    ]
    records: list[ArchiveAuditRecord] = []
    for path in sorted(Path(item) for item in paths):
        facts = read_archive_facts(path)
        metadata = facts.metadata
        if facts.read_error is not None:
            records.append(
                ArchiveAuditRecord(facts, "unverifiable", (facts.read_error,))
            )
            continue
        if metadata is None:
            records.append(
                ArchiveAuditRecord(
                    facts,
                    "unverifiable",
                    ("essential metadata are absent",),
                )
            )
            continue
        missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
        if missing:
            records.append(
                ArchiveAuditRecord(
                    facts,
                    "unverifiable",
                    ("missing metadata: " + ", ".join(missing),),
                )
            )
            continue
        if not _archive_is_complete(facts):
            records.append(
                ArchiveAuditRecord(
                    facts,
                    "incompatible",
                    ("run is incomplete or lacks raw ARI, final SMC weights, or log Z",),
                )
            )
            continue

        exact = [
            (run_id, expected)
            for run_id, expected in expected_items
            if canonical_json(metadata) == canonical_json(expected)
        ]
        if exact:
            records.append(
                ArchiveAuditRecord(
                    facts,
                    "exactly_reusable",
                    ("all scientific and cache metadata match",),
                    expected_run_id=exact[0][0],
                    selected_for_mixture=True,
                )
            )
            continue

        scientifically_matching = [
            (run_id, expected)
            for run_id, expected in expected_items
            if not _mismatched_keys(metadata, expected, SCIENTIFIC_METADATA_KEYS)
        ]
        if scientifically_matching:
            run_id, expected = scientifically_matching[0]
            differences = _mismatched_keys(
                metadata,
                expected,
                COMPUTATIONAL_METADATA_KEYS,
            )
            records.append(
                ArchiveAuditRecord(
                    facts,
                    "comparable_but_not_cache_equivalent",
                    ("cache metadata differ: " + ", ".join(differences),),
                    expected_run_id=run_id,
                )
            )
            continue

        mismatch_sets = [
            _mismatched_keys(metadata, expected, SCIENTIFIC_METADATA_KEYS)
            for _, expected in expected_items
        ]
        closest = min(mismatch_sets, key=len) if mismatch_sets else ["no planned component"]
        records.append(
            ArchiveAuditRecord(
                facts,
                "incompatible",
                ("scientific metadata differ: " + ", ".join(closest),),
            )
        )

    exact_by_seed: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        if not record.selected_for_mixture or record.facts.metadata is None:
            continue
        exact_by_seed.setdefault(int(record.facts.metadata["seed"]), []).append(index)
    for indices in exact_by_seed.values():
        if len(indices) <= 1:
            continue
        indices.sort(
            key=lambda index: (
                records[index].facts.path.stem != records[index].expected_run_id,
                str(records[index].facts.path),
            )
        )
        for index in indices[1:]:
            record = records[index]
            records[index] = replace(
                record,
                classification="incompatible",
                reasons=record.reasons + ("duplicate seed/run content",),
                selected_for_mixture=False,
            )
    return records


def selected_archive_paths(records: Sequence[ArchiveAuditRecord]) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for record in records:
        if not record.selected_for_mixture:
            continue
        if record.expected_run_id is None:
            raise AssertionError("selected archive has no expected run ID")
        if record.expected_run_id in selected:
            raise ValueError(f"duplicate selected run ID: {record.expected_run_id}")
        selected[record.expected_run_id] = record.facts.path
    return selected


REUSE_DIMENSIONS = (
    "reusable_for_probability_estimation",
    "reusable_for_mbar",
    "reusable_for_temporal_ess",
    "reusable_for_runtime",
    "reusable_for_compute_count",
    "reusable_for_rescoring_phi",
    "reusable_for_official_comparison",
)


def assess_data_reuse(
    record: Mapping[str, Any],
    *,
    target_ari_convention: str | None = None,
) -> dict[str, Any]:
    """Classify one artifact along independent scientific reuse dimensions."""

    if (
        target_ari_convention is not None
        and target_ari_convention not in ARI_CONVENTIONS
    ):
        raise ValueError(f"unknown target ARI convention: {target_ari_convention!r}")
    artifact_type = str(record.get("artifact_type", "unknown"))
    convention = record.get("ari_convention")
    if convention is None:
        convention = infer_ari_convention(record)
    convention_matches = (
        target_ari_convention is None or convention == target_ari_convention
    )
    has_phi = bool(record.get("has_phi") or record.get("has_raw_ari"))
    has_rescore_source = bool(
        record.get("has_completion_text")
        or record.get("has_completions")
        or record.get("has_token_ids")
    )
    has_mbar_layout = bool(
        has_phi
        and (
            record.get("known_lambda_zero")
            or (
                record.get("has_lambda_schedule")
                and record.get("has_level_labels")
            )
        )
    )
    is_tps = artifact_type in {"tps", "official_tps_archive", "tps_run"}
    is_smc = artifact_type in {"smc", "smc_component", "smc_run"}
    is_direct = artifact_type in {"direct", "official_direct_archive", "direct_run"}

    if is_direct:
        probability_ok = has_phi and convention_matches
        probability_reason = (
            "IID observable values match the requested convention"
            if probability_ok
            else "direct observable values are absent or use another/unverified convention"
        )
    elif is_smc:
        probability_ok = bool(
            has_phi
            and record.get("has_final_smc_weights")
            and record.get("has_log_z")
            and convention_matches
        )
        probability_reason = (
            "final SMC weights, log Z, and observable convention match"
            if probability_ok
            else "SMC weights/log Z are incomplete or were built with another convention"
        )
    elif is_tps:
        probability_ok = has_mbar_layout and convention_matches
        probability_reason = (
            "ordered level-labelled TPS values can contribute to MBAR"
            if probability_ok
            else "TPS layout or observable convention is insufficient for MBAR"
        )
    else:
        probability_ok = False
        probability_reason = "artifact type is not recognized as direct, SMC, or TPS"

    mbar_ok = bool(has_mbar_layout and (is_tps or is_direct) and convention_matches)
    temporal_ok = bool(is_tps and has_phi and record.get("has_chain_order"))
    runtime_ok = bool(
        record.get("has_runtime")
        and record.get("has_hardware_metadata")
        and record.get("has_configuration_metadata")
    )
    exact_compute = bool(
        record.get("has_generated_token_count")
        or record.get("has_suffix_lengths")
        or record.get("has_cut_points")
        or record.get("has_fixed_length_and_count")
    )
    approximate_compute = bool(
        not exact_compute and record.get("has_suffix_distribution")
    )
    compute_ok = exact_compute or approximate_compute
    official_ok = bool(
        has_phi
        and convention in {"official_public", "official_paper_ari_unpublished"}
    )

    decisions = {
        "reusable_for_probability_estimation": (
            probability_ok,
            probability_reason,
        ),
        "reusable_for_mbar": (
            mbar_ok,
            (
                "phi, lambda membership, and sample counts are available"
                if mbar_ok
                else "MBAR needs matching phi plus known lambda/level membership"
            ),
        ),
        "reusable_for_temporal_ess": (
            temporal_ok,
            (
                "ordered values are preserved within TPS chains"
                if temporal_ok
                else "temporal ESS requires ordered values within each chain"
            ),
        ),
        "reusable_for_runtime": (
            runtime_ok,
            (
                "runtime, hardware, and configuration metadata are present"
                if runtime_ok
                else "comparable runtime requires runtime, hardware, and full configuration"
            ),
        ),
        "reusable_for_compute_count": (
            compute_ok,
            (
                "exact generated-token count can be reconstructed"
                if exact_compute
                else "only an expected token count can be reconstructed from the suffix law"
                if approximate_compute
                else "token counts, cut points, suffix lengths, and sufficient fixed-length metadata are absent"
            ),
        ),
        "reusable_for_rescoring_phi": (
            has_rescore_source,
            (
                "completion text or token IDs are available"
                if has_rescore_source
                else "stored scalar ARI values cannot be converted to another convention"
            ),
        ),
        "reusable_for_official_comparison": (
            official_ok,
            (
                "artifact uses a declared official convention"
                if official_ok
                else "artifact uses current ARI or its official convention is unverified"
            ),
        ),
    }
    result = {
        "ari_convention": convention,
        "ari_target_match": convention_matches,
    }
    for key, (value, reason) in decisions.items():
        result[key] = bool(value)
        result[f"reason_{key}"] = reason
    return result


def _has_hardware_metadata(metadata: Mapping[str, Any]) -> bool:
    return any(
        metadata.get(key)
        for key in ("gpu_name", "device_name", "hardware", "cpu_model")
    )


def _has_configuration_metadata(metadata: Mapping[str, Any]) -> bool:
    return all(
        metadata.get(key) is not None
        for key in ("model_name", "tokenizer_name", "prompt", "max_new_tokens")
    )


def _metadata_inventory(metadata: Mapping[str, Any]) -> dict[str, Any]:
    prompt = metadata.get("prompt")
    return {
        "model_name": metadata.get("model_name", metadata.get("model")),
        "model_revision": metadata.get("model_revision"),
        "tokenizer_name": metadata.get("tokenizer_name", metadata.get("tokenizer")),
        "tokenizer_revision": metadata.get("tokenizer_revision"),
        "prompt_repr": repr(prompt) if isinstance(prompt, str) else None,
        "prompt_sha256": (
            prompt_sha256(prompt) if isinstance(prompt, str) else None
        ),
        "prompt_token_ids": metadata.get("prompt_token_ids"),
        "prompt_token_count": metadata.get("prompt_token_count"),
        "max_new_tokens": metadata.get(
            "max_new_tokens",
            metadata.get("completion_length"),
        ),
        "fixed_completion_length": metadata.get("fixed_completion_length"),
        "eos_stopping": metadata.get("eos_stopping"),
        "generation_temperature": metadata.get(
            "generation_temperature",
            metadata.get("temperature"),
        ),
        "top_k": metadata.get("top_k"),
        "top_p": metadata.get("top_p"),
        "decoding_convention": metadata.get("decoding_convention"),
        "ari_mode": metadata.get("ari_mode"),
        "ari_cap": metadata.get("ari_cap"),
        "lambda_schedule": metadata.get(
            "lambdas",
            metadata.get("lambda_schedule"),
        ),
        "lambda_final": metadata.get("lambda_final"),
        "num_particles": metadata.get("num_particles"),
        "mcmc_steps_per_level": metadata.get("mcmc_steps_per_level"),
        "seed": metadata.get("seed"),
        "seed_group": metadata.get("seed_group"),
        "suffix_sizes": metadata.get("suffix_sizes"),
        "suffix_rhos": metadata.get("suffix_rhos"),
        "batch_size": metadata.get("batch_size"),
        "hardware": metadata.get(
            "gpu_name",
            metadata.get("device_name", metadata.get("hardware")),
        ),
    }


def _first_jsonl_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                return payload if isinstance(payload, dict) else None
    return None


def audit_run_directory(
    root: Path,
    *,
    target_ari_convention: str = ARI_CONVENTION_CURRENT,
) -> list[dict[str, Any]]:
    """Read known Drive run artifacts and return an in-memory audit table.

    The function never writes, moves, hashes, or deletes artifacts. It inspects
    compact component archives plus run summaries and their adjacent trajectory
    files.
    """

    root = Path(root)
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "component_runs").glob("*.npz")):
        facts = read_archive_facts(path)
        metadata = facts.metadata or {}
        base = {
            "path": str(path),
            "artifact_type": "smc_component",
            "algorithm": metadata.get("algorithm"),
            **_metadata_inventory(metadata),
            "ari_convention": infer_ari_convention(metadata),
            "has_phi": facts.has_phi,
            "has_raw_ari": facts.raw_ari is not None,
            "has_final_smc_weights": facts.final_log_weights is not None,
            "has_log_z": facts.log_normalizing_constant is not None,
            "has_completions": facts.has_completions,
            "has_token_ids": facts.has_completions,
            "has_runtime": facts.runtime_seconds is not None,
            "runtime_seconds": facts.runtime_seconds,
            "has_hardware_metadata": _has_hardware_metadata(metadata),
            "has_configuration_metadata": _has_configuration_metadata(metadata),
            "has_generated_token_count": facts.generated_token_count is not None,
            "generated_token_count": facts.generated_token_count,
            "has_proposal_count": False,
            "proposal_count": None,
            "has_suffix_lengths": False,
            "has_cut_points": False,
            "has_suffix_distribution": bool(
                metadata.get("suffix_sizes") and metadata.get("suffix_rhos")
            ),
            "has_chain_order": False,
            "has_lambda_schedule": bool(metadata.get("lambdas")),
            "has_level_labels": False,
            "read_error": facts.read_error,
        }
        reuse = assess_data_reuse(
            base,
            target_ari_convention=target_ari_convention,
        )
        rows.append({**base, **reuse})

    summary_paths = sorted(root.rglob("summary.json")) + sorted(root.rglob("run.json"))
    for path in summary_paths:
        if "component_runs" in path.parts:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append(
                {
                    "path": str(path),
                    "artifact_type": "unknown",
                    "read_error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        metadata = _metadata_from_json_payload(payload) or {}
        algorithm = payload.get("algorithm", metadata.get("algorithm"))
        reference_type = metadata.get("reference_type")
        if reference_type in {"current_pipeline_iid_direct", "official_public_iid_direct"}:
            artifact_type = "direct_run"
        elif algorithm == "official_annealed_tps":
            artifact_type = "tps_run"
        elif algorithm == "suffix_direct_mh_smc":
            artifact_type = "smc_run"
        else:
            continue
        trajectory_path = path.parent / "trajectory.npz"
        trajectory_keys: set[str] = set()
        if trajectory_path.is_file():
            with np.load(trajectory_path, allow_pickle=False) as trajectory:
                trajectory_keys = set(trajectory.files)
        samples = payload.get("samples")
        first_sample = (
            samples[0]
            if isinstance(samples, list) and samples and isinstance(samples[0], dict)
            else _first_jsonl_record(path.parent / "samples.jsonl")
        )
        sample_has_text = bool(
            first_sample
            and any(key in first_sample for key in ("completion", "completion_text"))
        )
        sample_has_tokens = bool(
            first_sample and "completion_token_ids" in first_sample
        )
        statistics = payload.get("statistics", {})
        summary = payload.get("summary", payload)
        config = payload.get("config", metadata)
        combined_metadata = {**metadata, **config, **payload}
        runtime_seconds = summary.get(
            "runtime_total_seconds",
            summary.get("runtime_seconds"),
        )
        generated_token_count = summary.get(
            "total_regenerated_tokens",
            summary.get("generated_token_transitions"),
        )
        proposal_count = summary.get(
            "total_proposals",
            summary.get("proposal_count"),
        )
        base = {
            "path": str(path),
            "artifact_type": artifact_type,
            "algorithm": algorithm or reference_type,
            **_metadata_inventory(combined_metadata),
            "ari_convention": infer_ari_convention(combined_metadata),
            "has_phi": bool("phi" in trajectory_keys or summary.get("mean_phi") is not None),
            "has_raw_ari": bool(
                "raw_ari" in trajectory_keys
                or statistics.get("count")
                or sample_has_text
            ),
            "has_final_smc_weights": bool(
                artifact_type == "smc_run"
                and first_sample
                and "normalized_weight" in first_sample
            ),
            "has_log_z": bool(
                summary.get("log_normalizing_constant_estimate") is not None
                or summary.get("log_z") is not None
            ),
            "has_completion_text": sample_has_text,
            "has_token_ids": sample_has_tokens,
            "has_runtime": bool(
                runtime_seconds is not None
            ),
            "runtime_seconds": runtime_seconds,
            "has_hardware_metadata": _has_hardware_metadata(
                {**metadata, **config, **payload}
            ),
            "has_configuration_metadata": _has_configuration_metadata(
                {**metadata, **config}
            ),
            "has_generated_token_count": bool(
                generated_token_count is not None
            ),
            "generated_token_count": generated_token_count,
            "has_proposal_count": proposal_count is not None,
            "proposal_count": proposal_count,
            "has_suffix_lengths": "suffix_size" in trajectory_keys,
            "has_cut_points": "cut_index" in trajectory_keys,
            "has_suffix_distribution": bool(config.get("suffix_sizes")),
            "has_fixed_length_and_count": bool(
                artifact_type == "direct_run"
                and metadata.get("fixed_completion_length")
                and metadata.get("max_new_tokens")
                and statistics.get("count")
            ),
            "has_chain_order": bool(
                artifact_type == "tps_run" and "step" in trajectory_keys
            ),
            "has_lambda_schedule": bool(
                config.get("lambdas") or "lambda" in trajectory_keys
            ),
            "has_level_labels": bool("lambda_index" in trajectory_keys),
            "known_lambda_zero": artifact_type == "direct_run",
            "read_error": None,
        }
        reuse = assess_data_reuse(
            base,
            target_ari_convention=target_ari_convention,
        )
        rows.append({**base, **reuse})
    return rows
