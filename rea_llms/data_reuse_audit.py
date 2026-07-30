from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


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

REQUIRED_METADATA_KEYS = SCIENTIFIC_METADATA_KEYS + COMPUTATIONAL_METADATA_KEYS


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
                )
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            keys = tuple(sorted(payload)) if isinstance(payload, dict) else ()
            metadata = _metadata_from_json_payload(payload)
            raw = None
            weights = None
            log_z = None
            if isinstance(payload, dict):
                if isinstance(payload.get("raw_ari"), list):
                    raw = np.asarray(payload["raw_ari"], dtype=np.float64)
                if isinstance(payload.get("final_log_weights"), list):
                    weights = np.asarray(payload["final_log_weights"], dtype=np.float64)
                if payload.get("log_normalizing_constant") is not None:
                    log_z = float(payload["log_normalizing_constant"])
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
    return math.isfinite(log_z) and metadata.get("resample_final") is False


def _mismatched_keys(
    metadata: Mapping[str, Any],
    expected: Mapping[str, Any],
    keys: Sequence[str],
) -> list[str]:
    return [key for key in keys if metadata.get(key) != expected.get(key)]


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
