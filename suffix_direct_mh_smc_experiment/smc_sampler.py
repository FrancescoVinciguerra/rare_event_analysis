"""Batched full-completion SMC with exact direct-MH suffix mutation."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

import numpy as np
import torch

from rea_llms.generation import (
    GenerationDiagnostics,
    generate_fixed_length_completions,
    generate_fixed_length_suffixes,
    prompt_token_ids,
)
from rea_llms.models import set_seed
from rea_llms.observables.base import observable_configuration
from rea_llms.scoring import (
    ARI_CONVENTION_CURRENT,
    score_completion_tokens,
    text_for_ari_from_token_ids,
)


ScheduleValue = TypeVar("ScheduleValue")
ALGORITHM_VERSION = "suffix_direct_mh_smc_v5_weighted_initial_population"


def _ari_texts_from_completion_ids(
    tokenizer,
    observable,
    *,
    prompt: str,
    prompt_ids: torch.Tensor,
    completion_ids: Sequence[Sequence[int]],
) -> list[str]:
    prompt_list = [int(value) for value in prompt_ids.detach().cpu().tolist()]
    convention = getattr(observable, "convention", ARI_CONVENTION_CURRENT)
    return [
        text_for_ari_from_token_ids(
            tokenizer,
            prompt=prompt,
            token_ids=prompt_list + [int(value) for value in completion],
            prompt_token_count=len(prompt_list),
            convention=convention,
        )
        for completion in completion_ids
    ]


# Configuration and result containers


@dataclass(frozen=True)
class SuffixDirectMHSMCConfig:
    """Configuration for full-completion SMC with direct-MH suffix moves.

    Particles always contain exactly ``max_new_tokens`` completion tokens. The
    suffix-size law is independent of particle state. Suffixes are sampled
    exactly from the base language model at temperature one, so the forward
    and reverse language-model factors cancel in the direct MH ratio.
    """

    num_particles: int = 64
    max_new_tokens: int = 100
    lambdas: tuple[float, ...] = (0.0, -0.25, -0.5, -1.0)
    mcmc_steps_per_level: int = 5
    ess_threshold: float = 0.5
    resampling: bool = True
    resample_final: bool = True
    resampling_method: str = "systematic"
    suffix_mode: str = "stochastic"
    suffix_sizes: tuple[int, ...] | None = None
    suffix_rhos: tuple[float, ...] | None = None
    suffix_size_schedule: tuple[tuple[int, ...], ...] | None = None
    suffix_rho_schedule: tuple[tuple[float, ...], ...] | None = None
    require_global_irreducibility: bool = True
    batch_size: int = 32
    seed: int = 12345
    tail_thresholds: tuple[float, ...] = (8.0, 10.0, 12.0)

    def __post_init__(self) -> None:
        if self.num_particles <= 0:
            raise ValueError("num_particles must be > 0")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if not self.lambdas or not math.isclose(float(self.lambdas[0]), 0.0, abs_tol=1e-12):
            raise ValueError("lambdas must be non-empty and start at 0")
        if any(not math.isfinite(float(value)) for value in self.lambdas):
            raise ValueError("all lambdas must be finite")
        if any(
            float(current) > float(previous) + 1e-12
            for previous, current in zip(self.lambdas, self.lambdas[1:])
        ):
            raise ValueError("lambdas must be non-increasing for the high-ARI convention")
        if self.mcmc_steps_per_level < 0:
            raise ValueError("mcmc_steps_per_level must be >= 0")
        if not 0.0 <= self.ess_threshold <= 1.0:
            raise ValueError("ess_threshold must be in [0, 1]")
        if self.resampling_method != "systematic":
            raise ValueError("resampling_method must be 'systematic'")
        if self.suffix_mode not in {"stochastic", "deterministic"}:
            raise ValueError("suffix_mode must be one of {'stochastic', 'deterministic'}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if any(not math.isfinite(float(value)) for value in self.tail_thresholds):
            raise ValueError("tail_thresholds must be finite")

        suffix_sizes = self.suffix_sizes
        if suffix_sizes is None:
            suffix_sizes = tuple(
                sorted({size for size in (1, 2, 4, 8, 16, 32, 64, self.max_new_tokens) if size <= self.max_new_tokens})
            )
            object.__setattr__(self, "suffix_sizes", suffix_sizes)
        if not suffix_sizes:
            raise ValueError("suffix_sizes must be non-empty")
        if len(set(suffix_sizes)) != len(suffix_sizes):
            raise ValueError("suffix_sizes must not contain duplicates")
        if any(size <= 0 or size > self.max_new_tokens for size in suffix_sizes):
            raise ValueError("every suffix size must be in [1, max_new_tokens]")

        transitions = max(0, len(self.lambdas) - 1)
        if self.suffix_mode == "stochastic":
            if self.suffix_size_schedule is not None:
                raise ValueError("suffix_size_schedule is only valid in deterministic mode")
            self._validate_stochastic_schedule(suffix_sizes, transitions)
        else:
            if self.suffix_rhos is not None or self.suffix_rho_schedule is not None:
                raise ValueError("suffix rhos are only valid in stochastic mode")
            self._validate_deterministic_schedule(transitions)

    def _validate_rho_row(self, row: Sequence[float], suffix_sizes: Sequence[int]) -> None:
        if len(row) != len(suffix_sizes):
            raise ValueError("each suffix-rho row must match the number of suffix_sizes")
        if any(not math.isfinite(float(value)) or value < 0.0 for value in row):
            raise ValueError("suffix rho values must be finite and non-negative")
        if not math.isclose(sum(row), 1.0, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError("each suffix-rho row must sum to one")
        if self.require_global_irreducibility:
            full_index = suffix_sizes.index(self.max_new_tokens) if self.max_new_tokens in suffix_sizes else None
            if full_index is None or row[full_index] <= 0.0:
                raise ValueError(
                    "global irreducibility requires suffix size T with positive probability; "
                    "use --allow-local-suffix-support only for deliberate local-kernel studies"
                )

    def _validate_stochastic_schedule(
        self,
        suffix_sizes: Sequence[int],
        transitions: int,
    ) -> None:
        if self.suffix_rhos is not None and self.suffix_rho_schedule is not None:
            raise ValueError("use either suffix_rhos or suffix_rho_schedule, not both")
        rows = self.suffix_rho_schedule
        if rows is not None:
            if len(rows) not in {1, transitions}:
                raise ValueError("suffix_rho_schedule must have one row or one row per transition")
            for row in rows:
                self._validate_rho_row(row, suffix_sizes)
            return
        row = self.suffix_rhos
        if row is None:
            row = tuple(1.0 / len(suffix_sizes) for _ in suffix_sizes)
            object.__setattr__(self, "suffix_rhos", row)
        self._validate_rho_row(row, suffix_sizes)

    def _validate_deterministic_schedule(self, transitions: int) -> None:
        rows = self.suffix_size_schedule
        if rows is None:
            rows = ((self.max_new_tokens,),)
            object.__setattr__(self, "suffix_size_schedule", rows)
        if len(rows) not in {1, transitions}:
            raise ValueError("suffix_size_schedule must have one row or one row per transition")
        valid_lengths = {1, self.mcmc_steps_per_level}
        for row in rows:
            if len(row) not in valid_lengths:
                raise ValueError(
                    "each suffix-size row must have one value or one value per MCMC step"
                )
            if any(size <= 0 or size > self.max_new_tokens for size in row):
                raise ValueError("deterministic suffix sizes must be in [1, max_new_tokens]")
            if self.require_global_irreducibility and self.max_new_tokens not in row:
                raise ValueError(
                    "each deterministic transition must include suffix size T when global "
                    "irreducibility is required"
                )

@dataclass
class ParticleBatch:
    """Device-resident particles and cached observables with shape ``[N, ...]``."""

    completion_token_ids: torch.Tensor
    phi: torch.Tensor
    reported_values: torch.Tensor
    root_ancestors: torch.Tensor
    last_suffix_sizes: torch.Tensor


@dataclass
class MutationDiagnostics:
    """Small host-side summary returned after one mutation level."""

    accepted: int
    proposals: int
    mean_acceptance_probability: float
    suffix_size_counts: dict[str, int]
    accepted_by_suffix_size: dict[str, int]
    acceptance_rate_by_suffix_size: dict[str, float]
    regenerated_tokens: int
    generator_runtime_seconds: float
    observable_runtime_seconds: float


@dataclass
class SuffixDirectMHSMCResult:
    """Serializable final population, run summary, and level diagnostics."""

    samples: list[dict[str, Any]]
    summary: dict[str, Any]
    diagnostics: dict[str, Any]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "summary": self.summary,
            "diagnostics": self.diagnostics,
            "config": self.config,
        }


def parse_float_list(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"Could not parse comma-separated floats from {value!r}") from exc
    if not parsed:
        raise ValueError("Expected at least one comma-separated float")
    return parsed


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"Could not parse comma-separated integers from {value!r}") from exc
    if not parsed:
        raise ValueError("Expected at least one comma-separated integer")
    return parsed


def parse_float_schedule(value: str) -> tuple[tuple[float, ...], ...]:
    return tuple(parse_float_list(row) for row in value.split(";") if row.strip())


def parse_int_schedule(value: str) -> tuple[tuple[int, ...], ...]:
    return tuple(parse_int_list(row) for row in value.split(";") if row.strip())


def _schedule_row(rows: Sequence[ScheduleValue], transition_index: int) -> ScheduleValue:
    return rows[0] if len(rows) == 1 else rows[transition_index]


def _step_value(row: Sequence[ScheduleValue], step_index: int) -> ScheduleValue:
    return row[0] if len(row) == 1 else row[step_index]


def propose_suffixes_parallel(
    model,
    prompt_ids: torch.Tensor,
    current: torch.Tensor,
    suffix_sizes: torch.Tensor,
    *,
    batch_size: int,
    generator: torch.Generator,
    generation_diagnostics: GenerationDiagnostics | None = None,
) -> torch.Tensor:
    """Regenerate exact LM suffixes in GPU batches grouped by suffix length.

    Particles with the same retained-prefix length form one GPU workload. The
    loop is over distinct suffix sizes, never over individual particles. Tokens
    inside one suffix remain autoregressive. No proposal probability is scored:
    at temperature one its forward and reverse LM factors cancel exactly in the
    direct-MH acceptance ratio.
    """
    proposals = current.clone()
    for suffix_size in torch.unique(suffix_sizes, sorted=True).detach().cpu().tolist():
        indices = torch.nonzero(suffix_sizes == suffix_size, as_tuple=False).flatten()
        prefix_size = current.shape[1] - int(suffix_size)
        current_group = current.index_select(0, indices)
        prompt = prompt_ids.unsqueeze(0).expand(indices.numel(), -1)
        context = torch.cat((prompt, current_group[:, :prefix_size]), dim=1)
        proposed_suffix = generate_fixed_length_suffixes(
            model,
            context,
            suffix_size=int(suffix_size),
            batch_size=batch_size,
            generator=generator,
            diagnostics=generation_diagnostics,
        )
        proposal_group = torch.cat((current_group[:, :prefix_size], proposed_suffix), dim=1)
        proposals.index_copy_(0, indices, proposal_group)
    return proposals


# Exact direct-Metropolis mutation kernel


def metropolis_log_acceptance(
    current_phi: torch.Tensor,
    proposal_phi: torch.Tensor,
    *,
    target_lambda: float,
) -> torch.Tensor:
    """Direct suffix-MH acceptance after exact cancellation of all LM terms.

    At mutation level k-1, both target and proposal contain the same conditional
    suffix probabilities. The remaining ratio is exactly
    ``exp(-lambda * (phi_proposal - phi_current))``.
    """
    log_ratio = -target_lambda * (proposal_phi - current_phi)
    if not bool(torch.isfinite(log_ratio).all().item()):
        raise ValueError("Non-finite direct Metropolis log-acceptance ratio")
    return torch.minimum(torch.zeros_like(log_ratio), log_ratio)


def score_observable(
    tokenizer,
    observable,
    *,
    prompt: str,
    observable_prompt: str | None = None,
    prompt_ids: torch.Tensor,
    completions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[Any]]:
    """Evaluate one batched completion observable at its declared boundary.

    Full token sequences are transferred together, then completion tokens are
    decoded with the repository observable's convention before textstat runs on
    CPU. Decoding the actual token sequence is essential: BPE boundaries,
    punctuation, repeated tokens, and concatenated words make token-local ARI
    arithmetic unsafe unless exact textual equivalence is proved.
    """
    return score_completion_tokens(
        tokenizer,
        observable,
        prompt=observable_prompt if observable_prompt is not None else prompt,
        prompt_ids=prompt_ids,
        completions=completions,
    )


def generate_base_completions(
    model,
    prompt_ids: torch.Tensor,
    *,
    num_particles: int,
    max_new_tokens: int,
    batch_size: int,
    generator: torch.Generator,
    generation_diagnostics: GenerationDiagnostics | None = None,
) -> torch.Tensor:
    """Initialize all particles from p_M using the batched autoregressive path."""
    return generate_fixed_length_completions(
        model,
        prompt_ids,
        num_sequences=num_particles,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        generator=generator,
        diagnostics=generation_diagnostics,
    )


@dataclass(frozen=True)
class _SuffixSizeSamplingState:
    deterministic_row: tuple[int, ...] | None
    sizes: torch.Tensor | None
    probabilities: torch.Tensor | None


def _prepare_suffix_size_sampling(
    config: SuffixDirectMHSMCConfig,
    *,
    transition_index: int,
    device: torch.device,
) -> _SuffixSizeSamplingState:
    """Materialize immutable suffix-sampling constants once per transition."""

    if config.suffix_mode == "deterministic":
        assert config.suffix_size_schedule is not None
        row = tuple(_schedule_row(config.suffix_size_schedule, transition_index))
        return _SuffixSizeSamplingState(row, None, None)

    assert config.suffix_sizes is not None
    rows = config.suffix_rho_schedule
    rho_values = _schedule_row(rows, transition_index) if rows is not None else config.suffix_rhos
    assert rho_values is not None
    return _SuffixSizeSamplingState(
        deterministic_row=None,
        sizes=torch.tensor(config.suffix_sizes, dtype=torch.long, device=device),
        probabilities=torch.tensor(rho_values, dtype=torch.float32, device=device),
    )


def _sample_suffix_sizes_prepared(
    config: SuffixDirectMHSMCConfig,
    sampling: _SuffixSizeSamplingState,
    *,
    step_index: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if sampling.deterministic_row is not None:
        size = int(_step_value(sampling.deterministic_row, step_index))
        return torch.full((config.num_particles,), size, dtype=torch.long, device=device)

    assert sampling.sizes is not None and sampling.probabilities is not None
    choices = torch.multinomial(
        sampling.probabilities,
        config.num_particles,
        replacement=True,
        generator=generator,
    )
    return sampling.sizes.index_select(0, choices)


def sample_suffix_sizes(
    config: SuffixDirectMHSMCConfig,
    *,
    transition_index: int,
    step_index: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw all state-independent suffix lengths in one device operation.

    State independence and the shared forward/reverse law are what permit the
    suffix-size probability to cancel in direct Metropolis acceptance.
    """
    sampling = _prepare_suffix_size_sampling(
        config,
        transition_index=transition_index,
        device=device,
    )
    return _sample_suffix_sizes_prepared(
        config,
        sampling,
        step_index=step_index,
        device=device,
        generator=generator,
    )


def systematic_resample(
    normalized_log_weights: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw systematic-resampling ancestor indices entirely on the device."""
    n_particles = normalized_log_weights.numel()
    weights = torch.exp(normalized_log_weights)
    cumulative = torch.cumsum(weights, dim=0)
    cumulative[-1] = 1.0
    start = torch.rand((), device=weights.device, generator=generator) / n_particles
    positions = start + torch.arange(n_particles, device=weights.device) / n_particles
    return torch.searchsorted(cumulative, positions, right=False)


def normalize_log_weights(log_weights: torch.Tensor) -> torch.Tensor:
    """Return normalized log weights using one device-side logsumexp."""

    return log_weights - torch.logsumexp(log_weights, dim=0)


def ess_from_normalized_log_weights(normalized_log_weights: torch.Tensor) -> torch.Tensor:
    """Compute ``1/sum(W**2)`` directly from normalized log weights."""

    return torch.exp(-torch.logsumexp(2.0 * normalized_log_weights, dim=0))


def incremental_log_weight_update(
    normalized_log_weights: torch.Tensor,
    phi: torch.Tensor,
    *,
    previous_lambda: float,
    current_lambda: float,
) -> torch.Tensor:
    """Apply gamma_k/gamma_{k-1} and renormalize in log space.

    The LM factor is shared by adjacent targets, so their ratio is only
    ``exp(-(lambda_k - lambda_{k-1}) * phi)``.
    """
    updated, _ = incremental_log_weight_update_with_normalizer(
        normalized_log_weights,
        phi,
        previous_lambda=previous_lambda,
        current_lambda=current_lambda,
    )
    return updated


def incremental_log_weight_update_with_normalizer(
    normalized_log_weights: torch.Tensor,
    phi: torch.Tensor,
    *,
    previous_lambda: float,
    current_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update normalized weights and estimate the adjacent normalizer ratio.

    Because the input weights sum to one, the log average incremental
    potential is the particle estimate of ``log(Z_k / Z_(k-1))``. Returning it
    before normalization lets the SMC driver accumulate ``log Z_k`` without
    changing the existing mutation, weighting, or resampling order.
    """

    if normalized_log_weights.shape != phi.shape:
        raise ValueError("normalized_log_weights and phi must have the same shape")
    log_increment = -(
        float(current_lambda) - float(previous_lambda)
    ) * phi.to(dtype=torch.float64, device=normalized_log_weights.device)
    unnormalized = normalized_log_weights.to(torch.float64) + log_increment
    log_normalizer_increment = torch.logsumexp(unnormalized, dim=0)
    if not bool(torch.isfinite(log_normalizer_increment).item()):
        raise ValueError("non-finite SMC normalizing-constant increment")
    return unnormalized - log_normalizer_increment, log_normalizer_increment


# Full SMC mutation, reweighting, ESS, and resampling


def apply_suffix_direct_mh_mutation(
    particles: ParticleBatch,
    model,
    tokenizer,
    observable,
    *,
    prompt: str,
    observable_prompt: str | None = None,
    prompt_ids: torch.Tensor,
    target_lambda: float,
    transition_index: int,
    config: SuffixDirectMHSMCConfig,
    generator: torch.Generator,
    discovery_callback: Callable[[torch.Tensor, torch.Tensor, float], None] | None = None,
    generation_diagnostics: GenerationDiagnostics | None = None,
    profile_runtime: bool = False,
) -> tuple[ParticleBatch, MutationDiagnostics]:
    """Apply repeated direct-MH suffix moves invariant for ``pi_{k-1}``.

    Each step proposes all particles, evaluates proposal observables in one
    batch, and makes accept/reject decisions vectorially on the model device.
    The language-model and state-independent suffix-size factors cancel from
    the MH ratio. Only the subsequent SMC weight update changes target level.
    """
    device = particles.completion_token_ids.device
    accepted_total = torch.zeros((), dtype=torch.long, device=device)
    acceptance_probability_sum = torch.zeros((), dtype=torch.float64, device=device)
    size_counts = torch.zeros(config.max_new_tokens + 1, dtype=torch.long, device=device)
    accepted_by_size = torch.zeros(
        config.max_new_tokens + 1,
        dtype=torch.long,
        device=device,
    )
    regenerated_tokens = torch.zeros((), dtype=torch.long, device=device)
    generator_runtime_seconds = 0.0
    observable_runtime_seconds = 0.0
    proposals_total = config.num_particles * config.mcmc_steps_per_level
    suffix_sampling = _prepare_suffix_size_sampling(
        config,
        transition_index=transition_index,
        device=device,
    )

    for step_index in range(config.mcmc_steps_per_level):
        suffix_sizes = _sample_suffix_sizes_prepared(
            config,
            suffix_sampling,
            step_index=step_index,
            device=device,
            generator=generator,
        )
        if profile_runtime:
            _synchronize(device)
        generator_started = time.perf_counter() if profile_runtime else None
        proposal_tokens = propose_suffixes_parallel(
            model,
            prompt_ids,
            particles.completion_token_ids,
            suffix_sizes,
            batch_size=config.batch_size,
            generator=generator,
            generation_diagnostics=generation_diagnostics,
        )
        if profile_runtime:
            _synchronize(device)
            assert generator_started is not None
            generator_runtime_seconds += time.perf_counter() - generator_started
        # Current-particle scores remain cached in ParticleBatch. Only each
        # proposal batch crosses the selected observable's scoring boundary.
        observable_started = time.perf_counter() if profile_runtime else None
        proposal_phi, proposal_reported_values, _ = score_observable(
            tokenizer,
            observable,
            prompt=prompt,
            observable_prompt=observable_prompt,
            prompt_ids=prompt_ids,
            completions=proposal_tokens,
        )
        if profile_runtime:
            assert observable_started is not None
            observable_runtime_seconds += time.perf_counter() - observable_started
        if discovery_callback is not None:
            discovery_callback(proposal_tokens, proposal_reported_values, target_lambda)
        # Exact direct MH: p_M and suffix-selection factors cancel completely.
        log_acceptance = metropolis_log_acceptance(
            particles.phi,
            proposal_phi,
            target_lambda=target_lambda,
        )
        log_uniform = torch.log(
            torch.rand(
                config.num_particles,
                dtype=log_acceptance.dtype,
                device=device,
                generator=generator,
            )
        )
        accepted = log_uniform < log_acceptance
        particles = ParticleBatch(
            completion_token_ids=torch.where(
                accepted.unsqueeze(1),
                proposal_tokens,
                particles.completion_token_ids,
            ),
            phi=torch.where(accepted, proposal_phi, particles.phi),
            reported_values=torch.where(
                accepted,
                proposal_reported_values,
                particles.reported_values,
            ),
            root_ancestors=particles.root_ancestors,
            last_suffix_sizes=suffix_sizes,
        )
        accepted_total += accepted.sum()
        acceptance_probability_sum += torch.exp(log_acceptance).to(torch.float64).sum()
        size_counts += torch.bincount(suffix_sizes, minlength=config.max_new_tokens + 1)
        accepted_by_size += torch.bincount(
            suffix_sizes[accepted],
            minlength=config.max_new_tokens + 1,
        )
        regenerated_tokens += suffix_sizes.sum()

    denominator = max(1, proposals_total)
    counts_on_cpu = size_counts.detach().cpu().tolist()
    suffix_count_rows = {
        str(size): int(counts_on_cpu[size])
        for size in range(1, config.max_new_tokens + 1)
        if counts_on_cpu[size] > 0
    }
    accepted_on_cpu = accepted_by_size.detach().cpu().tolist()
    accepted_size_rows = {
        str(size): int(accepted_on_cpu[size])
        for size in range(1, config.max_new_tokens + 1)
        if counts_on_cpu[size] > 0
    }
    acceptance_rate_rows = {
        key: accepted_size_rows[key] / suffix_count_rows[key]
        for key in suffix_count_rows
    }
    return particles, MutationDiagnostics(
        accepted=int(accepted_total.item()),
        proposals=proposals_total,
        mean_acceptance_probability=float(
            (acceptance_probability_sum / denominator).item()
        ),
        suffix_size_counts=suffix_count_rows,
        accepted_by_suffix_size=accepted_size_rows,
        acceptance_rate_by_suffix_size=acceptance_rate_rows,
        regenerated_tokens=int(regenerated_tokens.item()),
        generator_runtime_seconds=float(generator_runtime_seconds),
        observable_runtime_seconds=float(observable_runtime_seconds),
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _final_samples(
    tokenizer,
    observable,
    *,
    prompt: str,
    observable_prompt: str | None,
    prompt_ids: torch.Tensor,
    particles: ParticleBatch,
    normalized_log_weights: torch.Tensor,
    seed: int,
    lambda_final: float,
) -> list[dict[str, Any]]:
    """Serialize the final particle system and its current normalized weights.

    This currently reconstructs score metadata and decoded strings on CPU. The
    repeated work is retained during this audit because changing it would need
    an explicit equivalence test for every serialized field.
    """
    _, _, scores = score_observable(
        tokenizer,
        observable,
        prompt=prompt,
        observable_prompt=observable_prompt,
        prompt_ids=prompt_ids,
        completions=particles.completion_token_ids,
    )
    completion_ids = particles.completion_token_ids.detach().cpu().tolist()
    if getattr(observable, "name", "ari") == "ari":
        texts = _ari_texts_from_completion_ids(
            tokenizer,
            observable,
            prompt=prompt,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
        )
        completions = [text[len(prompt) :] for text in texts]
    else:
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        score_prompt = observable_prompt if observable_prompt is not None else prompt
        texts = [score_prompt + completion for completion in completions]
    weights = torch.exp(normalized_log_weights).detach().cpu().tolist()
    log_weights = normalized_log_weights.detach().cpu().tolist()
    phi = particles.phi.detach().cpu().tolist()
    report_values = particles.reported_values.detach().cpu().tolist()
    roots = particles.root_ancestors.detach().cpu().tolist()
    last_sizes = particles.last_suffix_sizes.detach().cpu().tolist()
    rows: list[dict[str, Any]] = []
    is_ari = getattr(observable, "name", "ari") == "ari"
    for index, (ids, completion, score) in enumerate(zip(completion_ids, completions, scores)):
        metadata = dict(score.metadata)
        row = {
                "particle_id": index,
                "ancestor_id": int(roots[index]),
                "root_ancestor": int(roots[index]),
                "seed": int(seed),
                "text": texts[index],
                "completion": completion,
                "phi": float(phi[index]),
                "reported_value": float(report_values[index]),
                "observable": str(getattr(observable, "name", "unknown")),
                "log_weight": float(log_weights[index]),
                "normalized_weight": float(weights[index]),
                "lambda_final": float(lambda_final),
                "mutation_kernel": "suffix_direct_mh",
                "last_suffix_size": int(last_sizes[index]),
                "completion_token_ids": ids,
                "metadata": metadata,
            }
        if is_ari:
            row["raw_ari"] = float(metadata["raw_ari"])
            row["capped_ari"] = float(metadata["capped_ari"])
        else:
            row["harmfulness_score"] = float(report_values[index])
            row["guardrail_class"] = metadata.get("predicted_class")
        rows.append(row)
    return rows


def _particle_snapshot(
    *,
    level: int,
    lambda_value: float,
    particles: ParticleBatch,
    normalized_log_weights: torch.Tensor,
) -> dict[str, Any]:
    """Copy one optional diagnostic population to serializable host memory."""

    return {
        "level": int(level),
        "lambda": float(lambda_value),
        "completion_token_ids": particles.completion_token_ids.detach().cpu().tolist(),
        "phi": particles.phi.detach().cpu().tolist(),
        "reported_values": particles.reported_values.detach().cpu().tolist(),
        "normalized_weights": torch.exp(normalized_log_weights).detach().cpu().tolist(),
        "root_ancestors": particles.root_ancestors.detach().cpu().tolist(),
        "last_suffix_sizes": particles.last_suffix_sizes.detach().cpu().tolist(),
    }


def _observable_counter_snapshot(observable) -> dict[str, float]:
    names = (
        "cache_hits",
        "cache_misses",
        "evaluation_batches",
        "evaluated_completions",
        "inference_seconds",
    )
    return {
        name: float(getattr(observable, name, 0.0))
        for name in names
    }


def run_suffix_direct_mh_smc(
    model,
    tokenizer,
    observable,
    *,
    prompt: str,
    observable_prompt: str | None = None,
    config: SuffixDirectMHSMCConfig,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    track_discovery_diagnostics: bool = True,
    capture_level_particles: bool = False,
    profile_runtime: bool = False,
    initial_completion_token_ids: torch.Tensor | None = None,
    initial_log_weights: torch.Tensor | None = None,
) -> SuffixDirectMHSMCResult:
    """Run full-sequence SMC with an exact direct-MH suffix mutation.

    The order at transition k is mutation under pi_{k-1}, incremental weighting
    from gamma_{k-1} to gamma_k, then optional systematic resampling. By default
    the final level is resampled too, so returned particles carry uniform
    empirical weights under the terminal tilted target. An externally sampled
    initial population may be supplied with unnormalized importance log-weights;
    this supports proposal distributions other than the base language model.
    """
    set_seed(config.seed)
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(config.seed)
    generation_diagnostics = GenerationDiagnostics()
    observable_counters_before = _observable_counter_snapshot(observable)
    prompt_ids = prompt_token_ids(tokenizer, prompt, device)
    uniform_log_weight = -math.log(config.num_particles)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    _synchronize(device)
    start = time.perf_counter()
    discovered_by_threshold = {
        float(threshold): set() for threshold in config.tail_thresholds
    }
    first_tail_discovery: dict[float, dict[str, Any] | None] = {
        float(threshold): None for threshold in config.tail_thresholds
    }
    best_discoveries: dict[str, dict[str, Any]] = {}
    is_ari_observable = getattr(observable, "name", "ari") == "ari"
    generator_runtime_seconds = 0.0
    observable_runtime_seconds = 0.0
    level_particles: list[dict[str, Any]] = []

    def track_discoveries(
        token_batch: torch.Tensor,
        reported_value_batch: torch.Tensor,
        lambda_value: float,
    ) -> None:
        # Discovery statistics include every generated proposal, accepted or
        # rejected. This requires a second token transfer/decode beyond ARI
        # scoring and is intentionally preserved as part of metric semantics.
        completion_ids = token_batch.detach().cpu().tolist()
        if getattr(observable, "name", "ari") == "ari":
            decoded_texts = _ari_texts_from_completion_ids(
                tokenizer,
                observable,
                prompt=prompt,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
            )
            completion_texts = [text[len(prompt) :] for text in decoded_texts]
        else:
            completion_texts = tokenizer.batch_decode(
                completion_ids,
                skip_special_tokens=True,
            )
        reported_values = reported_value_batch.detach().cpu().tolist()
        elapsed_now = time.perf_counter() - start
        for completion, reported_value in zip(completion_texts, reported_values):
            value = float(reported_value)
            previous = best_discoveries.get(completion)
            if previous is None or value > float(previous["reported_value"]):
                discovery = {
                    "reported_value": value,
                    "lambda": float(lambda_value),
                    "time_first_seen_seconds": float(elapsed_now),
                    "completion": completion,
                }
                if is_ari_observable:
                    discovery["raw_ari"] = value
                best_discoveries[completion] = discovery
            for threshold in config.tail_thresholds:
                threshold_value = float(threshold)
                if value < threshold_value:
                    continue
                discovered_by_threshold[threshold_value].add(completion)
                if first_tail_discovery[threshold_value] is None:
                    discovery = {
                        "reported_value": value,
                        "lambda": float(lambda_value),
                        "time_first_seen_seconds": float(elapsed_now),
                        "completion": completion,
                    }
                    if is_ari_observable:
                        discovery["raw_ari"] = value
                    first_tail_discovery[threshold_value] = discovery

    with torch.inference_mode():
        if initial_completion_token_ids is None:
            if initial_log_weights is not None:
                raise ValueError(
                    "initial_log_weights requires initial_completion_token_ids"
                )
            if profile_runtime:
                _synchronize(device)
            generation_started = time.perf_counter() if profile_runtime else None
            completion_tokens = generate_base_completions(
                model,
                prompt_ids,
                num_particles=config.num_particles,
                max_new_tokens=config.max_new_tokens,
                batch_size=config.batch_size,
                generator=generator,
                generation_diagnostics=generation_diagnostics,
            )
            if profile_runtime:
                _synchronize(device)
                assert generation_started is not None
                generator_runtime_seconds += time.perf_counter() - generation_started
            log_weights = torch.full(
                (config.num_particles,),
                uniform_log_weight,
                dtype=torch.float64,
                device=device,
            )
            initial_log_normalizer = torch.zeros(
                (), dtype=torch.float64, device=device
            )
            initialization = "base_iid"
        else:
            completion_tokens = torch.as_tensor(
                initial_completion_token_ids,
                dtype=torch.long,
                device=device,
            ).clone()
            expected_shape = (config.num_particles, config.max_new_tokens)
            if tuple(completion_tokens.shape) != expected_shape:
                raise ValueError(
                    "initial_completion_token_ids must have shape "
                    f"{expected_shape}, got {tuple(completion_tokens.shape)}"
                )
            if initial_log_weights is None:
                log_weights = torch.full(
                    (config.num_particles,),
                    uniform_log_weight,
                    dtype=torch.float64,
                    device=device,
                )
                initial_log_normalizer = torch.zeros(
                    (), dtype=torch.float64, device=device
                )
                initialization = "external_uniform"
            else:
                raw_initial_log_weights = torch.as_tensor(
                    initial_log_weights,
                    dtype=torch.float64,
                    device=device,
                ).clone()
                if tuple(raw_initial_log_weights.shape) != (config.num_particles,):
                    raise ValueError(
                        "initial_log_weights must have shape "
                        f"({config.num_particles},), got "
                        f"{tuple(raw_initial_log_weights.shape)}"
                    )
                if not bool(torch.isfinite(raw_initial_log_weights).all().item()):
                    raise ValueError("initial_log_weights must all be finite")
                initial_log_normalizer = (
                    torch.logsumexp(raw_initial_log_weights, dim=0)
                    - math.log(config.num_particles)
                )
                log_weights = normalize_log_weights(raw_initial_log_weights)
                initialization = "external_importance_weighted"
        observable_started = time.perf_counter() if profile_runtime else None
        phi, report_values, _ = score_observable(
            tokenizer,
            observable,
            prompt=prompt,
            observable_prompt=observable_prompt,
            prompt_ids=prompt_ids,
            completions=completion_tokens,
        )
        if profile_runtime:
            assert observable_started is not None
            observable_runtime_seconds += time.perf_counter() - observable_started
        if track_discovery_diagnostics:
            track_discoveries(
                completion_tokens,
                report_values,
                float(config.lambdas[0]),
            )
        particles = ParticleBatch(
            completion_token_ids=completion_tokens,
            phi=phi,
            reported_values=report_values,
            root_ancestors=torch.arange(config.num_particles, device=device),
            last_suffix_sizes=torch.zeros(config.num_particles, dtype=torch.long, device=device),
        )
        initial_ess = ess_from_normalized_log_weights(log_weights)
        initial_ess_over_n = min(
            1.0, float((initial_ess / config.num_particles).item())
        )
        if capture_level_particles:
            level_particles.append(
                _particle_snapshot(
                    level=0,
                    lambda_value=float(config.lambdas[0]),
                    particles=particles,
                    normalized_log_weights=log_weights,
                )
            )
        initial_mean = float(
            torch.sum(
                torch.exp(log_weights)
                * report_values.to(dtype=torch.float64)
            ).item()
        )
        level_diagnostics: list[dict[str, Any]] = [
            {
                "level": 0,
                "lambda": float(config.lambdas[0]),
                "mutation_target_lambda": None,
                "log_normalizing_constant_increment": float(
                    initial_log_normalizer.item()
                ),
                "log_normalizing_constant_estimate": float(
                    initial_log_normalizer.item()
                ),
                "ess_over_n_before_resampling": initial_ess_over_n,
                "ess_over_n_after_resampling": initial_ess_over_n,
                "resampled": False,
                "accepted": 0,
                "proposals": 0,
                "acceptance_rate": None,
                "mean_acceptance_probability": None,
                "weighted_mean_observable_before_resampling": initial_mean,
                "unweighted_mean_observable_after_resampling": initial_mean,
                "runtime_seconds": 0.0,
                "suffix_size_counts": {},
                "accepted_by_suffix_size": {},
                "acceptance_rate_by_suffix_size": {},
                "generator_runtime_seconds": (
                    generator_runtime_seconds if profile_runtime else None
                ),
                "observable_runtime_seconds": (
                    observable_runtime_seconds if profile_runtime else None
                ),
            }
        ]
        total_accepted = 0
        total_proposals = 0
        total_regenerated_tokens = 0
        resampling_steps = 0
        min_ess_over_n = initial_ess_over_n
        final_weight_ess_over_n_before_resampling = initial_ess_over_n
        log_normalizing_constant = initial_log_normalizer.clone()

        for level in range(1, len(config.lambdas)):
            _synchronize(device)
            level_start = time.perf_counter()
            previous_lambda = float(config.lambdas[level - 1])
            current_lambda = float(config.lambdas[level])

            # 1. Mutate with M_{k-1}, which leaves pi_{k-1} invariant.
            particles, mutation = apply_suffix_direct_mh_mutation(
                particles,
                model,
                tokenizer,
                observable,
                prompt=prompt,
                observable_prompt=observable_prompt,
                prompt_ids=prompt_ids,
                target_lambda=previous_lambda,
                transition_index=level - 1,
                config=config,
                generator=generator,
                discovery_callback=(
                    track_discoveries if track_discovery_diagnostics else None
                ),
                generation_diagnostics=generation_diagnostics,
                profile_runtime=profile_runtime,
            )
            generator_runtime_seconds += mutation.generator_runtime_seconds
            observable_runtime_seconds += mutation.observable_runtime_seconds
            total_accepted += mutation.accepted
            total_proposals += mutation.proposals
            total_regenerated_tokens += mutation.regenerated_tokens

            # 2. Since p_M is common to adjacent targets, only the exponential
            # potential ratio is needed to move from gamma_{k-1} to gamma_k.
            log_weights, log_normalizer_increment = (
                incremental_log_weight_update_with_normalizer(
                    log_weights,
                    particles.phi,
                    previous_lambda=previous_lambda,
                    current_lambda=current_lambda,
                )
            )
            log_normalizing_constant += log_normalizer_increment
            # 3. Diagnose normalized weights. Intermediate low-ESS resampling
            # helps subsequent mutation; final resampling returns an unweighted
            # terminal population for the downstream SNIS correction.
            ess = ess_from_normalized_log_weights(log_weights)
            ess_over_n_before = min(1.0, float((ess / config.num_particles).item()))
            weights_before = torch.exp(log_weights)
            weighted_mean_before = float(
                torch.sum(
                    weights_before * particles.reported_values.to(torch.float64)
                ).item()
            )
            min_ess_over_n = min(min_ess_over_n, ess_over_n_before)
            is_final_level = level == len(config.lambdas) - 1
            if is_final_level:
                final_weight_ess_over_n_before_resampling = ess_over_n_before
            resampled = bool(
                config.resampling
                and (
                    (not is_final_level and ess_over_n_before < config.ess_threshold)
                    or (is_final_level and config.resample_final)
                )
            )
            if resampled:
                indices = systematic_resample(log_weights, generator=generator)
                particles = ParticleBatch(
                    completion_token_ids=particles.completion_token_ids.index_select(
                        0, indices
                    ),
                    phi=particles.phi.index_select(0, indices),
                    reported_values=particles.reported_values.index_select(0, indices),
                    root_ancestors=particles.root_ancestors.index_select(0, indices),
                    last_suffix_sizes=particles.last_suffix_sizes.index_select(0, indices),
                )
                log_weights = torch.full_like(log_weights, uniform_log_weight)
                resampling_steps += 1
            ess_over_n_after = 1.0 if resampled else ess_over_n_before
            _synchronize(device)
            row = {
                "level": level,
                "lambda": current_lambda,
                "previous_lambda": previous_lambda,
                "delta_lambda": current_lambda - previous_lambda,
                "mutation_target_lambda": previous_lambda,
                "log_normalizing_constant_increment": float(
                    log_normalizer_increment.item()
                ),
                "log_normalizing_constant_estimate": float(
                    log_normalizing_constant.item()
                ),
                "ess_over_n_before_resampling": ess_over_n_before,
                "ess_over_n_after_resampling": ess_over_n_after,
                "resampled": resampled,
                "accepted": mutation.accepted,
                "proposals": mutation.proposals,
                "acceptance_rate": (
                    mutation.accepted / mutation.proposals if mutation.proposals else None
                ),
                "mean_acceptance_probability": mutation.mean_acceptance_probability,
                "weighted_mean_observable_before_resampling": weighted_mean_before,
                "unweighted_mean_observable_after_resampling": float(
                    particles.reported_values.mean().item()
                ),
                "runtime_seconds": float(time.perf_counter() - level_start),
                "suffix_size_counts": mutation.suffix_size_counts,
                "accepted_by_suffix_size": mutation.accepted_by_suffix_size,
                "acceptance_rate_by_suffix_size": mutation.acceptance_rate_by_suffix_size,
                "generator_runtime_seconds": (
                    mutation.generator_runtime_seconds if profile_runtime else None
                ),
                "observable_runtime_seconds": (
                    mutation.observable_runtime_seconds if profile_runtime else None
                ),
            }
            level_diagnostics.append(row)
            if capture_level_particles:
                level_particles.append(
                    _particle_snapshot(
                        level=level,
                        lambda_value=current_lambda,
                        particles=particles,
                        normalized_log_weights=log_weights,
                    )
                )
            if progress_callback is not None:
                progress_callback(row)

        samples = _final_samples(
            tokenizer,
            observable,
            prompt=prompt,
            observable_prompt=observable_prompt,
            prompt_ids=prompt_ids,
            particles=particles,
            normalized_log_weights=log_weights,
            seed=config.seed,
            lambda_final=float(config.lambdas[-1]),
        )

    _synchronize(device)
    elapsed = time.perf_counter() - start
    final_ess = ess_from_normalized_log_weights(log_weights)
    final_ess_over_n = min(1.0, float((final_ess / config.num_particles).item()))
    report_values = np.asarray(
        [float(row["reported_value"]) for row in samples],
        dtype=np.float64,
    )
    normalized_weights = np.asarray(
        [float(row["normalized_weight"]) for row in samples],
        dtype=np.float64,
    )
    observable_name = str(getattr(observable, "name", "ari"))
    tail_prefix = "ari_ge" if observable_name == "ari" else "phi_ge"
    tail_counts = {
        f"{tail_prefix}_{threshold:g}": int(np.sum(report_values >= threshold))
        for threshold in config.tail_thresholds
    }
    tail_proportions = {
        key: float(value / config.num_particles) for key, value in tail_counts.items()
    }
    weighted_tail_probabilities = {
        f"{tail_prefix}_{threshold:g}": float(
            np.sum(normalized_weights[report_values >= threshold])
        )
        for threshold in config.tail_thresholds
    }
    acceptance_rate = total_accepted / total_proposals if total_proposals else 0.0
    unique_fraction = len({row["completion"] for row in samples}) / config.num_particles
    level_runtimes = [float(row["runtime_seconds"]) for row in level_diagnostics[1:]]
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    observable_counters_after = _observable_counter_snapshot(observable)
    observable_counter_deltas = {
        name: observable_counters_after[name] - observable_counters_before[name]
        for name in observable_counters_after
    }
    summary = {
        "method": "suffix_direct_mh_smc",
        "algorithm": "suffix_direct_mh_smc",
        "algorithm_version": ALGORITHM_VERSION,
        "mutation_kernel": "suffix_direct_mh",
        "observable": observable_name,
        "observable_configuration": observable_configuration(observable),
        "tilt_convention": "gamma_lambda(x) = p_M(x|c) * exp(-lambda * phi(x))",
        "ari_convention": getattr(observable, "convention", None),
        "num_particles": config.num_particles,
        "max_new_tokens": config.max_new_tokens,
        "lambdas": [float(value) for value in config.lambdas],
        "target_lambda": float(config.lambdas[-1]),
        "mcmc_steps_per_level": config.mcmc_steps_per_level,
        "ari_mode": getattr(observable, "mode", None),
        "ess_threshold": config.ess_threshold,
        "resampling": config.resampling,
        "resampling_method": config.resampling_method,
        "resample_final": bool(config.resample_final),
        "resampling_steps": resampling_steps,
        "initialization": initialization,
        "initial_ess_over_n": float(initial_ess_over_n),
        "initial_log_normalizing_constant_estimate": float(
            initial_log_normalizer.item()
        ),
        "log_normalizing_constant_estimate": float(
            log_normalizing_constant.item()
        ),
        "runtime_total_seconds": float(elapsed),
        "runtime_seconds_per_final_particle": float(elapsed / config.num_particles),
        "particles_per_second": float(config.num_particles / elapsed),
        "mean_runtime_seconds_per_level": float(np.mean(level_runtimes)) if level_runtimes else 0.0,
        "cuda_peak_memory_bytes": peak_memory,
        "final_ess_over_n": final_ess_over_n,
        "final_weight_ess_over_n_before_resampling": float(
            final_weight_ess_over_n_before_resampling
        ),
        "min_ess_over_n": float(min_ess_over_n),
        "mean_mcmc_acceptance_rate": float(acceptance_rate),
        "total_proposals": int(total_proposals),
        "total_regenerated_tokens": int(total_regenerated_tokens),
        "proposals_per_second": float(total_proposals / elapsed) if total_proposals else 0.0,
        "regenerated_tokens_per_second": (
            float(total_regenerated_tokens / elapsed) if total_regenerated_tokens else 0.0
        ),
        "runtime_profile_enabled": bool(profile_runtime),
        "generator_runtime_seconds": (
            float(generator_runtime_seconds) if profile_runtime else None
        ),
        "observable_runtime_seconds": (
            float(observable_runtime_seconds) if profile_runtime else None
        ),
        "generator_batch_calls": int(generation_diagnostics.batch_calls),
        "generator_model_forward_calls": int(
            generation_diagnostics.model_forward_calls
        ),
        "generated_token_transitions": int(
            generation_diagnostics.generated_token_transitions
        ),
        "observable_cache_hits": int(observable_counter_deltas["cache_hits"]),
        "observable_cache_misses": int(observable_counter_deltas["cache_misses"]),
        "observable_evaluation_batches": int(
            observable_counter_deltas["evaluation_batches"]
        ),
        "observable_evaluated_completions": int(
            observable_counter_deltas["evaluated_completions"]
        ),
        "observable_inference_seconds": float(
            observable_counter_deltas["inference_seconds"]
        ),
        "unweighted_mean_observable": float(np.mean(report_values)),
        "weighted_mean_observable": float(
            np.sum(normalized_weights * report_values)
        ),
        "min_observable": float(np.min(report_values)),
        "max_observable": float(np.max(report_values)),
        "tail_counts": tail_counts,
        "tail_proportions": tail_proportions,
        "weighted_tail_probabilities": weighted_tail_probabilities,
        "unique_completion_fraction": float(unique_fraction),
        "final_distinct_root_ancestors": len(
            {int(row["ancestor_id"]) for row in samples}
        ),
        "tail_discovery": {
            f"{tail_prefix}_{threshold:g}": {
                "first": first_tail_discovery[float(threshold)],
                "unique_completions": len(discovered_by_threshold[float(threshold)]),
            }
            for threshold in config.tail_thresholds
        },
        "top_discovered_completions": sorted(
            best_discoveries.values(),
            key=lambda row: float(row["reported_value"]),
            reverse=True,
        )[:20],
    }
    if observable_name == "ari":
        summary.update(
            {
                "unweighted_mean_ari": summary["unweighted_mean_observable"],
                "weighted_mean_ari": summary["weighted_mean_observable"],
                "min_ari": summary["min_observable"],
                "max_ari": summary["max_observable"],
            }
        )
        for row in level_diagnostics:
            row["weighted_mean_ari_before_resampling"] = row[
                "weighted_mean_observable_before_resampling"
            ]
            row["unweighted_mean_ari_after_resampling"] = row[
                "unweighted_mean_observable_after_resampling"
            ]
    diagnostics = {
        "level_diagnostics": level_diagnostics,
        "total_mcmc_accepted": total_accepted,
        "total_mcmc_proposals": total_proposals,
        "resampling_steps": resampling_steps,
        "final_distinct_root_ancestors": summary["final_distinct_root_ancestors"],
        "observable_device_boundary": (
            "The selected observable is evaluated once per batched proposal step; "
            "current-particle phi values stay cached. Proposal generation, acceptance, "
            "weights, ESS, and resampling remain batched. Exact direct-MH cancellation "
            "means no forward or reverse suffix probability is evaluated."
        ),
    }
    if capture_level_particles:
        diagnostics["level_particles"] = level_particles
    return SuffixDirectMHSMCResult(
        samples=samples,
        summary=summary,
        diagnostics=diagnostics,
        config={**asdict(config), "algorithm_version": ALGORITHM_VERSION},
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_report(summary: dict[str, Any], diagnostics: dict[str, Any]) -> str:
    observable_name = summary.get("observable", "ari")
    metric_label = "raw ARI" if observable_name == "ari" else "harmfulness phi"
    lines = [
        "# Suffix direct-MH SMC",
        "",
        "Particles are complete fixed-length LM completions. Mutation regenerates a "
        "state-independent suffix from the base model at temperature one. The exact "
        "direct-MH cancellation avoids proposal-probability evaluation. No target "
        "normalizer is evaluated.",
        "",
        "## Summary",
        "",
        "- Mutation kernel: `suffix_direct_mh`",
        f"- Observable: `{observable_name}`",
        f"- Observable mode: `{summary.get('ari_mode', 'n/a')}`",
        f"- Runtime: {summary['runtime_total_seconds']:.3f} s",
        f"- Seconds per final particle: {summary['runtime_seconds_per_final_particle']:.6f}",
        f"- Final serialized ESS/N: {summary['final_ess_over_n']:.4f}",
        "- Final pre-resampling ESS/N: "
        f"{summary['final_weight_ess_over_n_before_resampling']:.4f}",
        f"- Minimum pre-resampling ESS/N: {summary['min_ess_over_n']:.4f}",
        f"- Resampling steps: {summary['resampling_steps']} (including final when enabled)",
        f"- Total MCMC acceptance rate: {summary['mean_mcmc_acceptance_rate']:.4f}",
        f"- Total proposals: {summary.get('total_proposals', 0)}",
        f"- Total regenerated proposal tokens: {summary.get('total_regenerated_tokens', 0)}",
        f"- Unweighted mean {metric_label}: "
        f"{summary['unweighted_mean_observable']:.4f}",
        f"- Weighted mean {metric_label}: {summary['weighted_mean_observable']:.4f}",
        f"- Maximum {metric_label}: {summary['max_observable']:.4f}",
        f"- Unique completion fraction: {summary['unique_completion_fraction']:.4f}",
        f"- Distinct root ancestors: {summary['final_distinct_root_ancestors']}",
        "",
        f"## {metric_label} thresholds",
        "",
        "| threshold | raw count | raw proportion | weighted probability |",
        "|---:|---:|---:|---:|",
    ]
    for key, count in summary["tail_counts"].items():
        threshold = key.split("_ge_", maxsplit=1)[-1]
        lines.append(
            f"| {threshold} | {count} | {summary['tail_proportions'][key]:.6f} | "
            f"{summary['weighted_tail_probabilities'][key]:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Per-level diagnostics",
            "",
            "| level | lambda | mutation lambda | ESS/N pre | ESS/N post | resampled | "
            "acceptance | mean acceptance probability | weighted observable pre | "
            "unweighted observable post |",
            "|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostics["level_diagnostics"]:
        mutation_lambda = row["mutation_target_lambda"]
        acceptance = row["acceptance_rate"]
        mean_acceptance = row["mean_acceptance_probability"]
        lines.append(
            f"| {row['level']} | {row['lambda']:.4g} | "
            f"{'-' if mutation_lambda is None else f'{mutation_lambda:.4g}'} | "
            f"{row['ess_over_n_before_resampling']:.4f} | "
            f"{row['ess_over_n_after_resampling']:.4f} | {row['resampled']} | "
            f"{'-' if acceptance is None else f'{acceptance:.4f}'} | "
            f"{'-' if mean_acceptance is None else f'{mean_acceptance:.4f}'} | "
            f"{row['weighted_mean_observable_before_resampling']:.4f} | "
            f"{row['unweighted_mean_observable_after_resampling']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Language-model generation and SMC tensor operations are batched on the "
            "selected model device; observable scoring uses its configured device.",
            "",
        ]
    )
    return "\n".join(lines)
