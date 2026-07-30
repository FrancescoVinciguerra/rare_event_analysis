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

from rea_llms.models import set_seed


ScheduleValue = TypeVar("ScheduleValue")


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
    raw_ari: torch.Tensor
    root_ancestors: torch.Tensor
    last_suffix_sizes: torch.Tensor


@dataclass
class MutationDiagnostics:
    """Small host-side summary returned after one mutation level."""

    accepted: int
    proposals: int
    mean_acceptance_probability: float
    suffix_size_counts: dict[str, int]
    regenerated_tokens: int


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


# Batched language-model proposal primitives


def _base_model_parts(model) -> tuple[Any | None, Any | None]:
    base_model = getattr(model, "base_model", None)
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    output_embeddings = get_output_embeddings() if callable(get_output_embeddings) else None
    if base_model is None or base_model is model or output_embeddings is None:
        return None, None
    return base_model, output_embeddings


def _project_hidden(model, output_embeddings, hidden: torch.Tensor) -> torch.Tensor:
    logits = output_embeddings(hidden)
    final_bias = getattr(model, "final_logits_bias", None)
    return logits if final_bias is None else logits + final_bias


def _full_inputs(prompt_ids: torch.Tensor, completions: torch.Tensor) -> torch.Tensor:
    prompt = prompt_ids.unsqueeze(0).expand(completions.shape[0], -1)
    return torch.cat((prompt, completions), dim=1)


def _selected_next_token_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    base_model, output_embeddings = _base_model_parts(model)
    if base_model is not None:
        outputs = base_model(input_ids=input_ids, use_cache=False, return_dict=True)
        return _project_hidden(model, output_embeddings, outputs.last_hidden_state[:, -1])
    return model(input_ids=input_ids).logits[:, -1]


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _generate_suffix_chunk(
    model,
    context_ids: torch.Tensor,
    *,
    suffix_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate an exact fixed-length LM suffix for an equal-length batch.

    Particles in the batch are sampled in parallel, but future tokens of one
    completion remain sequential because every next-token distribution depends
    on the tokens already sampled. For GPT-Neo, ``base_model`` exposes the KV
    cache and the output head can be applied only to the latest hidden state.
    The fallback is semantically equivalent but recomputes the growing prefix.

    ``model.generate`` is not assumed equivalent or faster automatically. An
    equivalent replacement would need unrestricted multinomial sampling at
    temperature one, no EOS stopping, penalties, beams, top-k/top-p filtering,
    or length normalization, plus validation of cache and RNG behavior.
    """
    base_model, output_embeddings = _base_model_parts(model)
    # Length is fixed, so write each autoregressive draw directly into its final
    # column instead of retaining one allocation per token for a final cat.
    generated = torch.empty(
        (context_ids.shape[0], suffix_size),
        dtype=context_ids.dtype,
        device=context_ids.device,
    )
    if base_model is not None:
        outputs = base_model(input_ids=context_ids, use_cache=True, return_dict=True)
        hidden = outputs.last_hidden_state[:, -1]
        past = getattr(outputs, "past_key_values", None)
        work = context_ids
        for step in range(suffix_size):
            logits = _project_hidden(model, output_embeddings, hidden)
            token = torch.multinomial(
                torch.softmax(logits, dim=-1),
                num_samples=1,
                generator=generator,
            )
            generated[:, step : step + 1].copy_(token)
            if step + 1 == suffix_size:
                break
            if past is not None:
                # Only the newly sampled token crosses the transformer when a
                # model supplies a usable KV cache.
                outputs = base_model(
                    input_ids=token,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                hidden = outputs.last_hidden_state[:, -1]
                past = getattr(outputs, "past_key_values", None)
            else:
                # Cache-less models must re-evaluate the growing context. This
                # preserves the proposal distribution but is substantially
                # more expensive for long suffixes.
                work = torch.cat((work, token), dim=1)
                hidden = base_model(
                    input_ids=work,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state[:, -1]
        return generated

    work = context_ids
    for step in range(suffix_size):
        logits = _selected_next_token_logits(model, work)
        token = torch.multinomial(
            torch.softmax(logits, dim=-1),
            num_samples=1,
            generator=generator,
        )
        generated[:, step : step + 1].copy_(token)
        work = torch.cat((work, token), dim=1)
    return generated


def _generate_suffix_with_backoff(
    model,
    context_ids: torch.Tensor,
    *,
    suffix_size: int,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate in configurable chunks, retrying smaller chunks after CUDA OOM.

    Chunking changes random-number consumption relative to particlewise code,
    but every returned suffix is still sampled from the same LM proposal.
    """
    chunks: list[torch.Tensor] = []
    start = 0
    active_batch_size = min(batch_size, context_ids.shape[0])
    while start < context_ids.shape[0]:
        stop = min(start + active_batch_size, context_ids.shape[0])
        try:
            chunks.append(
                _generate_suffix_chunk(
                    model,
                    context_ids[start:stop],
                    suffix_size=suffix_size,
                    generator=generator,
                )
            )
            start = stop
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            if not _is_cuda_oom(exc) or active_batch_size == 1:
                raise RuntimeError(
                    f"CUDA OOM while generating suffix size {suffix_size} at batch size "
                    f"{active_batch_size}; reduce --batch-size"
                ) from exc
            active_batch_size = max(1, active_batch_size // 2)
            if context_ids.device.type == "cuda":
                torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def propose_suffixes_parallel(
    model,
    prompt_ids: torch.Tensor,
    current: torch.Tensor,
    suffix_sizes: torch.Tensor,
    *,
    batch_size: int,
    generator: torch.Generator,
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
        proposed_suffix = _generate_suffix_with_backoff(
            model,
            context,
            suffix_size=int(suffix_size),
            batch_size=batch_size,
            generator=generator,
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
    prompt_ids: torch.Tensor,
    completions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[Any]]:
    """Evaluate exact text ARI at this function's batched GPU-to-CPU boundary.

    Full token sequences are transferred together, then completion tokens are
    decoded with the repository observable's convention before textstat runs on
    CPU. Decoding the actual token sequence is essential: BPE boundaries,
    punctuation, repeated tokens, and concatenated words make token-local ARI
    arithmetic unsafe unless exact textual equivalence is proved.
    """
    # Include prompt token IDs so score_token_sequences can slice and decode
    # the completion exactly as the repository's final ARI implementation.
    full_ids = _full_inputs(prompt_ids, completions).detach().cpu().tolist()
    score_batch = getattr(observable, "score_token_sequences", None)
    if callable(score_batch):
        scores = score_batch(
            tokenizer=tokenizer,
            prompt=prompt,
            token_sequences=full_ids,
            prompt_token_count=int(prompt_ids.numel()),
        )
    else:
        completion_ids = completions.detach().cpu().tolist()
        texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        scores = [observable.score_completion(prompt, text) for text in texts]
    phi = torch.tensor(
        [float(score.value) for score in scores],
        dtype=torch.float32,
        device=completions.device,
    )
    raw_ari = torch.tensor(
        [float(score.metadata["raw_ari"]) for score in scores],
        dtype=torch.float32,
        device=completions.device,
    )
    if not bool(torch.isfinite(phi).all().item()) or not bool(
        torch.isfinite(raw_ari).all().item()
    ):
        raise ValueError("Observable returned a non-finite value")
    return phi, raw_ari, scores


def generate_base_completions(
    model,
    prompt_ids: torch.Tensor,
    *,
    num_particles: int,
    max_new_tokens: int,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Initialize all particles from p_M using the batched autoregressive path."""
    contexts = prompt_ids.unsqueeze(0).expand(num_particles, -1)
    return _generate_suffix_with_backoff(
        model,
        contexts,
        suffix_size=max_new_tokens,
        batch_size=batch_size,
        generator=generator,
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
    updated = normalized_log_weights - (current_lambda - previous_lambda) * phi.to(torch.float64)
    return normalize_log_weights(updated)


# Full SMC mutation, reweighting, ESS, and resampling


def apply_suffix_direct_mh_mutation(
    particles: ParticleBatch,
    model,
    tokenizer,
    observable,
    *,
    prompt: str,
    prompt_ids: torch.Tensor,
    target_lambda: float,
    transition_index: int,
    config: SuffixDirectMHSMCConfig,
    generator: torch.Generator,
    discovery_callback: Callable[[torch.Tensor, torch.Tensor, float], None] | None = None,
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
    regenerated_tokens = torch.zeros((), dtype=torch.long, device=device)
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
        proposal_tokens = propose_suffixes_parallel(
            model,
            prompt_ids,
            particles.completion_token_ids,
            suffix_sizes,
            batch_size=config.batch_size,
            generator=generator,
        )
        # Exact text ARI is the deliberate CPU boundary in each mutation step;
        # current-particle scores remain cached in ParticleBatch.
        proposal_phi, proposal_raw_ari, _ = score_observable(
            tokenizer,
            observable,
            prompt=prompt,
            prompt_ids=prompt_ids,
            completions=proposal_tokens,
        )
        if discovery_callback is not None:
            discovery_callback(proposal_tokens, proposal_raw_ari, target_lambda)
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
            raw_ari=torch.where(accepted, proposal_raw_ari, particles.raw_ari),
            root_ancestors=particles.root_ancestors,
            last_suffix_sizes=suffix_sizes,
        )
        accepted_total += accepted.sum()
        acceptance_probability_sum += torch.exp(log_acceptance).to(torch.float64).sum()
        size_counts += torch.bincount(suffix_sizes, minlength=config.max_new_tokens + 1)
        regenerated_tokens += suffix_sizes.sum()

    denominator = max(1, proposals_total)
    counts_on_cpu = size_counts.detach().cpu().tolist()
    suffix_count_rows = {
        str(size): int(counts_on_cpu[size])
        for size in range(1, config.max_new_tokens + 1)
        if counts_on_cpu[size] > 0
    }
    return particles, MutationDiagnostics(
        accepted=int(accepted_total.item()),
        proposals=proposals_total,
        mean_acceptance_probability=float(
            (acceptance_probability_sum / denominator).item()
        ),
        suffix_size_counts=suffix_count_rows,
        regenerated_tokens=int(regenerated_tokens.item()),
    )


def _prompt_token_ids(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt")["input_ids"]
    if not isinstance(encoded, torch.Tensor):
        encoded = torch.tensor(encoded, dtype=torch.long)
    return encoded[0].to(device=device, dtype=torch.long)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _final_samples(
    tokenizer,
    observable,
    *,
    prompt: str,
    prompt_ids: torch.Tensor,
    particles: ParticleBatch,
    normalized_log_weights: torch.Tensor,
    seed: int,
    lambda_final: float,
) -> list[dict[str, Any]]:
    """Serialize the final weighted particle system without resampling it.

    This currently reconstructs score metadata and decoded strings on CPU. The
    repeated work is retained during this audit because changing it would need
    an explicit equivalence test for every serialized field.
    """
    _, _, scores = score_observable(
        tokenizer,
        observable,
        prompt=prompt,
        prompt_ids=prompt_ids,
        completions=particles.completion_token_ids,
    )
    completion_ids = particles.completion_token_ids.detach().cpu().tolist()
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    weights = torch.exp(normalized_log_weights).detach().cpu().tolist()
    log_weights = normalized_log_weights.detach().cpu().tolist()
    phi = particles.phi.detach().cpu().tolist()
    roots = particles.root_ancestors.detach().cpu().tolist()
    last_sizes = particles.last_suffix_sizes.detach().cpu().tolist()
    rows: list[dict[str, Any]] = []
    for index, (ids, completion, score) in enumerate(zip(completion_ids, completions, scores)):
        metadata = dict(score.metadata)
        rows.append(
            {
                "particle_id": index,
                "ancestor_id": int(roots[index]),
                "root_ancestor": int(roots[index]),
                "seed": int(seed),
                "text": prompt + completion,
                "completion": completion,
                "phi": float(phi[index]),
                "raw_ari": float(metadata["raw_ari"]),
                "capped_ari": float(metadata["capped_ari"]),
                "log_weight": float(log_weights[index]),
                "normalized_weight": float(weights[index]),
                "lambda_final": float(lambda_final),
                "mutation_kernel": "suffix_direct_mh",
                "last_suffix_size": int(last_sizes[index]),
                "completion_token_ids": ids,
                "metadata": metadata,
            }
        )
    return rows


def run_suffix_direct_mh_smc(
    model,
    tokenizer,
    observable,
    *,
    prompt: str,
    config: SuffixDirectMHSMCConfig,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> SuffixDirectMHSMCResult:
    """Run full-sequence SMC with an exact direct-MH suffix mutation.

    The order at transition k is mutation under pi_{k-1}, incremental weighting
    from gamma_{k-1} to gamma_k, then optional systematic resampling. The final
    resampling is intentionally omitted so returned normalized weights retain
    the final target approximation.
    """
    set_seed(config.seed)
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(config.seed)
    prompt_ids = _prompt_token_ids(tokenizer, prompt, device)
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

    def track_discoveries(
        token_batch: torch.Tensor,
        raw_ari_batch: torch.Tensor,
        lambda_value: float,
    ) -> None:
        # Discovery statistics include every generated proposal, accepted or
        # rejected. This requires a second token transfer/decode beyond ARI
        # scoring and is intentionally preserved as part of metric semantics.
        completion_ids = token_batch.detach().cpu().tolist()
        completion_texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        raw_values = raw_ari_batch.detach().cpu().tolist()
        elapsed_now = time.perf_counter() - start
        for completion, raw_value in zip(completion_texts, raw_values):
            value = float(raw_value)
            previous = best_discoveries.get(completion)
            if previous is None or value > float(previous["raw_ari"]):
                best_discoveries[completion] = {
                    "raw_ari": value,
                    "lambda": float(lambda_value),
                    "time_first_seen_seconds": float(elapsed_now),
                    "completion": completion,
                }
            for threshold in config.tail_thresholds:
                threshold_value = float(threshold)
                if value < threshold_value:
                    continue
                discovered_by_threshold[threshold_value].add(completion)
                if first_tail_discovery[threshold_value] is None:
                    first_tail_discovery[threshold_value] = {
                        "raw_ari": value,
                        "lambda": float(lambda_value),
                        "time_first_seen_seconds": float(elapsed_now),
                        "completion": completion,
                    }

    with torch.inference_mode():
        completion_tokens = generate_base_completions(
            model,
            prompt_ids,
            num_particles=config.num_particles,
            max_new_tokens=config.max_new_tokens,
            batch_size=config.batch_size,
            generator=generator,
        )
        phi, raw_ari, _ = score_observable(
            tokenizer,
            observable,
            prompt=prompt,
            prompt_ids=prompt_ids,
            completions=completion_tokens,
        )
        track_discoveries(completion_tokens, raw_ari, float(config.lambdas[0]))
        particles = ParticleBatch(
            completion_token_ids=completion_tokens,
            phi=phi,
            raw_ari=raw_ari,
            root_ancestors=torch.arange(config.num_particles, device=device),
            last_suffix_sizes=torch.zeros(config.num_particles, dtype=torch.long, device=device),
        )
        log_weights = torch.full(
            (config.num_particles,),
            uniform_log_weight,
            dtype=torch.float64,
            device=device,
        )
        initial_mean = float(raw_ari.mean().item())
        level_diagnostics: list[dict[str, Any]] = [
            {
                "level": 0,
                "lambda": float(config.lambdas[0]),
                "mutation_target_lambda": None,
                "ess_over_n_before_resampling": 1.0,
                "ess_over_n_after_resampling": 1.0,
                "resampled": False,
                "accepted": 0,
                "proposals": 0,
                "acceptance_rate": None,
                "mean_acceptance_probability": None,
                "weighted_mean_ari_before_resampling": initial_mean,
                "unweighted_mean_ari_after_resampling": initial_mean,
                "runtime_seconds": 0.0,
                "suffix_size_counts": {},
            }
        ]
        total_accepted = 0
        total_proposals = 0
        total_regenerated_tokens = 0
        resampling_steps = 0
        min_ess_over_n = 1.0

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
                prompt_ids=prompt_ids,
                target_lambda=previous_lambda,
                transition_index=level - 1,
                config=config,
                generator=generator,
                discovery_callback=track_discoveries,
            )
            total_accepted += mutation.accepted
            total_proposals += mutation.proposals
            total_regenerated_tokens += mutation.regenerated_tokens

            # 2. Since p_M is common to adjacent targets, only the exponential
            # potential ratio is needed to move from gamma_{k-1} to gamma_k.
            log_weights = incremental_log_weight_update(
                log_weights,
                particles.phi,
                previous_lambda=previous_lambda,
                current_lambda=current_lambda,
            )
            # 3. Diagnose normalized weights and resample only before the final
            # level, where equal weights can benefit subsequent mutation.
            ess = ess_from_normalized_log_weights(log_weights)
            ess_over_n_before = min(1.0, float((ess / config.num_particles).item()))
            weights_before = torch.exp(log_weights)
            weighted_mean_before = float(
                torch.sum(weights_before * particles.raw_ari.to(torch.float64)).item()
            )
            min_ess_over_n = min(min_ess_over_n, ess_over_n_before)

            # Keep final weights: resampling there would discard the information
            # needed by both the final target estimator and the SNIS correction.
            resampled = bool(
                config.resampling
                and level < len(config.lambdas) - 1
                and ess_over_n_before < config.ess_threshold
            )
            if resampled:
                indices = systematic_resample(log_weights, generator=generator)
                particles = ParticleBatch(
                    completion_token_ids=particles.completion_token_ids.index_select(
                        0, indices
                    ),
                    phi=particles.phi.index_select(0, indices),
                    raw_ari=particles.raw_ari.index_select(0, indices),
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
                "ess_over_n_before_resampling": ess_over_n_before,
                "ess_over_n_after_resampling": ess_over_n_after,
                "resampled": resampled,
                "accepted": mutation.accepted,
                "proposals": mutation.proposals,
                "acceptance_rate": (
                    mutation.accepted / mutation.proposals if mutation.proposals else None
                ),
                "mean_acceptance_probability": mutation.mean_acceptance_probability,
                "weighted_mean_ari_before_resampling": weighted_mean_before,
                "unweighted_mean_ari_after_resampling": float(particles.raw_ari.mean().item()),
                "runtime_seconds": float(time.perf_counter() - level_start),
                "suffix_size_counts": mutation.suffix_size_counts,
            }
            level_diagnostics.append(row)
            if progress_callback is not None:
                progress_callback(row)

        samples = _final_samples(
            tokenizer,
            observable,
            prompt=prompt,
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
    raw_values = np.asarray([float(row["raw_ari"]) for row in samples], dtype=np.float64)
    normalized_weights = np.asarray(
        [float(row["normalized_weight"]) for row in samples],
        dtype=np.float64,
    )
    tail_counts = {
        f"ari_ge_{threshold:g}": int(np.sum(raw_values >= threshold))
        for threshold in config.tail_thresholds
    }
    tail_proportions = {
        key: float(value / config.num_particles) for key, value in tail_counts.items()
    }
    weighted_tail_probabilities = {
        f"ari_ge_{threshold:g}": float(np.sum(normalized_weights[raw_values >= threshold]))
        for threshold in config.tail_thresholds
    }
    acceptance_rate = total_accepted / total_proposals if total_proposals else 0.0
    unique_fraction = len({row["completion"] for row in samples}) / config.num_particles
    level_runtimes = [float(row["runtime_seconds"]) for row in level_diagnostics[1:]]
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    summary = {
        "method": "suffix_direct_mh_smc",
        "algorithm": "suffix_direct_mh_smc",
        "mutation_kernel": "suffix_direct_mh",
        "num_particles": config.num_particles,
        "max_new_tokens": config.max_new_tokens,
        "lambdas": [float(value) for value in config.lambdas],
        "target_lambda": float(config.lambdas[-1]),
        "mcmc_steps_per_level": config.mcmc_steps_per_level,
        "ari_mode": getattr(observable, "mode", "unknown"),
        "ess_threshold": config.ess_threshold,
        "resampling": config.resampling,
        "resampling_method": config.resampling_method,
        "resample_final": False,
        "resampling_steps": resampling_steps,
        "runtime_total_seconds": float(elapsed),
        "runtime_seconds_per_final_particle": float(elapsed / config.num_particles),
        "particles_per_second": float(config.num_particles / elapsed),
        "mean_runtime_seconds_per_level": float(np.mean(level_runtimes)) if level_runtimes else 0.0,
        "cuda_peak_memory_bytes": peak_memory,
        "final_ess_over_n": final_ess_over_n,
        "min_ess_over_n": float(min_ess_over_n),
        "mean_mcmc_acceptance_rate": float(acceptance_rate),
        "total_proposals": int(total_proposals),
        "total_regenerated_tokens": int(total_regenerated_tokens),
        "proposals_per_second": float(total_proposals / elapsed) if total_proposals else 0.0,
        "regenerated_tokens_per_second": (
            float(total_regenerated_tokens / elapsed) if total_regenerated_tokens else 0.0
        ),
        "unweighted_mean_ari": float(np.mean(raw_values)),
        "weighted_mean_ari": float(np.sum(normalized_weights * raw_values)),
        "min_ari": float(np.min(raw_values)),
        "max_ari": float(np.max(raw_values)),
        "tail_counts": tail_counts,
        "tail_proportions": tail_proportions,
        "weighted_tail_probabilities": weighted_tail_probabilities,
        "unique_completion_fraction": float(unique_fraction),
        "final_distinct_root_ancestors": len(
            {int(row["ancestor_id"]) for row in samples}
        ),
        "tail_discovery": {
            f"ari_ge_{threshold:g}": {
                "first": first_tail_discovery[float(threshold)],
                "unique_completions": len(discovered_by_threshold[float(threshold)]),
            }
            for threshold in config.tail_thresholds
        },
        "top_discovered_completions": sorted(
            best_discoveries.values(),
            key=lambda row: float(row["raw_ari"]),
            reverse=True,
        )[:20],
    }
    diagnostics = {
        "level_diagnostics": level_diagnostics,
        "total_mcmc_accepted": total_accepted,
        "total_mcmc_proposals": total_proposals,
        "resampling_steps": resampling_steps,
        "final_distinct_root_ancestors": summary["final_distinct_root_ancestors"],
        "observable_device_boundary": (
            "Exact textstat ARI is decoded and evaluated on CPU once per batched proposal "
            "step. Proposal generation, acceptance, weights, ESS, and resampling remain "
            "on the model device. Exact direct-MH cancellation means no forward or "
            "reverse suffix probability is evaluated."
        ),
    }
    return SuffixDirectMHSMCResult(
        samples=samples,
        summary=summary,
        diagnostics=diagnostics,
        config=asdict(config),
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
        f"- ARI mode: `{summary.get('ari_mode', 'unknown')}`",
        f"- Runtime: {summary['runtime_total_seconds']:.3f} s",
        f"- Seconds per final particle: {summary['runtime_seconds_per_final_particle']:.6f}",
        f"- Final ESS/N: {summary['final_ess_over_n']:.4f}",
        f"- Minimum pre-resampling ESS/N: {summary['min_ess_over_n']:.4f}",
        f"- Resampling steps: {summary['resampling_steps']} (never at the final level)",
        f"- Total MCMC acceptance rate: {summary['mean_mcmc_acceptance_rate']:.4f}",
        f"- Total proposals: {summary.get('total_proposals', 0)}",
        f"- Total regenerated proposal tokens: {summary.get('total_regenerated_tokens', 0)}",
        f"- Unweighted mean raw ARI: {summary['unweighted_mean_ari']:.4f}",
        f"- Weighted mean raw ARI: {summary['weighted_mean_ari']:.4f}",
        f"- Maximum raw ARI: {summary['max_ari']:.4f}",
        f"- Unique completion fraction: {summary['unique_completion_fraction']:.4f}",
        f"- Distinct root ancestors: {summary['final_distinct_root_ancestors']}",
        "",
        "## ARI tails",
        "",
        "| threshold | raw count | raw proportion | weighted probability |",
        "|---:|---:|---:|---:|",
    ]
    for key, count in summary["tail_counts"].items():
        threshold = key.removeprefix("ari_ge_")
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
            "acceptance | mean acceptance probability | weighted ARI pre | "
            "unweighted ARI post |",
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
            f"{row['weighted_mean_ari_before_resampling']:.4f} | "
            f"{row['unweighted_mean_ari_after_resampling']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The exact text ARI boundary remains on CPU. All language-model and SMC tensor "
            "operations are batched on the selected model device.",
            "",
        ]
    )
    return "\n".join(lines)
