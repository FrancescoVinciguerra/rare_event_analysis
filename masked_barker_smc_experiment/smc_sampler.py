from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

import numpy as np
import torch
import torch.nn.functional as F

from rea_llms.models import set_seed


ScheduleValue = TypeVar("ScheduleValue")


@dataclass(frozen=True)
class MaskedBarkerSMCConfig:
    num_particles: int = 64
    max_new_tokens: int = 100
    lambdas: tuple[float, ...] = (0.0, -0.25, -0.5, -1.0)
    mcmc_steps_per_level: int = 5
    ess_threshold: float = 0.5
    resampling: bool = True
    resampling_method: str = "systematic"
    mask_mode: str = "stochastic"
    mask_sizes: tuple[int, ...] = (1, 2, 4, 8)
    mask_rho_schedule: tuple[tuple[float, ...], ...] | None = None
    deterministic_mask_size_schedule: tuple[tuple[int, ...], ...] | None = None
    mask_policy: str = "uniform_noncontiguous"
    explicit_masks: tuple[tuple[tuple[int, ...], ...], ...] | None = None
    proposal_temperature: float = 1.0
    batch_size: int = 64
    score_token_chunk_size: int = 8
    seed: int = 12345
    tail_thresholds: tuple[float, ...] = (6.0, 7.0, 8.0)

    def __post_init__(self) -> None:
        if self.num_particles <= 0:
            raise ValueError("num_particles must be > 0")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if not self.lambdas or not math.isclose(float(self.lambdas[0]), 0.0, abs_tol=1e-12):
            raise ValueError("lambdas must be non-empty and start at 0")
        if any(not math.isfinite(float(value)) for value in self.lambdas):
            raise ValueError("all lambdas must be finite")
        if self.mcmc_steps_per_level < 0:
            raise ValueError("mcmc_steps_per_level must be >= 0")
        if not 0.0 <= self.ess_threshold <= 1.0:
            raise ValueError("ess_threshold must be in [0, 1]")
        if self.resampling_method != "systematic":
            raise ValueError("resampling_method must be 'systematic'")
        if self.mask_mode not in {"stochastic", "deterministic"}:
            raise ValueError("mask_mode must be one of {'stochastic', 'deterministic'}")
        if self.mask_policy not in {"uniform_noncontiguous", "contiguous", "explicit"}:
            raise ValueError(
                "mask_policy must be one of {'uniform_noncontiguous', 'contiguous', 'explicit'}"
            )
        if not self.mask_sizes:
            raise ValueError("mask_sizes must be non-empty")
        if self.mask_policy != "explicit" and any(
            size <= 0 or size > self.max_new_tokens for size in self.mask_sizes
        ):
            raise ValueError("every mask size must be in [1, max_new_tokens]")
        if not math.isfinite(self.proposal_temperature) or self.proposal_temperature <= 0.0:
            raise ValueError("proposal_temperature must be finite and > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.score_token_chunk_size <= 0:
            raise ValueError("score_token_chunk_size must be > 0")
        if any(not math.isfinite(float(value)) for value in self.tail_thresholds):
            raise ValueError("tail_thresholds must be finite")

        transitions = max(0, len(self.lambdas) - 1)
        if self.mask_mode == "stochastic":
            self._validate_rho_schedule(transitions)
        else:
            self._validate_deterministic_schedule(transitions)
        if self.mask_policy == "explicit":
            if self.mask_mode != "deterministic":
                raise ValueError("mask_policy='explicit' requires mask_mode='deterministic'")
            self._validate_explicit_masks(transitions)
        elif self.explicit_masks is not None:
            raise ValueError("explicit_masks may only be used with mask_policy='explicit'")

    def _validate_rho_schedule(self, transitions: int) -> None:
        rows = self.mask_rho_schedule
        if rows is None:
            return
        if len(rows) not in {1, transitions}:
            raise ValueError("mask_rho_schedule must have one row or one row per SMC transition")
        for row in rows:
            if len(row) != len(self.mask_sizes):
                raise ValueError("each rho row must match the number of mask_sizes")
            if any(not math.isfinite(float(value)) or value < 0.0 for value in row):
                raise ValueError("rho values must be finite and non-negative")
            if not math.isclose(sum(row), 1.0, rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError("each rho row must sum to one")

    def _validate_deterministic_schedule(self, transitions: int) -> None:
        rows = self.deterministic_mask_size_schedule
        if rows is None:
            return
        if len(rows) not in {1, transitions}:
            raise ValueError(
                "deterministic_mask_size_schedule must have one row or one row per SMC transition"
            )
        valid_row_lengths = {1, self.mcmc_steps_per_level}
        for row in rows:
            if len(row) not in valid_row_lengths:
                raise ValueError(
                    "each deterministic mask-size row must have one value or one value per MCMC step"
                )
            if any(size <= 0 or size > self.max_new_tokens for size in row):
                raise ValueError("deterministic mask sizes must be in [1, max_new_tokens]")

    def _validate_explicit_masks(self, transitions: int) -> None:
        rows = self.explicit_masks
        if not rows:
            raise ValueError("explicit_masks is required when mask_policy='explicit'")
        if len(rows) not in {1, transitions}:
            raise ValueError("explicit_masks must have one level or one level per SMC transition")
        valid_step_counts = {1, self.mcmc_steps_per_level}
        for level_masks in rows:
            if len(level_masks) not in valid_step_counts:
                raise ValueError(
                    "each explicit-mask level must have one mask or one mask per MCMC step"
                )
            for mask in level_masks:
                if not mask:
                    raise ValueError("explicit masks must be non-empty")
                if len(set(mask)) != len(mask):
                    raise ValueError("explicit mask indices must be distinct")
                if any(index < 0 or index >= self.max_new_tokens for index in mask):
                    raise ValueError("explicit mask indices use zero-based completion positions in [0, T)")


@dataclass
class MaskBatch:
    positions: torch.Tensor
    valid: torch.Tensor
    sizes: torch.Tensor


@dataclass
class ParticleBatch:
    completion_token_ids: torch.Tensor
    phi: torch.Tensor
    raw_ari: torch.Tensor
    log_p_model: torch.Tensor
    root_ancestors: torch.Tensor


@dataclass
class MutationDiagnostics:
    accepted: int
    proposals: int
    mean_symmetrization_probability: float
    mean_barker_probability: float


@dataclass
class MaskedBarkerSMCResult:
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


def parse_explicit_masks(value: str) -> tuple[tuple[tuple[int, ...], ...], ...]:
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("explicit masks must be valid JSON") from exc
    if not isinstance(data, list) or not data:
        raise ValueError("explicit masks must be a non-empty JSON list")
    if all(isinstance(item, int) for item in data):
        data = [[data]]
    elif all(isinstance(item, list) and all(isinstance(index, int) for index in item) for item in data):
        data = [data]
    try:
        return tuple(
            tuple(tuple(int(index) for index in step_mask) for step_mask in level_masks)
            for level_masks in data
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("explicit masks must have shape [level][step][index]") from exc


def _schedule_row(
    rows: Sequence[ScheduleValue],
    transition_index: int,
) -> ScheduleValue:
    return rows[0] if len(rows) == 1 else rows[transition_index]


def _step_value(
    row: Sequence[ScheduleValue],
    step_index: int,
) -> ScheduleValue:
    return row[0] if len(row) == 1 else row[step_index]


def sample_masks(
    config: MaskedBarkerSMCConfig,
    *,
    transition_index: int,
    step_index: int,
    device: torch.device,
    generator: torch.Generator,
) -> MaskBatch:
    """Draw state-independent completion-token masks entirely on the model device."""
    n_particles = config.num_particles
    length = config.max_new_tokens

    if config.mask_policy == "explicit":
        assert config.explicit_masks is not None
        level_masks = _schedule_row(config.explicit_masks, transition_index)
        selected = tuple(sorted(_step_value(level_masks, step_index)))
        positions = torch.tensor(selected, dtype=torch.long, device=device).repeat(n_particles, 1)
        valid = torch.ones_like(positions, dtype=torch.bool)
        sizes = torch.full((n_particles,), len(selected), dtype=torch.long, device=device)
        return MaskBatch(positions=positions, valid=valid, sizes=sizes)

    if config.mask_mode == "stochastic":
        rows = config.mask_rho_schedule
        if rows is None:
            rho_values = tuple(1.0 / len(config.mask_sizes) for _ in config.mask_sizes)
            rho = torch.full(
                (len(config.mask_sizes),),
                1.0 / len(config.mask_sizes),
                dtype=torch.float32,
                device=device,
            )
        else:
            rho_values = _schedule_row(rows, transition_index)
            rho = torch.tensor(
                rho_values,
                dtype=torch.float32,
                device=device,
            )
        size_indices = torch.multinomial(rho, n_particles, replacement=True, generator=generator)
        size_options = torch.tensor(config.mask_sizes, dtype=torch.long, device=device)
        sizes = size_options[size_indices]
        max_size = max(
            size for size, probability in zip(config.mask_sizes, rho_values) if probability > 0.0
        )
    else:
        rows = config.deterministic_mask_size_schedule or ((config.mask_sizes[0],),)
        row = _schedule_row(rows, transition_index)
        size = int(_step_value(row, step_index))
        sizes = torch.full((n_particles,), size, dtype=torch.long, device=device)
        max_size = size

    rank = torch.arange(max_size, device=device).unsqueeze(0)
    valid = rank < sizes.unsqueeze(1)
    if config.mask_policy == "contiguous":
        possible_starts = length - sizes + 1
        starts = torch.floor(
            torch.rand(n_particles, device=device, generator=generator) * possible_starts
        ).to(torch.long)
        positions = starts.unsqueeze(1) + rank
        positions = torch.where(valid, positions, torch.zeros_like(positions))
        return MaskBatch(positions=positions, valid=valid, sizes=sizes)

    random_order = torch.argsort(
        torch.rand((n_particles, length), device=device, generator=generator),
        dim=1,
    )
    chosen = random_order[:, :max_size]
    sentinel = torch.full_like(chosen, length)
    positions = torch.sort(torch.where(valid, chosen, sentinel), dim=1).values
    positions = torch.where(valid, positions, torch.zeros_like(positions))
    return MaskBatch(positions=positions, valid=valid, sizes=sizes)


def _base_model_parts(model) -> tuple[Any | None, Any | None]:
    base_model = getattr(model, "base_model", None)
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    output_embeddings = get_output_embeddings() if callable(get_output_embeddings) else None
    if base_model is None or base_model is model or output_embeddings is None:
        return None, None
    return base_model, output_embeddings


def _project_selected_hidden(model, output_embeddings, hidden: torch.Tensor) -> torch.Tensor:
    logits = output_embeddings(hidden)
    final_bias = getattr(model, "final_logits_bias", None)
    if final_bias is not None:
        logits = logits + final_bias
    return logits


def selected_next_token_logits(
    model,
    input_ids: torch.Tensor,
    predictor_positions: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Return only the vocabulary logits needed by each row.

    For Hugging Face causal LMs this avoids materializing [batch, length, vocab].
    """
    chunks: list[torch.Tensor] = []
    base_model, output_embeddings = _base_model_parts(model)
    for start in range(0, input_ids.shape[0], batch_size):
        stop = min(start + batch_size, input_ids.shape[0])
        ids = input_ids[start:stop]
        positions = predictor_positions[start:stop]
        row = torch.arange(stop - start, device=ids.device)
        if base_model is not None:
            outputs = base_model(input_ids=ids, use_cache=False, return_dict=True)
            hidden = outputs.last_hidden_state[row, positions]
            logits = _project_selected_hidden(model, output_embeddings, hidden)
        else:
            logits = model(input_ids=ids).logits[row, positions]
        chunks.append(logits)
    return torch.cat(chunks, dim=0)


def completion_log_probabilities(
    model,
    prompt_ids: torch.Tensor,
    completions: torch.Tensor,
    *,
    batch_size: int,
    token_chunk_size: int,
) -> torch.Tensor:
    """Compute log p_M(completion | prompt) without a [N, T, |V|] tensor."""
    n_particles, length = completions.shape
    prompt_length = int(prompt_ids.numel())
    if prompt_length == 0:
        raise ValueError("The prompt must contain at least one token")
    result = torch.empty(n_particles, dtype=torch.float32, device=completions.device)
    base_model, output_embeddings = _base_model_parts(model)

    for start in range(0, n_particles, batch_size):
        stop = min(start + batch_size, n_particles)
        completion_chunk = completions[start:stop]
        prompt_chunk = prompt_ids.unsqueeze(0).expand(stop - start, -1)
        full_ids = torch.cat((prompt_chunk, completion_chunk), dim=1)
        targets = completion_chunk
        if base_model is not None:
            outputs = base_model(input_ids=full_ids, use_cache=False, return_dict=True)
            predictor_hidden = outputs.last_hidden_state[:, prompt_length - 1 : prompt_length + length - 1]
            log_p = torch.zeros(stop - start, dtype=torch.float32, device=completions.device)
            for token_start in range(0, length, token_chunk_size):
                token_stop = min(token_start + token_chunk_size, length)
                logits = _project_selected_hidden(
                    model,
                    output_embeddings,
                    predictor_hidden[:, token_start:token_stop],
                )
                token_log_p = F.log_softmax(logits, dim=-1).gather(
                    -1,
                    targets[:, token_start:token_stop].unsqueeze(-1),
                )
                log_p += token_log_p.squeeze(-1).sum(dim=1)
            result[start:stop] = log_p
        else:
            logits = model(input_ids=full_ids).logits[
                :, prompt_length - 1 : prompt_length + length - 1
            ]
            token_log_p = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1))
            result[start:stop] = token_log_p.squeeze(-1).sum(dim=1)
    return result


def _full_inputs(prompt_ids: torch.Tensor, completions: torch.Tensor) -> torch.Tensor:
    prompt = prompt_ids.unsqueeze(0).expand(completions.shape[0], -1)
    return torch.cat((prompt, completions), dim=1)


def generate_masked_proposal(
    model,
    prompt_ids: torch.Tensor,
    current: torch.Tensor,
    masks: MaskBatch,
    *,
    temperature: float,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample y from R_S(x, .) left-to-right and return log R_S(x, y)."""
    proposal = current.clone()
    log_probability = torch.zeros(current.shape[0], dtype=torch.float32, device=current.device)
    prompt_length = int(prompt_ids.numel())
    rows = torch.arange(current.shape[0], device=current.device)

    for mask_rank in range(masks.positions.shape[1]):
        position = masks.positions[:, mask_rank]
        valid = masks.valid[:, mask_rank]
        full_ids = _full_inputs(prompt_ids, proposal)
        predictor_positions = prompt_length + position - 1
        logits = selected_next_token_logits(
            model,
            full_ids,
            predictor_positions,
            batch_size=batch_size,
        )
        proposal_log_probs = F.log_softmax(logits / temperature, dim=-1)
        sampled = torch.multinomial(
            torch.exp(proposal_log_probs),
            num_samples=1,
            generator=generator,
        ).squeeze(1)
        sampled_log_p = proposal_log_probs.gather(1, sampled.unsqueeze(1)).squeeze(1)
        old_at_position = proposal[rows, position]
        replacement = torch.where(valid, sampled, old_at_position)
        proposal = proposal.scatter(1, position.unsqueeze(1), replacement.unsqueeze(1))
        log_probability += torch.where(valid, sampled_log_p, torch.zeros_like(sampled_log_p))
    return proposal, log_probability


def masked_transition_log_probabilities(
    model,
    prompt_ids: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    masks: MaskBatch,
    *,
    temperature: float,
    batch_size: int,
) -> torch.Tensor:
    """Evaluate log R_S(source, target) with the same left-to-right procedure."""
    work = source.clone()
    log_probability = torch.zeros(source.shape[0], dtype=torch.float32, device=source.device)
    prompt_length = int(prompt_ids.numel())
    rows = torch.arange(source.shape[0], device=source.device)

    for mask_rank in range(masks.positions.shape[1]):
        position = masks.positions[:, mask_rank]
        valid = masks.valid[:, mask_rank]
        full_ids = _full_inputs(prompt_ids, work)
        predictor_positions = prompt_length + position - 1
        logits = selected_next_token_logits(
            model,
            full_ids,
            predictor_positions,
            batch_size=batch_size,
        )
        proposal_log_probs = F.log_softmax(logits / temperature, dim=-1)
        target_token = target[rows, position]
        token_log_p = proposal_log_probs.gather(1, target_token.unsqueeze(1)).squeeze(1)
        old_at_position = work[rows, position]
        replacement = torch.where(valid, target_token, old_at_position)
        work = work.scatter(1, position.unsqueeze(1), replacement.unsqueeze(1))
        log_probability += torch.where(valid, token_log_p, torch.zeros_like(token_log_p))
    return log_probability


def symmetrization_log_acceptance(
    log_r_forward: torch.Tensor,
    log_r_reverse: torch.Tensor,
) -> torch.Tensor:
    return torch.minimum(torch.zeros_like(log_r_forward), log_r_reverse - log_r_forward)


def barker_log_acceptance(log_target_ratio: torch.Tensor) -> torch.Tensor:
    return F.logsigmoid(log_target_ratio)


def score_observable(
    tokenizer,
    observable,
    *,
    prompt: str,
    prompt_ids: torch.Tensor,
    completions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[Any]]:
    """Evaluate the exact text observable in one batched device-to-CPU boundary."""
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
    if not bool(torch.isfinite(phi).all().item()) or not bool(torch.isfinite(raw_ari).all().item()):
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
    pad_token_id: int | None,
) -> torch.Tensor:
    """Draw X_0 from p_M in configurable GPU batches."""
    generated_chunks: list[torch.Tensor] = []
    for start in range(0, num_particles, batch_size):
        chunk_size = min(batch_size, num_particles - start)
        input_ids = prompt_ids.unsqueeze(0).expand(chunk_size, -1)
        if hasattr(model, "generate"):
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=None,
                temperature=1.0,
                eos_token_id=None,
                pad_token_id=pad_token_id,
            )
            expected_length = int(prompt_ids.numel()) + max_new_tokens
            if generated.shape[1] != expected_length:
                raise RuntimeError(
                    "Base generation did not return exactly max_new_tokens tokens; "
                    "the masked SMC state requires a fixed completion length"
                )
            generated_chunks.append(generated[:, -max_new_tokens:])
            continue

        work = input_ids.clone()
        sampled_tokens: list[torch.Tensor] = []
        for _ in range(max_new_tokens):
            positions = torch.full(
                (chunk_size,),
                work.shape[1] - 1,
                dtype=torch.long,
                device=work.device,
            )
            logits = selected_next_token_logits(model, work, positions, batch_size=batch_size)
            next_token = torch.multinomial(
                torch.softmax(logits, dim=-1),
                num_samples=1,
                generator=generator,
            )
            sampled_tokens.append(next_token)
            work = torch.cat((work, next_token), dim=1)
        generated_chunks.append(torch.cat(sampled_tokens, dim=1))
    return torch.cat(generated_chunks, dim=0)


def systematic_resample(
    normalized_log_weights: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    n_particles = normalized_log_weights.numel()
    weights = torch.exp(normalized_log_weights)
    cumulative = torch.cumsum(weights, dim=0)
    cumulative[-1] = 1.0
    start = torch.rand((), device=weights.device, generator=generator) / n_particles
    positions = start + torch.arange(n_particles, device=weights.device) / n_particles
    return torch.searchsorted(cumulative, positions, right=False)


def normalize_log_weights(log_weights: torch.Tensor) -> torch.Tensor:
    return log_weights - torch.logsumexp(log_weights, dim=0)


def ess_from_normalized_log_weights(normalized_log_weights: torch.Tensor) -> torch.Tensor:
    return torch.exp(-torch.logsumexp(2.0 * normalized_log_weights, dim=0))


def apply_masked_barker_mutation(
    particles: ParticleBatch,
    model,
    tokenizer,
    observable,
    *,
    prompt: str,
    prompt_ids: torch.Tensor,
    target_lambda: float,
    transition_index: int,
    config: MaskedBarkerSMCConfig,
    generator: torch.Generator,
) -> tuple[ParticleBatch, MutationDiagnostics]:
    """Apply M_{k-1}, a masked symmetric-proposal Barker kernel for pi_{k-1}."""
    accepted_total = torch.zeros((), dtype=torch.long, device=prompt_ids.device)
    sym_probability_sum = torch.zeros((), dtype=torch.float64, device=prompt_ids.device)
    barker_probability_sum = torch.zeros((), dtype=torch.float64, device=prompt_ids.device)
    proposals = config.num_particles * config.mcmc_steps_per_level

    for step_index in range(config.mcmc_steps_per_level):
        masks = sample_masks(
            config,
            transition_index=transition_index,
            step_index=step_index,
            device=prompt_ids.device,
            generator=generator,
        )
        proposal_tokens, log_r_forward = generate_masked_proposal(
            model,
            prompt_ids,
            particles.completion_token_ids,
            masks,
            temperature=config.proposal_temperature,
            batch_size=config.batch_size,
            generator=generator,
        )
        log_r_reverse = masked_transition_log_probabilities(
            model,
            prompt_ids,
            proposal_tokens,
            particles.completion_token_ids,
            masks,
            temperature=config.proposal_temperature,
            batch_size=config.batch_size,
        )
        proposal_phi, proposal_raw_ari, _ = score_observable(
            tokenizer,
            observable,
            prompt=prompt,
            prompt_ids=prompt_ids,
            completions=proposal_tokens,
        )
        proposal_log_p = completion_log_probabilities(
            model,
            prompt_ids,
            proposal_tokens,
            batch_size=config.batch_size,
            token_chunk_size=config.score_token_chunk_size,
        )

        log_a_sym = symmetrization_log_acceptance(log_r_forward, log_r_reverse)
        log_target_ratio = (
            proposal_log_p
            - particles.log_p_model
            - target_lambda * (proposal_phi - particles.phi)
        )
        log_a_barker = barker_log_acceptance(log_target_ratio)
        log_a_total = log_a_sym + log_a_barker
        log_uniform = torch.log(
            torch.rand(
                config.num_particles,
                dtype=log_a_total.dtype,
                device=prompt_ids.device,
                generator=generator,
            )
        )
        accepted = log_uniform < log_a_total

        particles = ParticleBatch(
            completion_token_ids=torch.where(
                accepted.unsqueeze(1),
                proposal_tokens,
                particles.completion_token_ids,
            ),
            phi=torch.where(accepted, proposal_phi, particles.phi),
            raw_ari=torch.where(accepted, proposal_raw_ari, particles.raw_ari),
            log_p_model=torch.where(accepted, proposal_log_p, particles.log_p_model),
            root_ancestors=particles.root_ancestors,
        )
        accepted_total += accepted.sum()
        sym_probability_sum += torch.exp(log_a_sym).to(torch.float64).sum()
        barker_probability_sum += torch.exp(log_a_barker).to(torch.float64).sum()

    denominator = max(1, proposals)
    diagnostics = MutationDiagnostics(
        accepted=int(accepted_total.item()),
        proposals=proposals,
        mean_symmetrization_probability=float((sym_probability_sum / denominator).item()),
        mean_barker_probability=float((barker_probability_sum / denominator).item()),
    )
    return particles, diagnostics


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
) -> list[dict[str, Any]]:
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
    log_p_model = particles.log_p_model.detach().cpu().tolist()
    roots = particles.root_ancestors.detach().cpu().tolist()
    rows: list[dict[str, Any]] = []
    for index, (ids, completion, score) in enumerate(zip(completion_ids, completions, scores)):
        metadata = dict(score.metadata)
        rows.append(
            {
                "particle_id": index,
                "text": prompt + completion,
                "completion": completion,
                "phi": float(phi[index]),
                "raw_ari": metadata.get("raw_ari"),
                "capped_ari": metadata.get("capped_ari"),
                "log_p_model": float(log_p_model[index]),
                "log_weight": float(log_weights[index]),
                "normalized_weight": float(weights[index]),
                "root_ancestor": int(roots[index]),
                "completion_token_ids": ids,
                "metadata": metadata,
            }
        )
    return rows


def run_masked_barker_smc(
    model,
    tokenizer,
    observable,
    *,
    prompt: str,
    config: MaskedBarkerSMCConfig,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> MaskedBarkerSMCResult:
    set_seed(config.seed)
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(config.seed)
    prompt_ids = _prompt_token_ids(tokenizer, prompt, device)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    uniform_log_weight = -math.log(config.num_particles)

    _synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        completion_tokens = generate_base_completions(
            model,
            prompt_ids,
            num_particles=config.num_particles,
            max_new_tokens=config.max_new_tokens,
            batch_size=config.batch_size,
            generator=generator,
            pad_token_id=pad_token_id,
        )
        phi, raw_ari, _ = score_observable(
            tokenizer,
            observable,
            prompt=prompt,
            prompt_ids=prompt_ids,
            completions=completion_tokens,
        )
        log_p_model = completion_log_probabilities(
            model,
            prompt_ids,
            completion_tokens,
            batch_size=config.batch_size,
            token_chunk_size=config.score_token_chunk_size,
        )
        particles = ParticleBatch(
            completion_token_ids=completion_tokens,
            phi=phi,
            raw_ari=raw_ari,
            log_p_model=log_p_model,
            root_ancestors=torch.arange(config.num_particles, device=device),
        )
        log_weights = torch.full(
            (config.num_particles,),
            uniform_log_weight,
            dtype=torch.float64,
            device=device,
        )
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
                "mean_symmetrization_probability": None,
                "mean_barker_probability": None,
                "weighted_mean_ari_before_resampling": float(raw_ari.mean().item()),
                "unweighted_mean_ari_after_resampling": float(raw_ari.mean().item()),
            }
        ]
        total_accepted = 0
        total_proposals = 0
        resampling_steps = 0
        min_ess_over_n = 1.0

        for level in range(1, len(config.lambdas)):
            previous_lambda = float(config.lambdas[level - 1])
            current_lambda = float(config.lambdas[level])
            particles, mutation = apply_masked_barker_mutation(
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
            )
            total_accepted += mutation.accepted
            total_proposals += mutation.proposals

            log_weights = log_weights - (current_lambda - previous_lambda) * particles.phi.to(
                torch.float64
            )
            log_weights = normalize_log_weights(log_weights)
            ess = ess_from_normalized_log_weights(log_weights)
            ess_over_n_before = float((ess / config.num_particles).item())
            normalized_weights_before = torch.exp(log_weights)
            weighted_mean_ari_before = float(
                torch.sum(
                    normalized_weights_before * particles.raw_ari.to(torch.float64)
                ).item()
            )
            min_ess_over_n = min(min_ess_over_n, ess_over_n_before)
            resampled = bool(
                config.resampling
                and ess_over_n_before < config.ess_threshold
            )
            if resampled:
                indices = systematic_resample(log_weights, generator=generator)
                particles = ParticleBatch(
                    completion_token_ids=particles.completion_token_ids.index_select(0, indices),
                    phi=particles.phi.index_select(0, indices),
                    raw_ari=particles.raw_ari.index_select(0, indices),
                    log_p_model=particles.log_p_model.index_select(0, indices),
                    root_ancestors=particles.root_ancestors.index_select(0, indices),
                )
                log_weights = torch.full_like(log_weights, uniform_log_weight)
                resampling_steps += 1
            ess_over_n_after = 1.0 if resampled else ess_over_n_before
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
                "mean_symmetrization_probability": mutation.mean_symmetrization_probability,
                "mean_barker_probability": mutation.mean_barker_probability,
                "weighted_mean_ari_before_resampling": weighted_mean_ari_before,
                "unweighted_mean_ari_after_resampling": float(particles.raw_ari.mean().item()),
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
        )

    _synchronize(device)
    elapsed = time.perf_counter() - start
    final_ess = ess_from_normalized_log_weights(log_weights)
    final_ess_over_n = float((final_ess / config.num_particles).item())
    raw_ari = np.asarray([float(row["raw_ari"]) for row in samples], dtype=np.float64)
    normalized_weights = np.asarray(
        [float(row["normalized_weight"]) for row in samples],
        dtype=np.float64,
    )
    tail_counts = {
        f"ari_ge_{threshold:g}": int(np.sum(raw_ari >= threshold))
        for threshold in config.tail_thresholds
    }
    tail_proportions = {
        key: float(count / config.num_particles) for key, count in tail_counts.items()
    }
    weighted_tail_probabilities = {
        f"ari_ge_{threshold:g}": float(
            np.sum(normalized_weights[raw_ari >= threshold])
        )
        for threshold in config.tail_thresholds
    }
    acceptance_rate = total_accepted / total_proposals if total_proposals else 0.0
    summary = {
        "method": "masked_barker_smc",
        "algorithm": "masked_barker_smc",
        "num_particles": config.num_particles,
        "max_new_tokens": config.max_new_tokens,
        "lambdas": [float(value) for value in config.lambdas],
        "target_lambda": float(config.lambdas[-1]),
        "mcmc_steps_per_level": config.mcmc_steps_per_level,
        "ess_threshold": config.ess_threshold,
        "resampling": config.resampling,
        "resampling_method": config.resampling_method,
        "resampling_steps": resampling_steps,
        "runtime_total_seconds": float(elapsed),
        "runtime_seconds_per_final_particle": float(elapsed / config.num_particles),
        "final_ess_over_n": final_ess_over_n,
        "min_ess_over_n": float(min_ess_over_n),
        "mean_mcmc_acceptance_rate": float(acceptance_rate),
        "unweighted_mean_ari": float(np.mean(raw_ari)),
        "weighted_mean_ari": float(np.sum(normalized_weights * raw_ari)),
        "min_ari": float(np.min(raw_ari)),
        "max_ari": float(np.max(raw_ari)),
        "tail_counts": tail_counts,
        "tail_proportions": tail_proportions,
        "weighted_tail_probabilities": weighted_tail_probabilities,
        "final_distinct_root_ancestors": len(set(int(row["root_ancestor"]) for row in samples)),
    }
    diagnostics = {
        "level_diagnostics": level_diagnostics,
        "total_mcmc_accepted": total_accepted,
        "total_mcmc_proposals": total_proposals,
        "resampling_steps": resampling_steps,
        "final_distinct_root_ancestors": summary["final_distinct_root_ancestors"],
        "observable_device_boundary": (
            "Exact textstat ARI is decoded and evaluated on CPU once per batched proposal step; "
            "all LLM and SMC tensor operations remain on the model device."
        ),
    }
    return MaskedBarkerSMCResult(
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
        "# Masked Barker Full-Sequence SMC",
        "",
        "The particle state is a fixed-length complete LM completion. Mutation uses "
        "state-independent token masks, an exactly symmetrized masked-LM proposal, "
        "and Barker balancing. No target normalizing constant is evaluated.",
        "",
        "## Summary",
        "",
        f"- Runtime total seconds: {summary['runtime_total_seconds']:.6g}",
        f"- Seconds per final particle: {summary['runtime_seconds_per_final_particle']:.6g}",
        f"- Final ESS/N: {summary['final_ess_over_n']:.6g}",
        f"- Minimum pre-resampling ESS/N: {summary['min_ess_over_n']:.6g}",
        f"- Total MCMC acceptance rate: {summary['mean_mcmc_acceptance_rate']:.6g}",
        f"- Resampling steps: {summary['resampling_steps']}",
        f"- Unweighted mean raw ARI: {summary['unweighted_mean_ari']:.6g}",
        f"- Weighted mean raw ARI under final target: {summary['weighted_mean_ari']:.6g}",
        f"- Observed min/max raw ARI: {summary['min_ari']:.6g} / {summary['max_ari']:.6g}",
        "",
        "## ARI tails",
        "",
        "| threshold | raw count | raw proportion | weighted probability |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for key, count in summary["tail_counts"].items():
        lines.append(
            f"| {key.replace('ari_ge_', 'ARI >= ')} | {count} | "
            f"{summary['tail_proportions'][key]:.6g} | "
            f"{summary['weighted_tail_probabilities'][key]:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Schedule",
            "",
            "| level | lambda | mutation target | ESS/N pre | ESS/N post | resampled | acceptance | weighted mean ARI pre | unweighted mean ARI post |",
            "| ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: |",
        ]
    )
    for row in diagnostics["level_diagnostics"]:
        target = row["mutation_target_lambda"]
        acceptance = row["acceptance_rate"]
        lines.append(
            "| {level} | {lam:.6g} | {target} | {pre:.6g} | {post:.6g} | {resampled} | "
            "{acceptance} | {weighted_mean:.6g} | {unweighted_mean:.6g} |".format(
                level=row["level"],
                lam=row["lambda"],
                target="" if target is None else f"{target:.6g}",
                pre=row["ess_over_n_before_resampling"],
                post=row["ess_over_n_after_resampling"],
                resampled="yes" if row["resampled"] else "no",
                acceptance="" if acceptance is None else f"{acceptance:.6g}",
                weighted_mean=row["weighted_mean_ari_before_resampling"],
                unweighted_mean=row["unweighted_mean_ari_after_resampling"],
            )
        )
    return "\n".join(lines) + "\n"
