from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import torch

from rea_llms.benchmarking import temporal_ess_initial_positive
from rea_llms.models import set_seed
from suffix_balanced_smc_experiment.smc_sampler import (
    generate_base_completions,
    metropolis_log_acceptance_fast,
    propose_suffixes_parallel,
    score_observable,
)


OFFICIAL_REPOSITORY_URL = "https://github.com/MLiS-Research/rea_of_llms_supplementary_code"
OFFICIAL_NOTEBOOK_PATH = "sampling_examples.ipynb"
OFFICIAL_FUNCTIONS = (
    "generate_tokens",
    "generate_proposal",
    "compute_tps_acceptance_factor",
    "tps_step",
    "biased_sampling",
)


@dataclass(frozen=True)
class OfficialAnnealedTPSConfig:
    max_new_tokens: int = 100
    lambdas: tuple[float, ...] = (0.0, -0.25, -0.5, -1.0)
    steps_per_bias: int = 1000
    proposal_temperature: float = 1.0
    batch_size: int = 1
    seed: int = 12345
    ari_mode: str = "raw"
    ari_cap: float = 15.0
    tail_thresholds: tuple[float, ...] = (10.0, 12.0, 15.0, 18.0)
    top_completions: int = 20

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if not self.lambdas:
            raise ValueError("lambdas must be non-empty")
        if any(not math.isfinite(float(value)) for value in self.lambdas):
            raise ValueError("all lambdas must be finite")
        if any(
            float(current) > float(previous) + 1e-12
            for previous, current in zip(self.lambdas, self.lambdas[1:])
        ):
            raise ValueError("lambdas must be non-increasing for the high-ARI convention")
        if self.steps_per_bias <= 0:
            raise ValueError("steps_per_bias must be > 0")
        if not math.isclose(self.proposal_temperature, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("official annealed TPS requires proposal_temperature=1.0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.ari_mode not in {"raw", "capped"}:
            raise ValueError("ari_mode must be one of {'raw', 'capped'}")
        if self.top_completions <= 0:
            raise ValueError("top_completions must be > 0")


@dataclass
class OfficialAnnealedTPSResult:
    summary: dict[str, Any]
    trajectory: dict[str, np.ndarray]
    config: dict[str, Any]


def official_choice_indices(prompt_token_count: int, max_new_tokens: int) -> tuple[int, ...]:
    """Exact `choice_idxs` convention from the official sampling notebook."""
    return tuple(range(prompt_token_count, prompt_token_count + max_new_tokens))


def sample_official_cut_index(
    rng: random.Random,
    prompt_token_count: int,
    max_new_tokens: int,
) -> int:
    return int(rng.choice(official_choice_indices(prompt_token_count, max_new_tokens)))


def suffix_size_from_official_cut(
    cut_index: int,
    prompt_token_count: int,
    max_new_tokens: int,
) -> int:
    completion_prefix_size = cut_index - prompt_token_count
    if not 0 <= completion_prefix_size < max_new_tokens:
        raise ValueError("official TPS cut must retain between 0 and T-1 completion tokens")
    return max_new_tokens - completion_prefix_size


def official_tps_log_acceptance(
    current_phi: torch.Tensor,
    proposal_phi: torch.Tensor,
    *,
    target_lambda: float,
) -> torch.Tensor:
    return metropolis_log_acceptance_fast(
        current_phi,
        proposal_phi,
        target_lambda=target_lambda,
    )


def generate_official_tps_proposal(
    model,
    prompt_ids: torch.Tensor,
    current_tokens: torch.Tensor,
    *,
    cut_index: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, int]:
    """Regenerate exactly the suffix selected by the official full-token cut index."""
    prompt_token_count = int(prompt_ids.numel())
    suffix_size = suffix_size_from_official_cut(
        cut_index,
        prompt_token_count,
        int(current_tokens.shape[1]),
    )
    suffix_sizes = torch.tensor(
        [suffix_size], dtype=torch.long, device=current_tokens.device
    )
    proposal_tokens, log_forward, log_reverse = propose_suffixes_parallel(
        model,
        prompt_ids,
        current_tokens,
        suffix_sizes,
        batch_size=batch_size,
        token_chunk_size=1,
        generator=generator,
        score_probabilities=False,
    )
    assert log_forward is None and log_reverse is None
    return proposal_tokens, suffix_size


def _prompt_token_ids(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt")["input_ids"]
    if not isinstance(encoded, torch.Tensor):
        encoded = torch.tensor(encoded, dtype=torch.long)
    return encoded[0].to(device=device, dtype=torch.long)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_official_annealed_tps(
    model,
    tokenizer,
    observable,
    *,
    prompt: str,
    config: OfficialAnnealedTPSConfig,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> OfficialAnnealedTPSResult:
    """Run one complete annealed TPS chain following the official Algorithm 2 code."""
    if getattr(observable, "mode", config.ari_mode) != config.ari_mode:
        raise ValueError("observable mode does not match the TPS ari_mode configuration")
    set_seed(config.seed)
    device = next(model.parameters()).device
    torch_generator = torch.Generator(device=device).manual_seed(config.seed)
    python_rng = random.Random(config.seed)
    prompt_ids = _prompt_token_ids(tokenizer, prompt, device)
    prompt_token_count = int(prompt_ids.numel())
    total_steps = len(config.lambdas) * config.steps_per_bias
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    trajectory = {
        "lambda_index": np.empty(total_steps, dtype=np.int32),
        "lambda": np.empty(total_steps, dtype=np.float64),
        "step": np.empty(total_steps, dtype=np.int32),
        "phi": np.empty(total_steps, dtype=np.float64),
        "raw_ari": np.empty(total_steps, dtype=np.float64),
        "accepted": np.empty(total_steps, dtype=np.uint8),
        "suffix_size": np.empty(total_steps, dtype=np.int32),
        "cumulative_runtime_seconds": np.empty(total_steps, dtype=np.float64),
        "cumulative_proposals": np.empty(total_steps, dtype=np.int64),
        "cumulative_regenerated_tokens": np.empty(total_steps, dtype=np.int64),
    }
    first_tail_discovery: dict[float, dict[str, Any] | None] = {
        float(threshold): None for threshold in config.tail_thresholds
    }
    discovered_by_threshold = {
        float(threshold): set() for threshold in config.tail_thresholds
    }
    best_discoveries: dict[str, dict[str, Any]] = {}

    _synchronize(device)
    start = time.perf_counter()

    def track(completion_tokens: torch.Tensor, raw_ari: float, lambda_value: float) -> None:
        completion = tokenizer.decode(
            completion_tokens[0].detach().cpu().tolist(),
            skip_special_tokens=True,
        )
        elapsed = time.perf_counter() - start
        existing = best_discoveries.get(completion)
        if existing is None or raw_ari > float(existing["raw_ari"]):
            best_discoveries[completion] = {
                "raw_ari": float(raw_ari),
                "lambda": float(lambda_value),
                "time_first_seen_seconds": float(elapsed),
                "completion": completion,
            }
        for threshold in config.tail_thresholds:
            threshold_value = float(threshold)
            if raw_ari < threshold_value:
                continue
            discovered_by_threshold[threshold_value].add(completion)
            if first_tail_discovery[threshold_value] is None:
                first_tail_discovery[threshold_value] = {
                    "raw_ari": float(raw_ari),
                    "lambda": float(lambda_value),
                    "time_first_seen_seconds": float(elapsed),
                    "completion": completion,
                }

    with torch.inference_mode():
        current_tokens = generate_base_completions(
            model,
            prompt_ids,
            num_particles=1,
            max_new_tokens=config.max_new_tokens,
            batch_size=1,
            generator=torch_generator,
        )
        current_phi, current_raw_ari, _ = score_observable(
            tokenizer,
            observable,
            prompt=prompt,
            prompt_ids=prompt_ids,
            completions=current_tokens,
        )
        track(current_tokens, float(current_raw_ari[0].item()), float(config.lambdas[0]))
        accepted_total = 0
        regenerated_tokens = 0
        flat_index = 0

        for lambda_index, lambda_value in enumerate(config.lambdas):
            level_accepted = 0
            for step in range(config.steps_per_bias):
                cut_index = sample_official_cut_index(
                    python_rng,
                    prompt_token_count,
                    config.max_new_tokens,
                )
                proposal_tokens, suffix_size = generate_official_tps_proposal(
                    model,
                    prompt_ids,
                    current_tokens,
                    cut_index=cut_index,
                    batch_size=config.batch_size,
                    generator=torch_generator,
                )
                proposal_phi, proposal_raw_ari, _ = score_observable(
                    tokenizer,
                    observable,
                    prompt=prompt,
                    prompt_ids=prompt_ids,
                    completions=proposal_tokens,
                )
                track(proposal_tokens, float(proposal_raw_ari[0].item()), float(lambda_value))
                log_acceptance = official_tps_log_acceptance(
                    current_phi,
                    proposal_phi,
                    target_lambda=float(lambda_value),
                )
                accepted = bool(math.log(max(python_rng.random(), 1e-300)) < log_acceptance.item())
                if accepted:
                    current_tokens = proposal_tokens
                    current_phi = proposal_phi
                    current_raw_ari = proposal_raw_ari
                    accepted_total += 1
                    level_accepted += 1
                regenerated_tokens += suffix_size

                trajectory["lambda_index"][flat_index] = lambda_index
                trajectory["lambda"][flat_index] = float(lambda_value)
                trajectory["step"][flat_index] = step
                trajectory["phi"][flat_index] = float(current_phi[0].item())
                trajectory["raw_ari"][flat_index] = float(current_raw_ari[0].item())
                trajectory["accepted"][flat_index] = int(accepted)
                trajectory["suffix_size"][flat_index] = suffix_size
                trajectory["cumulative_runtime_seconds"][flat_index] = (
                    time.perf_counter() - start
                )
                trajectory["cumulative_proposals"][flat_index] = flat_index + 1
                trajectory["cumulative_regenerated_tokens"][flat_index] = regenerated_tokens
                flat_index += 1

            if progress_callback is not None:
                progress_callback(
                    {
                        "lambda_index": lambda_index,
                        "lambda": float(lambda_value),
                        "steps_completed": config.steps_per_bias,
                        "acceptance_rate": level_accepted / config.steps_per_bias,
                    }
                )

        final_completion = tokenizer.decode(
            current_tokens[0].detach().cpu().tolist(),
            skip_special_tokens=True,
        )

    _synchronize(device)
    elapsed = time.perf_counter() - start
    final_slice = slice(total_steps - config.steps_per_bias, total_steps)
    final_raw_ari = trajectory["raw_ari"][final_slice]
    temporal_ess = temporal_ess_initial_positive(final_raw_ari)
    tail_probabilities = {
        f"ari_ge_{threshold:g}": float(np.mean(final_raw_ari >= threshold))
        for threshold in config.tail_thresholds
    }
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    summary = {
        "algorithm": "official_annealed_tps",
        "method": "official_annealed_tps",
        "seed": config.seed,
        "ari_mode": config.ari_mode,
        "ari_cap": config.ari_cap if config.ari_mode == "capped" else None,
        "lambdas": [float(value) for value in config.lambdas],
        "target_lambda": float(config.lambdas[-1]),
        "steps_per_bias": config.steps_per_bias,
        "completed_full_schedule": flat_index == total_steps,
        "trajectory_values": flat_index,
        "runtime_total_seconds": float(elapsed),
        "total_proposals": total_steps,
        "total_regenerated_tokens": int(regenerated_tokens),
        "acceptance_rate": float(accepted_total / total_steps),
        "proposals_per_second": float(total_steps / elapsed),
        "regenerated_tokens_per_second": float(regenerated_tokens / elapsed),
        "cuda_peak_memory_bytes": peak_memory,
        "mean_raw_ari_final_target": float(np.mean(final_raw_ari)),
        "max_raw_ari_final_target": float(np.max(final_raw_ari)),
        "max_raw_ari_discovered": float(
            max(float(row["raw_ari"]) for row in best_discoveries.values())
        ),
        "tail_probabilities_final_target": tail_probabilities,
        "tps_temporal_ess": float(temporal_ess),
        "final_completion": final_completion,
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
        )[: config.top_completions],
        "official_source": {
            "repository_url": OFFICIAL_REPOSITORY_URL,
            "notebook": OFFICIAL_NOTEBOOK_PATH,
            "functions_replicated": list(OFFICIAL_FUNCTIONS),
        },
        "implementation_differences": [
            "The official functions live in a notebook and are not importable as a module; "
            "this adapter ports only their cut, suffix proposal, acceptance, and annealing semantics.",
            "Suffix generation uses the repository's equivalent cached GPU generator instead of "
            "calling Hugging Face model.generate for each scalar proposal.",
            "Both benchmark methods use the shared exact token-sequence ARI convention so they "
            "target the same phi. The official notebook instead decodes the full sequence and "
            "removes the first len(prompt) characters, which can differ at BPE boundaries.",
            "Trajectory values are recorded after each accept/reject, whereas the official "
            "notebook's step_info current_obs field is captured before the transition.",
        ],
        "resampling": False,
        "importance_weights": False,
        "num_chains": 1,
    }
    return OfficialAnnealedTPSResult(
        summary=summary,
        trajectory=trajectory,
        config=asdict(config),
    )


def render_tps_report(summary: dict[str, Any]) -> str:
    source = summary["official_source"]
    lines = [
        "# Official Annealed TPS",
        "",
        "This is one annealed Markov chain following the cut, suffix-regeneration, "
        "direct-MH acceptance, and warm-start conventions of the official sampling notebook.",
        "",
        f"- Official repository: `{source['repository_url']}`",
        f"- Official source commit: `{summary.get('official_commit', 'unknown')}`",
        f"- Official notebook: `{source['notebook']}`",
        f"- Official functions replicated: `{', '.join(source['functions_replicated'])}`",
        f"- ARI mode: `{summary['ari_mode']}`",
        f"- Completed full schedule: `{str(summary['completed_full_schedule']).lower()}`",
        f"- Steps per bias: {summary['steps_per_bias']}",
        f"- Runtime: {summary['runtime_total_seconds']:.3f} s",
        f"- Total proposals: {summary['total_proposals']}",
        f"- Total regenerated tokens: {summary['total_regenerated_tokens']}",
        f"- Acceptance rate: {summary['acceptance_rate']:.4f}",
        f"- Mean raw ARI at final target: {summary['mean_raw_ari_final_target']:.4f}",
        f"- Temporal ESS at final target: {summary['tps_temporal_ess']:.2f}",
        "- Importance weights: `false`",
        "- Resampling: `false`",
        "",
        "## Theory",
        "",
        "At each lambda, direct Metropolis-Hastings leaves the tilted target invariant. "
        "Convergence still requires irreducibility, aperiodicity, and sufficiently many "
        "steps. Consecutive trajectory values are correlated.",
        "",
        "## Instrumentation differences",
        "",
    ]
    lines.extend(f"- {difference}" for difference in summary["implementation_differences"])
    return "\n".join(lines) + "\n"
