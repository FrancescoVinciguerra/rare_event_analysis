"""Run unconditional native-chain SMC and component-wise reverse-SNIS."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

from .ancestral import NativeMDLMSampler
from .configuration import (
    decoder_id,
    evaluator_id,
    fingerprint,
    load_yaml,
    validate_benchmark_config,
)
from .evaluation import (
    ToxicityEvaluator,
    identical_classifier_inputs,
    unconditional_mdlm_prefix_classifier_text,
    unconditional_trimmed_classifier_text,
)
from .runtime import load_evaluator_runtime, load_generator_runtime
from .smc import (
    SMCComponentConfig,
    aggregate_component_estimates,
    ess_from_log_weights,
    run_smc_component,
)
from .storage import atomic_write_json, initialize_manifest


class UnconditionalToxicityFunctions:
    def __init__(
        self,
        *,
        generator_tokenizer: Any,
        evaluator: ToxicityEvaluator,
        reward_completion_tokens: int,
        event_completion_tokens: int,
        reward_max_length: int,
        event_max_length: int,
        event_text_scope: str,
        completion_trimming_tokenizer: Any,
    ) -> None:
        self.generator_tokenizer = generator_tokenizer
        self.evaluator = evaluator
        self.reward_completion_tokens = reward_completion_tokens
        self.event_completion_tokens = event_completion_tokens
        self.reward_max_length = reward_max_length
        self.event_max_length = event_max_length
        self.event_text_scope = event_text_scope
        self.completion_trimming_tokenizer = completion_trimming_tokenizer
        self.last_reward_rows: list[dict[str, Any]] = []
        self.last_reward_texts: list[str] = []
        self.last_event_rows: list[dict[str, Any]] = []
        self.last_event_texts: list[str] = []
        self.last_event_token_ids: list[list[int]] = []

    def reward(self, tokens: torch.Tensor) -> torch.Tensor:
        truncated = tokens[:, : self.reward_completion_tokens]
        self.last_reward_texts = self.generator_tokenizer.batch_decode(
            truncated.detach().cpu().tolist(), skip_special_tokens=True
        )
        self.last_reward_rows = self.evaluator.score_texts(
            self.last_reward_texts, max_length=self.reward_max_length
        )
        return torch.tensor(
            [row["reward"] for row in self.last_reward_rows],
            dtype=torch.float64,
            device=tokens.device,
        )

    def event(self, tokens: torch.Tensor) -> torch.Tensor:
        self.last_event_texts = []
        self.last_event_token_ids = []
        terminal_token_ids = tokens.detach().cpu().tolist()
        if self.event_text_scope == "first_generated_mdlm_tokens_only":
            for token_ids in terminal_token_ids:
                text, prefix_ids = unconditional_mdlm_prefix_classifier_text(
                    token_ids=token_ids,
                    decoder_tokenizer=self.generator_tokenizer,
                    completion_tokens=self.event_completion_tokens,
                )
                self.last_event_texts.append(text)
                self.last_event_token_ids.append(prefix_ids)
        elif (
            self.event_text_scope
            == "generated_text_retokenized_and_trimmed_with_roberta_large"
        ):
            decoded = self.generator_tokenizer.batch_decode(
                terminal_token_ids, skip_special_tokens=True
            )
            for text in decoded:
                trimmed_text, token_ids = unconditional_trimmed_classifier_text(
                    decoded_text=text,
                    trimming_tokenizer=self.completion_trimming_tokenizer,
                    completion_tokens=self.event_completion_tokens,
                )
                self.last_event_texts.append(trimmed_text)
                self.last_event_token_ids.append(token_ids)
        else:
            raise ValueError(f"unsupported event text scope: {self.event_text_scope}")
        if identical_classifier_inputs(
            self.last_reward_texts,
            first_max_length=self.reward_max_length,
            second_texts=self.last_event_texts,
            second_max_length=self.event_max_length,
        ):
            self.last_event_rows = self.last_reward_rows
        else:
            self.last_event_rows = self.evaluator.score_texts(
                self.last_event_texts, max_length=self.event_max_length
            )
        return torch.tensor(
            [row["is_event"] for row in self.last_event_rows],
            dtype=torch.bool,
            device=tokens.device,
        )


def _mean_se(values: list[float]) -> tuple[float, float | None]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None
    return mean, statistics.stdev(values) / math.sqrt(len(values))


def _smc_run_count(smc: dict[str, Any]) -> int:
    return int(smc.get("num_independent_runs", smc.get("independent_runs", 1)))


def _reward_interval(smc: dict[str, Any]) -> int:
    return int(smc.get("reward_interval", smc.get("reward_frequency", 20)))


def _component_seed(
    smc: dict[str, Any], *, run_index: int, component_index: int
) -> int:
    """Assign a disjoint deterministic RNG seed to every SMC population."""

    return int(smc["base_seed"]) + 1_000_000 * run_index + component_index


def _reward_quantiles(rewards: torch.Tensor) -> dict[str, float]:
    probabilities = torch.tensor(
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        dtype=torch.float64,
        device=rewards.device,
    )
    values = torch.quantile(rewards.double(), probabilities).cpu().tolist()
    labels = ("q00", "q10", "q25", "q50", "q75", "q90", "q100")
    return {label: float(value) for label, value in zip(labels, values)}


def _aggregate(
    tilted_component_summaries: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    smc_cfg = config["smc"]
    lambdas = [float(value) for value in smc_cfg["tilted_lambdas"]]
    weights_a = [float(value) for value in smc_cfg["tilted_component_weights_a"]]
    runs = _smc_run_count(smc_cfg)
    by_key = {
        (int(row["run_index"]), float(row["lambda"])): row
        for row in tilted_component_summaries
    }
    run_estimates = []
    component_estimates_by_run = []
    for run_index in range(runs):
        estimates = [
            float(by_key[(run_index, value)]["reverse_snis_estimate"])
            for value in lambdas
        ]
        component_estimates_by_run.append(estimates)
        run_estimates.append(aggregate_component_estimates(estimates, weights_a))
    mean, standard_error = _mean_se(run_estimates)
    return {
        "conditioning_mode": "unconditional",
        "definition": "p_hat = sum_j a_j * p_hat_j",
        "run_estimates": run_estimates,
        "component_estimates_by_run": component_estimates_by_run,
        "mean_reverse_snis_estimate": mean,
        "empirical_standard_error_across_runs": standard_error,
        "tilted_component_aggregation": {
            "symbol": "a_j",
            "definition": "sum_j a_j * p_hat_j; each reverse-SNIS ratio normalized separately",
            "tilted_lambdas": lambdas,
            "weights_a": weights_a,
            "weight_policy": smc_cfg["component_weight_policy"],
            "optimize_component_weights": bool(
                smc_cfg["optimize_component_weights"]
            ),
        },
        "finite_sample_note": (
            "Reverse self-normalized importance sampling is consistent under "
            "standard conditions and generally biased at finite particle count."
        ),
    }


def smc_manifest(config: dict[str, Any]) -> dict[str, Any]:
    validate_benchmark_config(config)
    smc = copy.deepcopy(config["smc"])
    tilted_sampler = smc.setdefault("tilted_sampler", {"type": "fk_bootstrap"})
    if tilted_sampler.get("type", "fk_bootstrap") == "fk_bootstrap":
        tilted_sampler.setdefault(
            "implementation_version", "fk_mdlm_multinomial_x0_k4_v1"
        )
        tilted_sampler.setdefault("intermediate_x0_sampling", "mdlm_multinomial")
        tilted_sampler.setdefault("num_x0_samples", 4)
        tilted_sampler.setdefault("intermediate_reward_aggregation", "logmeanexp")
    phase = str(smc.get("experiment_phase", "estimation"))
    manifest = {
        "decoder_id": decoder_id(config),
        "evaluator_id": evaluator_id(config),
        "conditioning_mode": "unconditional",
        "experiment_phase": phase,
        "smc": smc,
        "diagnostic_components": {
            "lambdas": [float(value) for value in smc.get("diagnostic_lambdas", [])],
            "included_in_tilted_component_aggregation": False,
        },
    }
    if phase == "calibration":
        manifest["calibration_components"] = {
            "lambdas": [float(value) for value in smc["calibration_lambdas"]],
            "aggregation_performed": False,
            "purpose": "terminal-lambda selection only",
        }
    else:
        manifest["tilted_component_aggregation"] = {
            "symbol": "a_j",
            "count": len(smc["tilted_lambdas"]),
            "tilted_lambdas": [float(value) for value in smc["tilted_lambdas"]],
            "weights_a": [
                float(value) for value in smc["tilted_component_weights_a"]
            ],
            "component_weight_policy": smc["component_weight_policy"],
            "aggregation": "p_hat = sum_j a_j * p_hat_j",
            "separate_reverse_snis_normalization": True,
            "optimize_component_weights": bool(smc["optimize_component_weights"]),
        }
    return manifest


def smc_run_directory(config: dict[str, Any], output_root: Path | None = None) -> Path:
    manifest = smc_manifest(config)
    root = output_root or Path(config.get("outputs", {}).get("root", "outputs"))
    return (
        root
        / "mdlm_owt"
        / "smc"
        / manifest["decoder_id"]
        / manifest["evaluator_id"]
        / fingerprint(manifest)
    )


def run_benchmark_smc(
    config: dict[str, Any], *, output_root: Path | None, allow_network: bool
) -> Path:
    validate_benchmark_config(config)
    decoder = config["decoder"]
    if decoder["generation_mode"] != "unconditional":
        raise ValueError("the current SMC benchmark requires unconditional generation")
    decoder_identifier = decoder_id(config)
    evaluator_identifier = evaluator_id(config)
    manifest = smc_manifest(config)
    smc_id = fingerprint(manifest)
    run_dir = smc_run_directory(config, output_root)
    initialize_manifest(run_dir, {"smc_id": smc_id, **manifest})
    if (run_dir / "summary.json").exists():
        return run_dir

    generator_runtime = load_generator_runtime(config, local_files_only=not allow_network)
    evaluator_runtime = load_evaluator_runtime(config, local_files_only=not allow_network)
    numerical_batching = config["evaluation"]["numerical_batching"]
    evaluator = ToxicityEvaluator(
        evaluator_runtime.classifier,
        evaluator_runtime.classifier_tokenizer,
        evaluator_runtime.device,
        positive_label_index=int(config["evaluation"]["classifier"]["positive_label_index"]),
        batch_size=int(config["evaluation"].get("batch_size", 16)),
        padding=str(numerical_batching["padding"]),
        pad_final_batch_to_batch_size=bool(
            numerical_batching["pad_final_batch_to_batch_size"]
        ),
    )
    sampler = NativeMDLMSampler(
        generator_runtime.model,
        mask_token_id=int(decoder["mask_token_id"]),
        predictor=str(decoder["predictor"]),
        noise_epsilon=float(decoder["noise_schedule"]["epsilon"]),
        sampling_epsilon=float(decoder["sampling_epsilon"]),
        final_noise_removal=bool(decoder["final_noise_removal"]),
        final_noise_removal_method=str(decoder["final_noise_removal_method"]),
    )
    fixed_prefix = torch.tensor(
        decoder["fixed_prefix_token_ids"],
        dtype=torch.long,
        device=generator_runtime.device,
    )
    if fixed_prefix.numel() != 0:
        raise ValueError("unconditional SMC cannot have a fixed prefix")
    reward_cfg = config["evaluation"]["reward"]
    event_cfg = config["evaluation"]["event"]
    functions = UnconditionalToxicityFunctions(
        generator_tokenizer=generator_runtime.tokenizer,
        evaluator=evaluator,
        reward_completion_tokens=int(reward_cfg["completion_tokens"]),
        event_completion_tokens=int(event_cfg["completion_tokens"]),
        reward_max_length=int(reward_cfg["classifier_max_length"]),
        event_max_length=int(event_cfg["classifier_max_length"]),
        event_text_scope=str(event_cfg["text_scope"]),
        completion_trimming_tokenizer=evaluator_runtime.completion_trimming_tokenizer,
    )
    smc_cfg = config["smc"]
    tilted_sampler_cfg = smc_cfg.get("tilted_sampler", {"type": "fk_bootstrap"})
    tilted_sampler_type = str(tilted_sampler_cfg.get("type", "fk_bootstrap"))
    experiment_phase = str(smc_cfg.get("experiment_phase", "estimation"))
    active_specs = (
        [
            ("calibration", float(value))
            for value in smc_cfg["calibration_lambdas"]
        ]
        if experiment_phase == "calibration"
        else [("tilted", float(value)) for value in smc_cfg["tilted_lambdas"]]
    )
    component_specs = [
        ("diagnostic", float(value))
        for value in smc_cfg.get("diagnostic_lambdas", [])
    ] + active_specs
    components: list[dict[str, Any]] = []

    for run_index in range(_smc_run_count(smc_cfg)):
        for component_index, (component_role, lambda_final) in enumerate(
            component_specs
        ):
            component_path = run_dir / "components" / (
                f"run_{run_index:03d}_{component_role}_{component_index:03d}.json"
            )
            if component_path.exists():
                components.append(
                    json.loads(component_path.read_text(encoding="utf-8"))["summary"]
                )
                continue
            seed = _component_seed(
                smc_cfg, run_index=run_index, component_index=component_index
            )
            classifier_counters_before = (
                evaluator.score_call_count,
                evaluator.model_forward_count,
                evaluator.scored_text_count,
            )
            if generator_runtime.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(generator_runtime.device)
            component_config = SMCComponentConfig(
                num_particles=int(smc_cfg["num_particles"]),
                num_steps=int(decoder["denoising_steps"]),
                sequence_length=int(decoder["sequence_length"]),
                lambda_final=lambda_final,
                ess_threshold=float(smc_cfg.get("ess_threshold", 0.5)),
                resampling_method=str(smc_cfg.get("resampling_method", "multinomial")),
                reward_frequency=_reward_interval(smc_cfg),
                beta_start_fraction=float(smc_cfg.get("beta_start_fraction", 0.0)),
                final_resampling=bool(smc_cfg.get("final_resampling", False)),
                microbatch_size=(
                    int(smc_cfg["microbatch_size"])
                    if smc_cfg.get("microbatch_size") is not None
                    else None
                ),
                seed=seed,
                tilted_sampler_type=tilted_sampler_type,
                num_tempering_levels=int(
                    tilted_sampler_cfg.get("num_tempering_levels", 8)
                ),
                mcmc_steps_per_level=int(
                    tilted_sampler_cfg.get("mcmc_steps_per_level", 1)
                ),
                mutation_policy=str(
                    tilted_sampler_cfg.get("mutation_policy", "after_resampling")
                ),
                rollback_steps=tuple(
                    int(value) for value in tilted_sampler_cfg.get("rollback_steps", [])
                ),
                rollback_probabilities=tuple(
                    float(value)
                    for value in tilted_sampler_cfg.get(
                        "rollback_probabilities", []
                    )
                ),
                waste_free_chain_length=int(
                    tilted_sampler_cfg.get("chain_length", 1)
                ),
                num_x0_samples=int(
                    tilted_sampler_cfg.get("num_x0_samples", 4)
                ),
                intermediate_reward_aggregation=str(
                    tilted_sampler_cfg.get(
                        "intermediate_reward_aggregation", "logmeanexp"
                    )
                ),
            )
            result = run_smc_component(
                sampler=sampler,
                fixed_prefix_token_ids=fixed_prefix,
                reward_fn=functions.reward,
                event_fn=functions.event,
                config=component_config,
                device=generator_runtime.device,
            )
            classifier_counters_after = (
                evaluator.score_call_count,
                evaluator.model_forward_count,
                evaluator.scored_text_count,
            )
            classifier_score_calls = (
                classifier_counters_after[0] - classifier_counters_before[0]
            )
            classifier_model_forwards = (
                classifier_counters_after[1] - classifier_counters_before[1]
            )
            classifier_texts_scored = (
                classifier_counters_after[2] - classifier_counters_before[2]
            )
            if lambda_final == 0.0:
                expected = torch.full_like(
                    result.final_log_weights, -math.log(component_config.num_particles)
                )
                if not bool(torch.allclose(result.final_log_weights, expected, atol=1e-10)):
                    raise RuntimeError("lambda=0 did not retain uniform particle weights")
                if not math.isclose(
                    result.log_normalizer_estimate, 0.0, abs_tol=1e-10
                ):
                    raise RuntimeError("lambda=0 did not reduce to the base sampler")
                if (
                    result.tilted_sampler_type != "trajectory_waste_free"
                    and result.genealogy
                ):
                    raise RuntimeError("lambda=0 unexpectedly changed genealogy")
            corrected = result.reverse_snis.corrected_log_weights
            corrected_tensor = torch.tensor(corrected, dtype=torch.float64)
            distinct_root_ancestors = len(
                set(int(value) for value in result.root_ancestors.cpu().tolist())
            )
            root_counts = Counter(
                int(value) for value in result.root_ancestors.cpu().tolist()
            )
            root_probabilities = [
                value / component_config.num_particles for value in root_counts.values()
            ]
            root_ancestor_entropy = -sum(
                probability * math.log(probability)
                for probability in root_probabilities
            )
            normalized_root_ancestor_entropy = (
                root_ancestor_entropy / math.log(component_config.num_particles)
                if component_config.num_particles > 1
                else 1.0
            )
            terminal_rows = [
                tuple(int(value) for value in row)
                for row in result.terminal_tokens.detach().cpu().tolist()
            ]
            terminal_counts = Counter(terminal_rows)
            ess_before_resampling = [
                {
                    "level": int(row["level"]),
                    "ess": float(row["ess_before_resampling"]),
                }
                for row in result.level_diagnostics
                if row.get("resampled") and row.get("ess_before_resampling") is not None
            ]
            particles = []
            for index in range(component_config.num_particles):
                token_ids = [
                    int(value) for value in result.terminal_tokens[index].cpu().tolist()
                ]
                decoded_text = generator_runtime.tokenizer.decode(
                    token_ids, skip_special_tokens=True
                )
                particles.append(
                    {
                        "particle": index,
                        "conditioning_mode": "unconditional",
                        "root_ancestor": int(result.root_ancestors[index].cpu()),
                        "token_ids": token_ids,
                        "decoded_text": decoded_text,
                        "event_classifier_text": functions.last_event_texts[index],
                        "event_completion_token_ids": functions.last_event_token_ids[index],
                        "event_token_id_space": (
                            "mdlm_decoder"
                            if event_cfg["text_scope"]
                            == "first_generated_mdlm_tokens_only"
                            else "roberta_large"
                        ),
                        "reward": float(result.rewards[index].cpu()),
                        "reward_logits": functions.last_reward_rows[index]["logits"],
                        "reward_toxic_probability": functions.last_reward_rows[index][
                            "toxic_probability"
                        ],
                        "is_event": bool(result.events[index].cpu()),
                        "final_log_weight": float(result.final_log_weights[index].cpu()),
                        "reverse_corrected_log_weight": corrected[index],
                        "event_logits": functions.last_event_rows[index]["logits"],
                        "event_toxic_probability": functions.last_event_rows[index][
                            "toxic_probability"
                        ],
                        "waste_free_chain_id": (
                            int(result.waste_free_chain_ids[index].cpu())
                            if result.waste_free_chain_ids is not None
                            else None
                        ),
                        "waste_free_chain_position": (
                            int(result.waste_free_chain_positions[index].cpu())
                            if result.waste_free_chain_positions is not None
                            else None
                        ),
                    }
                )
            component_summary = {
                "conditioning_mode": "unconditional",
                "run_index": run_index,
                "lambda": lambda_final,
                "component_role": component_role,
                "included_in_final_aggregation": component_role == "tilted",
                "seed": seed,
                "num_particles": component_config.num_particles,
                "tilted_sampler_type": result.tilted_sampler_type,
                "reverse_snis_estimate": result.reverse_snis.estimate,
                "reverse_weight_ess": result.reverse_snis.corrected_ess,
                "reverse_weight_ess_over_n": result.reverse_snis.corrected_ess_over_n,
                "tilted_event_discoveries": result.reverse_snis.event_discoveries,
                "tilted_event_fraction_not_p0_estimate": (
                    result.reverse_snis.tilted_event_fraction
                ),
                "final_smc_ess": float(
                    ess_from_log_weights(result.final_log_weights)
                ),
                "pre_final_resampling_ess": ess_from_log_weights(
                    result.pre_final_resampling_log_weights
                ),
                "post_final_resampling_ess": ess_from_log_weights(
                    result.post_final_resampling_log_weights
                ),
                "pre_final_resampling_max_normalized_weight": float(
                    result.pre_final_resampling_log_weights.max().exp().cpu()
                ),
                "post_final_resampling_max_normalized_weight": float(
                    result.post_final_resampling_log_weights.max().exp().cpu()
                ),
                "reverse_max_normalized_weight": float(
                    corrected_tensor.max().exp().item()
                ),
                "resampling_count": len(result.genealogy),
                "ess_before_each_resampling": ess_before_resampling,
                "distinct_root_ancestors": distinct_root_ancestors,
                "distinct_root_ancestor_fraction": (
                    distinct_root_ancestors / component_config.num_particles
                ),
                "normalized_root_ancestor_entropy": (
                    normalized_root_ancestor_entropy
                ),
                "largest_root_family_fraction": (
                    max(root_counts.values()) / component_config.num_particles
                ),
                "distinct_terminal_sequences": len(terminal_counts),
                "distinct_terminal_sequence_fraction": (
                    len(terminal_counts) / component_config.num_particles
                ),
                "largest_terminal_duplicate_fraction": (
                    max(terminal_counts.values()) / component_config.num_particles
                ),
                "log_normalizer_estimate": result.log_normalizer_estimate,
                "nfe": result.nfe,
                "classifier_score_calls": classifier_score_calls,
                "classifier_model_forward_calls": classifier_model_forwards,
                "classifier_texts_scored": classifier_texts_scored,
                "runtime_seconds": result.runtime_seconds,
                "trajectory_storage_bytes": result.trajectory_storage_bytes,
                "waste_free_seed_count": (
                    component_config.num_particles
                    // component_config.waste_free_chain_length
                    if result.tilted_sampler_type == "trajectory_waste_free"
                    else None
                ),
                "waste_free_chain_length": (
                    component_config.waste_free_chain_length
                    if result.tilted_sampler_type == "trajectory_waste_free"
                    else None
                ),
                "waste_free_retained_states_per_level": (
                    component_config.num_particles
                    if result.tilted_sampler_type == "trajectory_waste_free"
                    else None
                ),
                "waste_free_rejected_self_transitions": (
                    sum(
                        int(row["rejected_self_transitions"])
                        for row in (result.waste_free_diagnostics or [])
                    )
                    if result.tilted_sampler_type == "trajectory_waste_free"
                    else None
                ),
                "mcmc_proposals": sum(
                    int(row["proposals"])
                    for row in (result.mutation_diagnostics or [])
                ),
                "mcmc_accepted": sum(
                    int(row["accepted"])
                    for row in (result.mutation_diagnostics or [])
                ),
                "mcmc_acceptance_rate": (
                    sum(
                        int(row["accepted"])
                        for row in (result.mutation_diagnostics or [])
                    )
                    / sum(
                        int(row["proposals"])
                        for row in (result.mutation_diagnostics or [])
                    )
                    if result.mutation_diagnostics
                    else None
                ),
                "mcmc_rollback_particle_steps": sum(
                    int(row["rollback_particle_steps"])
                    for row in (result.mutation_diagnostics or [])
                ),
                "base_decoder_particle_steps": (
                    component_config.num_particles * component_config.num_steps
                ),
                "total_decoder_particle_steps": (
                    component_config.num_particles * component_config.num_steps
                    + sum(
                        int(row["rollback_particle_steps"])
                        for row in (result.mutation_diagnostics or [])
                    )
                ),
                "reward_min": float(result.rewards.min().cpu()),
                "reward_mean": float(result.rewards.mean().cpu()),
                "reward_max": float(result.rewards.max().cpu()),
                "reward_quantiles": _reward_quantiles(result.rewards),
                "nonfinite_detected": False,
                "final_resampling": component_config.final_resampling,
                "lambda_zero_base_sampler_check": (
                    "passed" if lambda_final == 0.0 else "not_applicable"
                ),
                "peak_cuda_memory_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(generator_runtime.device))
                    if generator_runtime.device.type == "cuda"
                    else 0
                ),
            }
            atomic_write_json(
                component_path,
                {
                    "decoder_id": decoder_identifier,
                    "evaluator_id": evaluator_identifier,
                    "summary": component_summary,
                    "particles": particles,
                    "genealogy": result.genealogy,
                    "level_diagnostics": result.level_diagnostics,
                    "mutation_diagnostics": result.mutation_diagnostics or [],
                    "waste_free_diagnostics": result.waste_free_diagnostics or [],
                    "pre_final_resampling_log_weights": [
                        float(value)
                        for value in result.pre_final_resampling_log_weights.cpu().tolist()
                    ],
                    "post_final_resampling_log_weights": [
                        float(value)
                        for value in result.post_final_resampling_log_weights.cpu().tolist()
                    ],
                },
            )
            components.append(component_summary)

    tilted = [row for row in components if row["component_role"] == "tilted"]
    calibration = [
        row for row in components if row["component_role"] == "calibration"
    ]
    diagnostics = [row for row in components if row["component_role"] == "diagnostic"]
    summary = {
        "conditioning_mode": "unconditional",
        "experiment_phase": experiment_phase,
        "decoder_id": decoder_identifier,
        "evaluator_id": evaluator_identifier,
        "smc_id": smc_id,
        "diagnostic_components": diagnostics,
    }
    if experiment_phase == "calibration":
        summary.update(
            {
                "calibration_components": calibration,
                "aggregation_performed": False,
                "selection_rule": (
                    "prefer reverse ESS/N > 0.1 (ideally > 0.2), nonzero events, "
                    "and no severe genealogical collapse; do not select automatically"
                ),
            }
        )
    else:
        summary.update(
            {
                "tilted_components": tilted,
                "aggregate": _aggregate(tilted, config),
            }
        )
    atomic_write_json(run_dir / "summary.json", summary)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    print(
        run_benchmark_smc(
            load_yaml(args.config),
            output_root=args.output_root,
            allow_network=args.allow_network,
        )
    )
    if torch.cuda.is_available():
        print(f"peak_cuda_memory_allocated_bytes={torch.cuda.max_memory_allocated()}")


if __name__ == "__main__":
    main()
