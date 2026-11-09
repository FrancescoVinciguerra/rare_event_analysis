"""Canonical benchmark configuration and deterministic identifiers."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = 2


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        # JSON is a valid YAML subset, but PyYAML's scalar resolver does not
        # preserve every JSON number.  In particular, ``1e-05`` is loaded as a
        # string, which changes distribution fingerprints even when runtime
        # code later casts it back to float.  Resolved comparison configs are
        # JSON, so parse them with the JSON parser that defines their types.
        payload = (
            json.load(handle)
            if path.suffix.lower() == ".json"
            else yaml.safe_load(handle)
        )
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    base = load_yaml(path.parent / str(parent))
    return _deep_merge(base, payload)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _require(mapping: dict[str, Any], key: str, expected_type: type) -> Any:
    value = mapping.get(key)
    if not isinstance(value, expected_type):
        raise ValueError(f"{key!r} must be {expected_type.__name__}")
    return value


def validate_benchmark_config(config: dict[str, Any]) -> None:
    decoder = _require(config, "decoder", dict)
    generation_mode = decoder.get("generation_mode")
    if generation_mode not in {"unconditional", "fixed_prefix"}:
        raise ValueError("generation_mode must be unconditional or fixed_prefix")
    fixed_prefix = _require(decoder, "fixed_prefix_token_ids", list)
    if any(not isinstance(value, int) or value < 0 for value in fixed_prefix):
        raise ValueError("fixed_prefix_token_ids must contain non-negative integers")
    prompt = decoder.get("prompt")
    if generation_mode == "unconditional":
        if fixed_prefix:
            raise ValueError("unconditional generation requires an empty fixed prefix")
        if prompt is not None:
            raise ValueError("unconditional generation requires prompt: null")
    elif not fixed_prefix:
        raise ValueError("fixed_prefix generation requires explicit token IDs")

    if decoder.get("predictor") not in {"ddpm", "ddpm_cache"}:
        raise ValueError("predictor must be ddpm or ddpm_cache")
    if float(decoder.get("temperature", 0.0)) != 1.0:
        raise ValueError("the native benchmark requires temperature=1")
    if decoder.get("top_k") is not None or decoder.get("top_p") is not None:
        raise ValueError("top-k and top-p are outside the native base distribution")
    if int(decoder.get("sequence_length", 0)) <= 0:
        raise ValueError("sequence_length must be positive")
    if int(decoder.get("evaluated_completion_tokens", 0)) <= 0:
        raise ValueError("evaluated_completion_tokens must be positive")
    if int(decoder.get("denoising_steps", 0)) <= 0:
        raise ValueError("denoising_steps must be positive")
    sampling_epsilon = decoder.get("sampling_epsilon")
    if (
        isinstance(sampling_epsilon, bool)
        or not isinstance(sampling_epsilon, (int, float))
        or not math.isfinite(float(sampling_epsilon))
        or not 0.0 < float(sampling_epsilon) < 1.0
    ):
        raise ValueError("sampling_epsilon must be a finite numeric value in (0, 1)")
    if decoder.get("final_noise_removal_method") not in {
        "stochastic_categorical",
        "argmax",
    }:
        raise ValueError("final_noise_removal_method is invalid")
    noise = _require(decoder, "noise_schedule", dict)
    if noise.get("type") != "loglinear":
        raise ValueError("the first benchmark implements the official loglinear schedule")

    evaluation = _require(config, "evaluation", dict)
    numerical_batching = _require(evaluation, "numerical_batching", dict)
    if numerical_batching.get("padding") != "max_length":
        raise ValueError(
            "evaluation.numerical_batching.padding must be max_length"
        )
    if numerical_batching.get("pad_final_batch_to_batch_size") is not True:
        raise ValueError(
            "evaluation.numerical_batching.pad_final_batch_to_batch_size must be true"
        )

    smc = config.get("smc")
    if smc is not None:
        if not isinstance(smc, dict):
            raise ValueError("smc must be a mapping")
        phase = str(smc.get("experiment_phase", "estimation"))
        if phase not in {"calibration", "estimation"}:
            raise ValueError("smc.experiment_phase must be calibration or estimation")
        if phase == "calibration":
            calibration_lambdas = [
                float(value) for value in _require(smc, "calibration_lambdas", list)
            ]
            if not calibration_lambdas or any(
                not math.isfinite(value) or value <= 0.0
                for value in calibration_lambdas
            ):
                raise ValueError("calibration_lambdas must be finite and positive")
            if len(set(calibration_lambdas)) != len(calibration_lambdas):
                raise ValueError("calibration_lambdas must be unique")
        else:
            lambdas = [
                float(value) for value in _require(smc, "tilted_lambdas", list)
            ]
            weights = [
                float(value)
                for value in _require(smc, "tilted_component_weights_a", list)
            ]
            if not lambdas or len(weights) != len(lambdas):
                raise ValueError("one component weight is required per lambda")
            if any(not math.isfinite(value) or value <= 0.0 for value in lambdas):
                raise ValueError("tilted_lambdas must be finite and positive")
            if any(weight < 0.0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-9:
                raise ValueError(
                    "tilted component weights must be non-negative and sum to one"
                )
            component_weight_policy = smc.get("component_weight_policy")
            if component_weight_policy not in {"uniform", "explicit"}:
                raise ValueError("component_weight_policy must be uniform or explicit")
            if bool(smc.get("optimize_component_weights", True)):
                raise ValueError("SMC component-weight optimization is not implemented")
            uniform_component_weight = 1.0 / len(lambdas)
            if component_weight_policy == "uniform" and any(
                abs(weight - uniform_component_weight) > 1e-12 for weight in weights
            ):
                raise ValueError("every SMC component weight must equal 1/J")
        diagnostic_lambdas = [
            float(value) for value in smc.get("diagnostic_lambdas", [])
        ]
        if len(set(diagnostic_lambdas)) != len(diagnostic_lambdas):
            raise ValueError("diagnostic_lambdas must be unique")
        active_lambdas = (
            calibration_lambdas if phase == "calibration" else lambdas
        )
        overlap = set(diagnostic_lambdas).intersection(active_lambdas)
        if overlap:
            raise ValueError(
                "diagnostic lambdas are excluded from aggregation and must not also "
                f"be tilted components: {sorted(overlap)}"
            )
        run_count = smc.get("num_independent_runs", smc.get("independent_runs"))
        if not isinstance(run_count, int) or run_count <= 0:
            raise ValueError("num_independent_runs must be a positive integer")
        if (
            "num_independent_runs" in smc
            and "independent_runs" in smc
            and smc["num_independent_runs"] != smc["independent_runs"]
        ):
            raise ValueError("num_independent_runs and independent_runs disagree")
        reward_interval = smc.get(
            "reward_interval", smc.get("reward_frequency", 20)
        )
        if not isinstance(reward_interval, int) or reward_interval <= 0:
            raise ValueError("reward_interval must be a positive integer")
        if (
            "reward_interval" in smc
            and "reward_frequency" in smc
            and smc["reward_interval"] != smc["reward_frequency"]
        ):
            raise ValueError("reward_interval and reward_frequency disagree")
        microbatch_size = smc.get("microbatch_size")
        if microbatch_size is not None and (
            not isinstance(microbatch_size, int) or microbatch_size <= 0
        ):
            raise ValueError("microbatch_size must be a positive integer")
        tilted_sampler = smc.get("tilted_sampler", {"type": "fk_bootstrap"})
        if not isinstance(tilted_sampler, dict):
            raise ValueError("smc.tilted_sampler must be a mapping")
        tilted_sampler_type = tilted_sampler.get("type")
        if tilted_sampler_type not in {
            "fk_bootstrap",
            "trajectory_resample_move",
            "trajectory_waste_free",
        }:
            raise ValueError(
                "smc.tilted_sampler.type must be fk_bootstrap, "
                "trajectory_resample_move, or trajectory_waste_free"
            )
        if tilted_sampler_type == "fk_bootstrap":
            x0_sampling = tilted_sampler.get(
                "intermediate_x0_sampling", "mdlm_multinomial"
            )
            if x0_sampling != "mdlm_multinomial":
                raise ValueError(
                    "FK intermediate_x0_sampling must be mdlm_multinomial"
                )
            num_x0_samples = tilted_sampler.get("num_x0_samples", 4)
            if not isinstance(num_x0_samples, int) or num_x0_samples <= 0:
                raise ValueError("FK num_x0_samples must be a positive integer")
            aggregation = tilted_sampler.get(
                "intermediate_reward_aggregation", "logmeanexp"
            )
            if aggregation != "logmeanexp":
                raise ValueError(
                    "FK intermediate_reward_aggregation must be logmeanexp"
                )
        if tilted_sampler_type in {
            "trajectory_resample_move",
            "trajectory_waste_free",
        }:
            implementation_version = tilted_sampler.get("implementation_version")
            if not isinstance(implementation_version, str) or not implementation_version:
                raise ValueError(
                    "trajectory SMC backends require a non-empty "
                    "implementation_version"
                )
            if decoder.get("predictor") != "ddpm":
                raise ValueError(
                    "trajectory SMC backends currently require predictor: ddpm"
                )
            tempering_levels = tilted_sampler.get("num_tempering_levels")
            if not isinstance(tempering_levels, int) or tempering_levels <= 0:
                raise ValueError("num_tempering_levels must be a positive integer")
            if tilted_sampler.get("mutation_policy") not in {
                "after_resampling",
                "every_level",
            }:
                raise ValueError(
                    "mutation_policy must be after_resampling or every_level"
                )
            rollback_steps = tilted_sampler.get("rollback_steps")
            if not isinstance(rollback_steps, list) or not rollback_steps:
                raise ValueError("rollback_steps must be a non-empty list")
            denoising_steps = int(decoder["denoising_steps"])
            if any(
                not isinstance(step, int) or step <= 0 or step > denoising_steps
                for step in rollback_steps
            ):
                raise ValueError(
                    "rollback_steps must be integer depths within denoising_steps"
                )
            if len(set(rollback_steps)) != len(rollback_steps):
                raise ValueError("rollback_steps must be unique")
            probabilities = tilted_sampler.get("rollback_probabilities")
            if probabilities is not None:
                if not isinstance(probabilities, list) or len(probabilities) != len(
                    rollback_steps
                ):
                    raise ValueError(
                        "rollback_probabilities must match rollback_steps"
                    )
                if any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                    for value in probabilities
                ) or not math.isclose(
                    sum(float(value) for value in probabilities),
                    1.0,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        "rollback_probabilities must be a non-negative simplex"
                    )
        if tilted_sampler_type == "trajectory_resample_move":
            mcmc_steps = tilted_sampler.get("mcmc_steps_per_level")
            if not isinstance(mcmc_steps, int) or mcmc_steps <= 0:
                raise ValueError("mcmc_steps_per_level must be a positive integer")
        if tilted_sampler_type == "trajectory_waste_free":
            chain_length = tilted_sampler.get("chain_length")
            if not isinstance(chain_length, int) or chain_length < 2:
                raise ValueError(
                    "trajectory_waste_free chain_length must be an integer >= 2"
                )
            particle_count = int(smc["num_particles"])
            if particle_count % chain_length != 0:
                raise ValueError(
                    "smc.num_particles must be divisible by trajectory_waste_free "
                    "chain_length"
                )
            if tilted_sampler.get("mutation_policy") != "every_level":
                raise ValueError(
                    "trajectory_waste_free requires mutation_policy: every_level"
                )
            if smc.get("final_resampling") is not False:
                raise ValueError(
                    "trajectory_waste_free requires smc.final_resampling: false"
                )


def decoder_manifest(config: dict[str, Any]) -> dict[str, Any]:
    validate_benchmark_config(config)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "mdlm_decoder_distribution",
        "decoder": copy.deepcopy(config["decoder"]),
        "source_provenance": copy.deepcopy(config.get("source_provenance", {})),
    }


def decoder_id(config: dict[str, Any]) -> str:
    return fingerprint(decoder_manifest(config))


def evaluator_manifest(config: dict[str, Any]) -> dict[str, Any]:
    evaluation = _require(config, "evaluation", dict)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "mdlm_rare_event_evaluator",
        "evaluation": copy.deepcopy(evaluation),
    }


def evaluator_id(config: dict[str, Any]) -> str:
    return fingerprint(evaluator_manifest(config))
