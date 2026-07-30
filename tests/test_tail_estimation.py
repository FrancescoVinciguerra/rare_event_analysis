from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rea_llms.data_reuse_audit import (
    classify_component_archives,
    prompt_sha256,
    selected_archive_paths,
)
from rea_llms.tail_estimation import (
    MixtureComponentPopulation,
    aggregate_terminal_lambda_runs,
    capped_potential,
    corrected_base_log_weights,
    diagnose_component_reverse_estimator,
    effective_sample_size_from_log_weights,
    estimate_base_expectation,
    estimate_base_tail_probabilities,
    estimate_base_tails_from_deterministic_mixture,
    estimate_base_tails_from_final_samples,
    log_mixture_density_ratio,
)
from suffix_direct_mh_smc_experiment.run_tail_validation import (
    ComponentSpec,
    _component_metadata,
    _load_component_archive,
    _save_component_archive,
    assess_reference_compatibility,
    build_lambda_schedule,
    cache_metadata_matches,
    component_run_id,
    component_seed,
    enforce_reference_compatibility,
    main as run_validation_main,
    parse_args as parse_validation_args,
)
from suffix_direct_mh_smc_experiment.run_tail_estimation import (
    _config_for_seed,
    parse_args,
    parse_lambda_schedules,
    tail_run_directory,
)


ROOT = Path(__file__).resolve().parents[1]


def test_official_reference_preflight_blocks_unverified_exact_validation() -> None:
    reference = json.loads(
        (ROOT / "results" / "reference_direct_sampling_summary.json").read_text(
            encoding="utf-8"
        )
    )
    args = parse_validation_args([])
    compatibility = assess_reference_compatibility(reference, args)
    assert compatibility.sampling_setup_match
    assert not compatibility.strict_observable_match_verified
    assert not compatibility.exact_reference_match
    with pytest.raises(RuntimeError, match="strict equivalence"):
        enforce_reference_compatibility(
            compatibility,
            smoke_mode=False,
            allow_unverified_observable_reference=False,
        )
    enforce_reference_compatibility(
        compatibility,
        smoke_mode=False,
        allow_unverified_observable_reference=True,
    )


def test_reference_preflight_rejects_sampling_mismatch_except_in_smoke() -> None:
    reference = json.loads(
        (ROOT / "results" / "reference_direct_sampling_summary.json").read_text(
            encoding="utf-8"
        )
    )
    args = parse_validation_args(["--max-new-tokens", "8"])
    compatibility = assess_reference_compatibility(reference, args)
    assert not compatibility.sampling_checks["completion_tokens"]
    with pytest.raises(RuntimeError, match="completion_tokens"):
        enforce_reference_compatibility(
            compatibility,
            smoke_mode=False,
            allow_unverified_observable_reference=True,
        )
    enforce_reference_compatibility(
        compatibility,
        smoke_mode=True,
        allow_unverified_observable_reference=False,
    )


def test_capped_potential_preserves_raw_values() -> None:
    raw = torch.tensor([2.0, 6.0, 9.0], dtype=torch.float64)
    capped = capped_potential(raw, 6.0)
    assert torch.equal(capped, torch.tensor([2.0, 6.0, 6.0], dtype=torch.float64))
    assert torch.equal(raw, torch.tensor([2.0, 6.0, 9.0], dtype=torch.float64))


def test_log_mixture_density_ratio_matches_direct_formula() -> None:
    raw = torch.tensor([1.0, 4.0], dtype=torch.float64)
    caps = torch.tensor([2.0, 5.0], dtype=torch.float64)
    lambdas = torch.tensor([-0.5, -1.0], dtype=torch.float64)
    log_z = torch.log(torch.tensor([1.4, 3.2], dtype=torch.float64))
    rho = torch.tensor([0.25, 0.75], dtype=torch.float64)
    actual = log_mixture_density_ratio(raw, caps, lambdas, log_z, rho)
    expected = []
    for value in raw.tolist():
        terms = [
            float(rho[index])
            * math.exp(-float(lambdas[index]) * min(value, float(caps[index])))
            / math.exp(float(log_z[index]))
            for index in range(2)
        ]
        expected.append(math.log(sum(terms)))
    assert torch.allclose(
        actual,
        torch.tensor(expected, dtype=torch.float64),
        atol=1e-12,
    )


def test_standard_ess_on_known_weights() -> None:
    log_weights = torch.log(torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64))
    ess = effective_sample_size_from_log_weights(log_weights)
    assert float(ess) == pytest.approx(1.0 / (0.5**2 + 0.25**2 + 0.25**2))


def test_deterministic_mixture_recovers_exact_toy_probability() -> None:
    base = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    raw = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    components = []
    for index, (cap, lambda_final) in enumerate(((2.0, -0.5), (3.0, -1.0))):
        potential = torch.clamp(raw, max=cap)
        unnormalized = base * torch.exp(-lambda_final * potential)
        z_value = unnormalized.sum()
        tilted = unnormalized / z_value
        components.append(
            MixtureComponentPopulation(
                run_id=f"run-{index}",
                seed=100 + index,
                cap=cap,
                lambda_final=lambda_final,
                log_normalizing_constant=float(torch.log(z_value)),
                raw_ari=raw,
                final_log_weights=torch.log(tilted),
            )
        )
    estimates = estimate_base_tails_from_deterministic_mixture(
        components,
        thresholds=[2.0],
    )
    assert float(estimates.estimated_probabilities[0]) == pytest.approx(0.7, abs=1e-12)
    assert float(torch.exp(estimates.corrected_log_weights).sum()) == pytest.approx(1.0)


def test_components_require_distinct_seed_and_run_id() -> None:
    component = MixtureComponentPopulation(
        run_id="same-run",
        seed=17,
        cap=6.0,
        lambda_final=-1.0,
        log_normalizing_constant=0.0,
        raw_ari=[1.0, 2.0],
        final_log_weights=[math.log(0.5), math.log(0.5)],
    )
    with pytest.raises(ValueError, match="run IDs"):
        estimate_base_tails_from_deterministic_mixture(
            [component, component], thresholds=[1.0]
        )
    second = MixtureComponentPopulation(
        **{**component.__dict__, "run_id": "other-run"}
    )
    with pytest.raises(ValueError, match="seeds"):
        estimate_base_tails_from_deterministic_mixture(
            [component, second], thresholds=[1.0]
        )


def test_component_seed_schedule_and_run_ids_are_independent() -> None:
    seeds = [component_seed(group, index) for group in (200, 300) for index in range(3)]
    assert len(seeds) == len(set(seeds))
    assert build_lambda_schedule(-0.6, 0.25) == (0.0, -0.25, -0.5, -0.6)
    metadata = {
        "seed_group": 200,
        "component_index": 0,
        "ari_cap": 6.0,
        "lambda_final": -1.0,
        "seed": seeds[0],
    }
    other = {**metadata, "seed": seeds[1], "component_index": 1}
    assert component_run_id(metadata) != component_run_id(other)


def test_component_cache_requires_exact_metadata(tmp_path) -> None:
    metadata = {
        "num_particles": 2,
        "seed_group": 200,
        "seed": component_seed(200, 0),
        "prompt": "Once",
    }
    path = tmp_path / "component.npz"
    _save_component_archive(
        path,
        metadata=metadata,
        raw_ari=np.asarray([5.0, 7.0]),
        phi=np.asarray([5.0, 6.0]),
        final_log_weights=np.log(np.asarray([0.4, 0.6])),
        log_normalizing_constant=1.2,
        runtime_seconds=3.0,
    )
    spec = ComponentSpec(cap=6.0, lambda_final=-1.0)
    loaded = _load_component_archive(
        path,
        expected_metadata=metadata,
        spec=spec,
        reused=True,
    )
    assert loaded.reused
    incompatible = {**metadata, "prompt": "Different"}
    assert not cache_metadata_matches(metadata, incompatible)
    with pytest.raises(ValueError, match="metadata mismatch"):
        _load_component_archive(
            path,
            expected_metadata=incompatible,
            spec=spec,
            reused=True,
        )


def _audit_metadata(
    *,
    prompt: str = "Once upon a time, in a big forest, there lived a rhinoc",
    num_particles: int = 2,
) -> dict:
    args = parse_validation_args(
        [
            "--prompt",
            prompt,
            "--seed-groups",
            "200",
            "--num-particles",
            str(num_particles),
            "--max-new-tokens",
            "4",
            "--mcmc-steps-per-level",
            "1",
            "--suffix-sizes",
            "1,4",
        ]
    )
    spec = ComponentSpec(cap=6.0, lambda_final=-0.5)
    return _component_metadata(
        args,
        spec,
        seed_group=200,
        component_index=0,
        seed=component_seed(200, 0),
        schedule=build_lambda_schedule(-0.5, 0.25),
    )


def _write_audit_archive(path: Path, metadata: dict) -> None:
    count = int(metadata.get("num_particles", 2))
    raw = np.linspace(5.0, 7.0, count)
    _save_component_archive(
        path,
        metadata=metadata,
        raw_ari=raw,
        phi=np.minimum(raw, 6.0),
        final_log_weights=np.full(count, -math.log(count)),
        log_normalizing_constant=1.2,
        runtime_seconds=3.0,
    )


def test_prompt_comparison_is_exact_and_hash_is_deterministic() -> None:
    prompt = "Once upon a time"
    assert prompt_sha256(prompt) == prompt_sha256(prompt)
    assert prompt_sha256(prompt) != prompt_sha256(prompt + " ")
    assert prompt_sha256(prompt) != prompt_sha256("once upon a time")


def test_archive_with_missing_metadata_is_unverifiable(tmp_path) -> None:
    metadata = _audit_metadata()
    incomplete_metadata = {key: value for key, value in metadata.items() if key != "prompt"}
    path = tmp_path / "missing_metadata.npz"
    _write_audit_archive(path, incomplete_metadata)
    records = classify_component_archives(
        [path],
        {component_run_id(metadata): metadata},
    )
    assert records[0].classification == "unverifiable"
    assert not records[0].selected_for_mixture


@pytest.mark.parametrize(
    "different_prompt",
    [
        "Once upon a time, in a big forest, there lived a rhinoc ",
        "once upon a time, in a big forest, there lived a rhinoc",
    ],
)
def test_archive_with_exact_prompt_mismatch_is_incompatible(
    tmp_path,
    different_prompt: str,
) -> None:
    expected = _audit_metadata()
    stored = _audit_metadata(prompt=different_prompt)
    path = tmp_path / "prompt_mismatch.npz"
    _write_audit_archive(path, stored)
    records = classify_component_archives(
        [path],
        {component_run_id(expected): expected},
    )
    assert records[0].classification == "incompatible"
    assert "prompt" in records[0].reasons[0]


def test_exact_archive_is_selected_and_smoke_archive_is_not(tmp_path) -> None:
    expected = _audit_metadata(num_particles=256)
    exact_path = tmp_path / f"{component_run_id(expected)}.npz"
    smoke_path = tmp_path / "smoke.npz"
    _write_audit_archive(exact_path, expected)
    _write_audit_archive(smoke_path, _audit_metadata(num_particles=8))
    records = classify_component_archives(
        [exact_path, smoke_path],
        {component_run_id(expected): expected},
    )
    by_name = {record.facts.path.name: record for record in records}
    assert by_name[exact_path.name].classification == "exactly_reusable"
    assert by_name[exact_path.name].selected_for_mixture
    assert by_name[smoke_path.name].classification == "comparable_but_not_cache_equivalent"
    assert not by_name[smoke_path.name].selected_for_mixture
    assert selected_archive_paths(records) == {
        component_run_id(expected): exact_path
    }


def test_duplicate_exact_seed_is_rejected(tmp_path) -> None:
    metadata = _audit_metadata()
    run_id = component_run_id(metadata)
    canonical = tmp_path / f"{run_id}.npz"
    duplicate = tmp_path / "copied_archive.npz"
    _write_audit_archive(canonical, metadata)
    _write_audit_archive(duplicate, metadata)
    records = classify_component_archives(
        [duplicate, canonical],
        {run_id: metadata},
    )
    assert sum(record.selected_for_mixture for record in records) == 1
    rejected = [record for record in records if not record.selected_for_mixture]
    assert rejected[0].classification == "incompatible"
    assert "duplicate" in "; ".join(rejected[0].reasons)


def test_excluded_archive_cannot_change_mixture_ess_or_denominator(tmp_path) -> None:
    expected = _audit_metadata(num_particles=2)
    exact_path = tmp_path / f"{component_run_id(expected)}.npz"
    excluded_path = tmp_path / "old_configuration.npz"
    _write_audit_archive(exact_path, expected)
    _write_audit_archive(excluded_path, _audit_metadata(num_particles=3))
    records = classify_component_archives(
        [exact_path, excluded_path],
        {component_run_id(expected): expected},
    )
    selected = selected_archive_paths(records)
    assert list(selected.values()) == [exact_path]
    loaded = _load_component_archive(
        exact_path,
        expected_metadata=expected,
        spec=ComponentSpec(cap=6.0, lambda_final=-0.5),
        reused=True,
    )
    estimate = estimate_base_tails_from_deterministic_mixture(
        [
            MixtureComponentPopulation(
                run_id=loaded.run_id,
                seed=loaded.seed,
                cap=loaded.spec.cap,
                lambda_final=loaded.spec.lambda_final,
                log_normalizing_constant=loaded.log_normalizing_constant,
                raw_ari=loaded.raw_ari,
                final_log_weights=loaded.final_log_weights,
            )
        ],
        thresholds=[6.0],
    )
    assert estimate.corrected_log_weights.numel() == 2
    assert float(estimate.corrected_ess) <= 2.0


def test_audit_only_reuses_archive_without_loading_model(tmp_path, monkeypatch) -> None:
    output_root = tmp_path / "audit"
    component_root = output_root / "component_runs"
    metadata = _audit_metadata(num_particles=2)
    archive_path = component_root / f"{component_run_id(metadata)}.npz"
    _write_audit_archive(archive_path, metadata)

    def fail_model_load(*args, **kwargs):
        raise AssertionError("audit-only must not load the model")

    monkeypatch.setattr(
        "suffix_direct_mh_smc_experiment.run_tail_validation.load_model_and_tokenizer",
        fail_model_load,
    )
    run_validation_main(
        [
            "--audit-only",
            "--smoke-mode",
            "--components-json",
            json.dumps([{"cap": 6.0, "lambda": -0.5}]),
            "--seed-groups",
            "200",
            "--num-particles",
            "2",
            "--max-new-tokens",
            "4",
            "--mcmc-steps-per-level",
            "1",
            "--suffix-sizes",
            "1,4",
            "--output-root",
            str(output_root),
        ]
    )
    audit = pd.read_csv(output_root / "data_reuse_audit.csv")
    diagnostics = pd.read_csv(output_root / "component_diagnostics.csv")
    comparison = pd.read_csv(output_root / "comparison.csv")
    report = (output_root / "REPORT.md").read_text(encoding="utf-8")
    assert audit["selected_for_mixture"].tolist() == [True]
    assert not diagnostics.empty
    assert not comparison.empty
    assert "Candidate archives found: `1`" in report
    assert "Selected for the mixture: `1`" in report


def test_component_reverse_probability_and_log_z_on_toy_model() -> None:
    base = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    raw = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    cap = 2.0
    lambda_final = -0.5
    potential = torch.clamp(raw, max=cap)
    unnormalized = base * torch.exp(-lambda_final * potential)
    z_value = unnormalized.sum()
    tilted = unnormalized / z_value
    diagnostic = diagnose_component_reverse_estimator(
        raw,
        torch.log(tilted),
        cap=cap,
        lambda_final=lambda_final,
        thresholds=[2.0],
    )
    assert float(diagnostic.estimated_probabilities[0]) == pytest.approx(0.7)
    assert float(diagnostic.reverse_log_normalizing_constant) == pytest.approx(
        float(torch.log(z_value))
    )


def test_single_component_mixture_equals_component_reverse_formula() -> None:
    component = MixtureComponentPopulation(
        run_id="single",
        seed=5,
        cap=2.0,
        lambda_final=-0.7,
        log_normalizing_constant=0.4,
        raw_ari=[0.0, 1.0, 3.0],
        final_log_weights=np.log([0.1, 0.3, 0.6]),
    )
    reverse = diagnose_component_reverse_estimator(
        component.raw_ari,
        component.final_log_weights,
        cap=component.cap,
        lambda_final=component.lambda_final,
        thresholds=[2.0],
    )
    mixture = estimate_base_tails_from_deterministic_mixture(
        [component],
        thresholds=[2.0],
    )
    assert float(mixture.estimated_probabilities[0]) == pytest.approx(
        float(reverse.estimated_probabilities[0])
    )


def test_nonuniform_smc_weights_are_not_dropped_from_mixture() -> None:
    common = {
        "run_id": "weighted",
        "seed": 9,
        "cap": 2.0,
        "lambda_final": -0.5,
        "log_normalizing_constant": 0.3,
        "raw_ari": [0.0, 3.0],
    }
    weighted = MixtureComponentPopulation(
        **common,
        final_log_weights=np.log([0.9, 0.1]),
    )
    uniform = MixtureComponentPopulation(
        **{**common, "run_id": "uniform", "seed": 10},
        final_log_weights=np.log([0.5, 0.5]),
    )
    weighted_probability = estimate_base_tails_from_deterministic_mixture(
        [weighted], thresholds=[2.0]
    ).estimated_probabilities[0]
    uniform_probability = estimate_base_tails_from_deterministic_mixture(
        [uniform], thresholds=[2.0]
    ).estimated_probabilities[0]
    assert float(weighted_probability) != pytest.approx(float(uniform_probability))


def test_component_weights_are_normalized_internally_and_rho_is_applied_once() -> None:
    first = MixtureComponentPopulation(
        run_id="first",
        seed=21,
        cap=2.0,
        lambda_final=-0.5,
        log_normalizing_constant=0.2,
        raw_ari=[0.0, 3.0],
        final_log_weights=np.log([0.8, 0.2]),
    )
    second = MixtureComponentPopulation(
        run_id="second",
        seed=22,
        cap=2.0,
        lambda_final=-0.5,
        log_normalizing_constant=0.2,
        raw_ari=[0.0, 3.0],
        final_log_weights=np.log([0.8, 0.2]) + 100.0,
    )
    one = estimate_base_tails_from_deterministic_mixture(
        [first], thresholds=[2.0]
    )
    combined = estimate_base_tails_from_deterministic_mixture(
        [first, second],
        thresholds=[2.0],
        mixture_coefficients=[0.2, 0.8],
    )
    assert float(combined.estimated_probabilities[0]) == pytest.approx(
        float(one.estimated_probabilities[0])
    )


def test_each_component_cap_changes_mixture_denominator() -> None:
    raw = torch.tensor([10.0], dtype=torch.float64)
    common = dict(
        raw_ari=raw,
        lambdas=[-1.0, -1.0],
        log_normalizing_constants=[0.0, 0.0],
        mixture_coefficients=[0.5, 0.5],
    )
    first = log_mixture_density_ratio(caps=[6.0, 8.0], **common)
    second = log_mixture_density_ratio(caps=[6.0, 10.0], **common)
    assert not torch.allclose(first, second)


def test_lambda_zero_preserves_normalized_smc_weights() -> None:
    phi = torch.tensor([0.0, 2.0, 5.0], dtype=torch.float64)
    smc_weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    corrected = torch.exp(corrected_base_log_weights(phi, smc_weights, 0.0))
    assert torch.allclose(corrected, smc_weights)
    expectation = estimate_base_expectation(
        phi,
        smc_weights,
        0.0,
        torch.tensor([1.0, 3.0, 7.0]),
    )
    assert float(expectation) == pytest.approx(4.6)


def test_corrected_weights_are_finite_nonnegative_and_normalized() -> None:
    corrected_log = corrected_base_log_weights(
        [0.0, 3.0, 10.0],
        [math.log(0.1), math.log(0.4), math.log(0.5)],
        -2.0,
        weights_are_log=True,
    )
    corrected = torch.exp(corrected_log)
    assert torch.isfinite(corrected).all()
    assert torch.all(corrected >= 0.0)
    assert float(corrected.sum()) == pytest.approx(1.0)
    estimates = estimate_base_tail_probabilities(
        [0.0, 3.0, 10.0],
        [math.log(0.1), math.log(0.4), math.log(0.5)],
        -2.0,
        [3.0],
        weights_are_log=True,
    )
    assert 0.0 < float(estimates.corrected_ess_over_n) <= 1.0


def test_tail_probabilities_are_monotone_for_sorted_thresholds() -> None:
    estimates = estimate_base_tail_probabilities(
        [1.0, 4.0, 8.0, 12.0],
        [0.1, 0.2, 0.3, 0.4],
        -0.5,
        [0.0, 4.0, 8.0, 20.0],
    )
    probabilities = estimates.estimated_probabilities
    assert torch.all(probabilities[:-1] >= probabilities[1:])
    assert estimates.raw_counts.tolist() == [4, 3, 2, 0]


def test_snis_converges_to_exact_base_probability_on_toy_space() -> None:
    base_probability = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    state_phi = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    lambda_final = -0.8
    tilted = base_probability * torch.exp(-lambda_final * state_phi)
    tilted /= tilted.sum()
    generator = torch.Generator().manual_seed(701)
    indices = torch.multinomial(tilted, 50_000, replacement=True, generator=generator)
    sampled_phi = state_phi.index_select(0, indices)
    estimates = estimate_base_tail_probabilities(
        sampled_phi,
        torch.ones(sampled_phi.numel(), dtype=torch.float64),
        lambda_final,
        [2.0],
    )
    exact_probability = float(base_probability[state_phi >= 2.0].sum())
    assert float(estimates.estimated_probabilities[0]) == pytest.approx(
        exact_probability,
        abs=0.015,
    )


def test_serialized_estimator_uses_final_lambda_and_final_log_weights_only() -> None:
    samples = [
        {
            "phi": 0.0,
            "log_weight": math.log(0.4),
            "normalized_weight": 0.99,
            "previous_log_weight": 50.0,
            "lambda_final": -1.0,
        },
        {
            "phi": 2.0,
            "log_weight": math.log(0.6),
            "normalized_weight": 0.01,
            "previous_log_weight": -50.0,
            "lambda_final": -1.0,
        },
    ]
    estimates = estimate_base_tails_from_final_samples(
        samples,
        lambda_final=-1.0,
        thresholds=[2.0],
    )
    expected = 0.6 * math.exp(-2.0) / (0.4 + 0.6 * math.exp(-2.0))
    assert float(estimates.estimated_probabilities[0]) == pytest.approx(expected)

    invalid = [dict(samples[0]), {**samples[1], "lambda_final": -0.5}]
    with pytest.raises(ValueError, match="final lambda"):
        estimate_base_tails_from_final_samples(
            invalid,
            lambda_final=-1.0,
            thresholds=[2.0],
        )


def test_tail_runner_builds_only_exact_direct_metropolis_smc() -> None:
    args = parse_args(
        [
            "--num-particles",
            "8",
            "--max-new-tokens",
            "4",
            "--lambdas",
            "0,-0.5",
            "--mcmc-steps-per-level",
            "1",
            "--suffix-sizes",
            "1,4",
            "--suffix-rho-schedule",
            "0.5,0.5",
        ]
    )
    config = _config_for_seed(
        args,
        seed=17,
        lambdas=(0.0, -0.5),
        thresholds=(2.0, 4.0),
    )
    assert config.resampling
    assert not hasattr(config, "balancing_function")
    assert not hasattr(config, "proposal_temperature")


def test_primary_colab_notebook_is_valid_and_calls_active_runner() -> None:
    path = ROOT / "notebooks" / "REA_Suffix_SMC_Quickstart.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = []
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        python_source = "\n".join(
            "pass" if line.lstrip().startswith(("%", "!")) else line
            for line in source.splitlines()
        )
        ast.parse(python_source, filename=f"{path.name}:cell-{index}")
        code_cells.append(source)
    joined = "\n".join(code_cells)
    assert joined.count("EDIT PARAMETERS HERE") == 1
    assert "suffix_direct_mh_smc_experiment/run_tail_validation.py" in joined
    assert "COMPONENTS" in joined
    assert "SEED_GROUPS" in joined
    assert "LAMBDA_STEP" in joined
    assert "SKIP_COMPLETED_RUNS" in joined
    assert "ALLOW_UNVERIFIED_OBSERVABLE_REFERENCE" in joined
    assert "RUN_CACHE_AUDIT" in joined
    assert "RUN_SMOKE_TEST" in joined
    assert "RUN_PRODUCTION" in joined
    assert "--components-json" in joined
    assert "--smoke-mode" in joined
    assert "--audit-only" in joined
    assert "Loaded component archives" not in joined
    assert "HAS_AUDITED_PRODUCTION_RESULTS" in joined
    assert "suffix SMC mixture versus official direct sampling" in joined
    assert "SEED_GROUPS = [200]" in joined
    assert "official_annealed_tps" not in joined


def _completed_sweep_payload(
    configuration: str,
    seed: int,
    lambda_final: float,
    *,
    estimate: float,
    raw_fraction: float,
    corrected_ess_over_n: float,
) -> dict:
    return {
        "algorithm": "suffix_direct_mh_smc",
        "configuration_label": configuration,
        "seed": seed,
        "repository_commit": "test-commit",
        "config": {
            "seed": seed,
            "lambdas": [0.0, lambda_final],
            "num_particles": 10,
        },
        "summary": {
            "lambda_final": lambda_final,
            "corrected_importance_ess_over_n": corrected_ess_over_n,
            "tails": [
                {
                    "threshold": 8.0,
                    "raw_count": round(10 * raw_fraction),
                    "raw_tilted_fraction": raw_fraction,
                    "estimated_probability_under_p_m": estimate,
                }
            ],
        },
    }


def test_terminal_configurations_have_distinct_output_directories(tmp_path) -> None:
    first = tail_run_directory(
        tmp_path,
        "sweep",
        200,
        configuration_label="lambda_m1",
    )
    second = tail_run_directory(
        tmp_path,
        "sweep",
        200,
        configuration_label="lambda_m2",
    )
    assert first != second
    assert first == tmp_path / "sweep" / "lambda_m1" / "seed_200"
    assert second == tmp_path / "sweep" / "lambda_m2" / "seed_200"
    schedules = parse_lambda_schedules(
        '{"lambda_m1":"0,-0.5,-1", "lambda_m2":[0,-1,-2]}'
    )
    assert schedules["lambda_m1"][-1] == -1.0
    assert schedules["lambda_m2"][-1] == -2.0


def test_sweep_aggregation_never_combines_terminal_configurations() -> None:
    payloads = [
        _completed_sweep_payload(
            "lambda_m1", seed, -1.0, estimate=estimate, raw_fraction=0.3,
            corrected_ess_over_n=0.4,
        )
        for seed, estimate in [(200, 0.10), (201, 0.20)]
    ] + [
        _completed_sweep_payload(
            "lambda_m2", seed, -2.0, estimate=estimate, raw_fraction=0.6,
            corrected_ess_over_n=0.2,
        )
        for seed, estimate in [(200, 0.02), (201, 0.04)]
    ]
    aggregate = aggregate_terminal_lambda_runs(
        payloads,
        min_corrected_ess_over_n=0.1,
        min_raw_event_fraction=0.05,
        max_raw_event_fraction=0.95,
    )
    rows = {row["configuration"]: row for row in aggregate["aggregate_rows"]}
    assert set(rows) == {"lambda_m1", "lambda_m2"}
    assert rows["lambda_m1"]["completed_seeds"] == 2
    assert rows["lambda_m1"]["mean_estimated_probability_under_p_m"] == pytest.approx(0.15)
    assert rows["lambda_m2"]["mean_estimated_probability_under_p_m"] == pytest.approx(0.03)

    incompatible = dict(payloads[0])
    incompatible["seed"] = 202
    incompatible["config"] = {**payloads[0]["config"], "seed": 202, "num_particles": 20}
    with pytest.raises(ValueError, match="mixes different run parameters"):
        aggregate_terminal_lambda_runs(
            [payloads[0], incompatible],
            min_corrected_ess_over_n=0.1,
            min_raw_event_fraction=0.05,
            max_raw_event_fraction=0.95,
        )


def test_overlap_flag_is_diagnostic_and_does_not_change_estimate() -> None:
    payload = _completed_sweep_payload(
        "lambda_m1",
        200,
        -1.0,
        estimate=0.07,
        raw_fraction=0.5,
        corrected_ess_over_n=0.2,
    )
    usable = aggregate_terminal_lambda_runs(
        [payload],
        min_corrected_ess_over_n=0.1,
        min_raw_event_fraction=0.05,
        max_raw_event_fraction=0.95,
    )
    unusable = aggregate_terminal_lambda_runs(
        [payload],
        min_corrected_ess_over_n=0.3,
        min_raw_event_fraction=0.05,
        max_raw_event_fraction=0.95,
    )
    usable_row = usable["aggregate_rows"][0]
    unusable_row = unusable["aggregate_rows"][0]
    assert usable_row["overlap_usable"] is True
    assert unusable_row["overlap_usable"] is False
    assert usable_row["mean_estimated_probability_under_p_m"] == pytest.approx(
        unusable_row["mean_estimated_probability_under_p_m"]
    )
