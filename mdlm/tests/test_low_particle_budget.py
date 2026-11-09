from pathlib import Path

import pytest

from diffusion_toxicity_smc_experiment.accuracy_budget import assigned_runs, study_configs
from diffusion_toxicity_smc_experiment.particle_scaling import selected_populations


STUDY = Path(__file__).resolve().parents[1] / "configs/mdlm_owt_low_particle_budget.yaml"


def test_one_ladder_per_seed_and_size():
    study, configs = study_configs(STUDY)
    assert [(c["_exploration"]["N"], c["_exploration"]["schedule"]) for c in configs] == [
        (64, [.5, 1., 1.5, 2., 2.5, 3.]), (96, [.5, 1., 1.5, 2., 2.5, 3.]),
    ]
    tasks = assigned_runs(study, configs, 0, 1)
    assert len(tasks) == len({seed for _, seed in tasks}) == 48
    assert all(len(assigned_runs(study, configs, worker, 3)) == 16 for worker in range(3))
    assert study["checkpoint_tilts"] == {.99: [.5], .995: [1.], .999: [.5, 3.]}


def test_ultra_low_budget_keeps_both_estimates_at_intermediate_checkpoints():
    study, configs = study_configs(STUDY.with_name("mdlm_owt_ultra_low_particle_budget.yaml"))
    assert [(c["_exploration"]["N"], c["_exploration"]["schedule"]) for c in configs] == [
        (32, [.5, 1., 1.5]), (48, [.5, 1., 1.5]),
    ]
    assert study["checkpoint_tilts"] == {.99: [.5], .995: [1.]}
    assert study["thresholds"] == [.99, .995]
    seeds = {seed for _, seed in assigned_runs(study, configs, 0, 1)}
    assert len(seeds) == 48
    assert all(not config["smc"]["final_resampling"] for config in configs)


def test_deeper_tilt_budget_keeps_targets_paired_on_one_ladder():
    study, configs = study_configs(STUDY.with_name("mdlm_owt_deeper_tilt_budget.yaml"))
    assert [(c["_exploration"]["N"], c["_exploration"]["schedule"]) for c in configs] == [
        (24, [.5, 1., 1.5, 2.]), (32, [.5, 1., 1.5, 2.]),
    ]
    assert study["checkpoint_tilts"] == {.99: [.5, 1.], .995: [1., 1.5, 2.]}
    seeds = {seed for _, seed in assigned_runs(study, configs, 0, 1)}
    assert len(seeds) == 48
    assert all(not config["smc"]["final_resampling"] for config in configs)


def test_extreme_tail_scaling_stores_each_level_and_keeps_crude_reference_separate():
    study, configs = study_configs(STUDY.with_name("mdlm_owt_extreme_tail_scaling.yaml"))
    config, = configs
    expected = [k / 2 for k in range(1, 19)]
    assert config["_exploration"]["schedule"] == expected
    assert config["smc"]["num_particles"] == 256
    assert not config["smc"]["final_resampling"]
    assert study["checkpoint_tilts"] == {.9995: expected}
    assert study["mc_pool"] == "camera_ready_gradual_budget"
    assert study["reference"]["samples"] == [200000] and study["reference"]["discoveries"] == [2]
    assert len({seed for _, seed in assigned_runs(study, configs, 0, 1)}) == 12


def test_terminal_particle_scaling_changes_only_particle_count_between_configs():
    study, configs = study_configs(STUDY.with_name("mdlm_owt_progressive_tail_particle_scaling.yaml"))
    assert [c["smc"]["num_particles"] for c in configs] == [320, 384, 448, 512]
    assert all(c["_exploration"]["schedule"] == [k / 2 for k in range(1, 14)] for c in configs)
    assert study["checkpoint_tilts"] == {.9995: [6.5]}
    assert all(not c["smc"]["final_resampling"] for c in configs)
    assert study["independent_runs"] == 24
    assert len({seed for _, seed in assigned_runs(study, configs, 0, 1)}) == 96
    assert all(len(assigned_runs(study, configs, worker, 3)) == 32 for worker in range(3))
    assert study["mc_pool_samples"] // study["independent_runs"] == 19823


def test_targets_use_each_checkpoints_own_clock_and_native_cost():
    study, _ = study_configs(STUDY)
    n = 64
    moves, checkpoints, levels = [], [], []
    native = n * 33
    checkpoints.append(dict(level=0, sweep=0, num_particles=n,
                            native_transitions_cumulative=native, runtime_seconds_cumulative=10))
    for level in range(1, 7):
        for sweep in range(4):
            moves.append(dict(level=level, sweep=sweep, rollback_particle_steps=16*n,
                              proposals=n, accepted=n//2))
            native += 17*n
            checkpoints.append(dict(
                level=level, sweep=sweep + 1, num_particles=n, log_z_hat=0, smc_ess_fraction=1,
                **{"lambda": level * .5}, native_transitions_cumulative=native,
                runtime_seconds_cumulative=10 + len(moves)*3,
            ))
        levels.append(dict(eta=level * .5, log_normalizer_estimate=0, distinct_root_ancestors=n))
    population = dict(
        summary=dict(num_particles=n, **{"lambda": 3.}, runtime_seconds=82, mcmc_proposals=24*n,
                     total_decoder_particle_steps=32*n + 24*16*n, mcmc_acceptance_rate=.5,
                     final_smc_ess=n, distinct_root_ancestors=n),
        mutation_diagnostics=moves, population_checkpoints=checkpoints, level_diagnostics=levels,
    )
    refs = {.99: .00222, .995: .001118, .999: .00013}
    points = list(selected_populations(population, study, refs))
    assert [(p["tilt"], p["seconds"], tuple(events)) for p, events in points] == [
        (.5, 22, (.99, .999)), (1., 34, (.995,)), (3., 82, (.999,)),
    ]
    assert [p["native"] for p, _ in points] == [n*(33 + 17*k) for k in (4, 8, 24)]
    checkpoints[4]["native_transitions_cumulative"] += 1
    with pytest.raises(ValueError, match="transition ledger"):
        list(selected_populations(population, study, refs))
