from pathlib import Path

from diffusion_toxicity_smc_experiment.accuracy_budget import assigned_runs, study_configs


def test_paper_seed_plan_preserves_terminal_runs_and_historical_seeds():
    configs_dir = Path(__file__).resolve().parents[1] / "configs/legacy"
    study, configs = study_configs(configs_dir / "mdlm_owt_paper_final_scaling.yaml")
    old, old_configs = study_configs(configs_dir / "mdlm_owt_progressive_tail_particle_scaling.yaml")
    assert study["checkpoint_tilts"] == {.99: [.5], .995: [1.], .999: [2.5], .9995: [6.5]}
    assert study["recorded_thresholds"] == old["thresholds"]
    tasks = assigned_runs(study, configs, 0, 1)
    assert len({seed for _, seed in tasks}) == len(tasks) == 288
    for worker in range(3):
        planned = assigned_runs(study, configs, worker, 3)
        reused = [(c, seed) for c, seed in planned if c["smc"]["num_particles"] in study["reuse_particle_counts"]]
        assert len(planned) - len(reused) == 64
        for (config, seed), (archived, archived_seed) in zip(reused, assigned_runs(old, old_configs, worker, 3), strict=True):
            case = {k: v for k, v in config["_exploration"].items() if k != "seed_start"}
            assert {**config, "_exploration": case} == archived and seed == archived_seed

    for n, campaign in [(128, "same_run_logit"), (256, "extreme_tail_scaling")]:
        historic, historic_configs = study_configs(configs_dir / f"mdlm_owt_{campaign}.yaml")
        historic_seeds = [seed for _, seed in assigned_runs(historic, historic_configs, 0, 1)]
        paper_seeds = [seed for c, seed in tasks if c["smc"]["num_particles"] == n]
        assert len(historic_seeds) == 12
        assert paper_seeds[:12] == historic_seeds
