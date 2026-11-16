from dataclasses import replace
import ast
import csv
import importlib
from pathlib import Path
import tarfile

import pytest
import yaml
from PIL import Image, ImageChops

from mdlm import reproduce
from diffusion_toxicity_smc_experiment import exploratory_population, pipeline
from diffusion_toxicity_smc_experiment.accuracy_budget import assigned_runs, study_configs
from diffusion_toxicity_smc_experiment.crude import CrudeData, load_crude, sample_batch
from diffusion_toxicity_smc_experiment.paper_analysis import checkpoint_log_estimates, comparison_rows
from diffusion_toxicity_smc_experiment.paper_experiment import (
    Run, load_settings, plan_runs, resolved_config, run_signature, saved_records,
)
from diffusion_toxicity_smc_experiment.paper_reuse import approved_sources, audit_archive, calibration_for
from diffusion_toxicity_smc_experiment.storage import atomic_write_json, output_lock, read_json


MDLM = Path(__file__).resolve().parents[1]
PAPER = MDLM / "configs/paper.yaml"
FROZEN = MDLM / "reproduction/frozen/paper_final_scaling.tar.gz"


def test_paper_entry_point_has_no_campaign_driver_dependencies():
    package = MDLM / "src/diffusion_toxicity_smc_experiment"
    pending = [MDLM / "reproduce.py"]
    visited = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.level == 1 or node.module.startswith(package.name + "."):
                    pending.append(package / (node.module.rsplit(".", 1)[-1] + ".py"))
    legacy = {"accuracy_budget", "camera_ready_gpu", "crude_tail_reference", "exploratory",
              "low_budget", "paper_final_scaling", "particle_scaling", "same_run_reuse"}
    assert not ({path.stem for path in visited} & legacy)


def test_paper_orders_384_runs_and_preserves_every_overlapping_numerical_seed():
    settings = load_settings(PAPER)
    plan = plan_runs(settings)
    by_rank = {(r.particles, r.replicate): r.seed for r in plan}
    assert len(by_rank) == len(set(by_rank.values())) == 384
    assert {r.particles for r in plan} == set(range(32, 513, 32))
    legacy, configs = study_configs(PAPER.parent / "legacy" / "mdlm_owt_paper_final_scaling.yaml")
    overlap = [(cfg["smc"]["num_particles"], i // len(configs), seed)
               for i, (cfg, seed) in enumerate(assigned_runs(legacy, configs, 0, 1))
               if cfg["smc"]["num_particles"] in settings["grid"]["seed_start"]]
    assert len(overlap) == 264
    assert all(by_rank[n, rank] == seed for n, rank, seed in overlap)
    legacy_configs = {c["smc"]["num_particles"]: c for c in configs}
    for run in plan:
        if run.particles in legacy_configs:
            old = run_signature(legacy_configs[run.particles], run, legacy["recorded_thresholds"])
            new = run_signature(resolved_config(settings, run.particles), run, settings["ladder"]["recorded_thresholds"])
            assert old == new
    settings["grid"]["seed_start"][64] = settings["grid"]["seed_start"][32]
    with pytest.raises(ValueError, match="distinct valid seed"):
        plan_runs(settings)


def test_duplicate_writer_cannot_enter_output_root(tmp_path):
    with output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already writing"):
            with output_lock(tmp_path):
                pytest.fail("two writers acquired the same lock")
    with output_lock(tmp_path):
        pass


def test_same_seed_replays_frozen_ladder(tmp_path, monkeypatch):
    settings = load_settings(PAPER, smoke=True)
    run = plan_runs(settings)[0]
    config = resolved_config(settings, run.particles)
    study = {"phi": config["smc"]["phi"], "record_population_checkpoints": True,
             "thresholds": settings["ladder"]["recorded_thresholds"]}
    package = tmp_path / "frozen_paper_sampler"
    package.mkdir()
    with tarfile.open(FROZEN) as bundle:
        for member in bundle.getmembers():
            if member.name.startswith("src/diffusion_toxicity_smc_experiment/") and member.name.endswith(".py"):
                (package / Path(member.name).name).write_bytes(bundle.extractfile(member).read())
    monkeypatch.syspath_prepend(str(tmp_path))
    frozen = importlib.import_module("frozen_paper_sampler.exploratory_population")
    before = frozen.population(reproduce.smoke_context(config), config, study, run.seed)
    after = exploratory_population.population(reproduce.smoke_context(config), config, study, run.seed)
    for key in ("particles", "corrections", "level_diagnostics", "mutation_diagnostics", "genealogy"):
        assert before[key] == after[key], key
    for old, new in zip(before["population_checkpoints"], after["population_checkpoints"], strict=True):
        assert all(new[key] == value for key, value in old.items() if key != "runtime_seconds_cumulative")

    frozen_crude = importlib.import_module("frozen_paper_sampler.camera_ready_gpu").crude
    args = (234, study["thresholds"])
    original = frozen_crude(reproduce.smoke_context(config), *args, count=8)
    current = sample_batch(reproduce.smoke_context(config), *args, count=8)
    assert all(current[key] == value for key, value in original.items() if key != "runtime_seconds")


def test_analysis_reproduces_saved_extreme_tail_populations():
    from diffusion_toxicity_smc_experiment.particle_scaling import validated_pool

    root = MDLM / "artifacts/current"
    campaign = root / "camera_ready_extreme_tail_scaling"
    files = sorted(campaign.glob("workers/worker_*/populations/*.json"))
    if not files or not (root / "camera_ready_gradual_budget/manifest.json").is_file():
        pytest.skip("Requires the saved extreme-tail populations and gradual-budget MC pool")
    old, configs = study_configs(PAPER.parent / "legacy" / "mdlm_owt_extreme_tail_scaling.yaml")
    settings = load_settings(PAPER)
    settings["grid"]["replicates"] = old["independent_runs"]
    settings["ladder"]["recorded_thresholds"] = old["thresholds"]
    settings["ladder"]["observe"] = [{"threshold": q, "tilt": tilt, "sweep": "last"}
                                    for q, tilts in old["checkpoint_tilts"].items() for tilt in tilts]
    pool, _ = validated_pool(root, old)
    reference = old["reference"]
    crude = CrudeData(pool, reference["samples"][0], dict(zip(old["thresholds"], reference["discoveries"])), [], "fixture")
    by_seed = {seed: Run(config["smc"]["num_particles"], rank, seed)
               for rank, (config, seed) in enumerate(assigned_runs(old, configs, 0, 1))}
    records = {}
    for file in files:
        population = read_json(file)
        _, timing = calibration_for(campaign, population)
        records[by_seed[population["summary"]["seed"]]] = {"population": population, "timing": {
            **timing, "origin": "archive", "group": "archive",
        }}
    with (campaign / "runs.csv").open(newline="", encoding="utf-8") as file:
        expected = {(int(row["seed"]), float(row["lambda"])): row for row in csv.DictReader(file)}
    actual = list(comparison_rows(settings, records, crude))
    assert len(actual) == len(expected) == 216
    for row in actual:
        saved = expected[row["seed"], row["lambda"]]
        for key in ("smc", "snis", "crude", "event_particles", "crude_hits", "crude_n", "crude_pool_start", "smc_seconds"):
            assert row[key] == pytest.approx(float(saved[key]), rel=1e-12, abs=0), (row["seed"], row["lambda"], key)


@pytest.mark.gpu
def test_pinned_models_on_gpu(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("A CUDA allocation and the pinned, cached models are required")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    settings = load_settings(PAPER)
    settings["grid"].update(replicates=2, seed_start={2: settings["grid"]["seed_start"][32]})
    settings["ladder"].update(levels=settings["ladder"]["levels"][:3],
                             observe=[{"threshold": 1e-6, "tilt": .5, "sweep": "last"}],
                             recorded_thresholds=[1e-6])
    mc = settings["mc"]
    mc["cohorts"] = [{**mc["cohorts"][1], "samples": 128}]
    mc["batch_size"] = 16
    mc["calibration"].update(paths=8, warmup_paths=4, repetitions=2)
    settings["figures"].update(usetex=False, bootstrap_resamples=100)
    settings["model"]["runtime"]["device"] = "cuda"
    config = tmp_path / "integration.yaml"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    output = tmp_path / "real_model"
    args = ["--config", str(config), "--output", str(output)]

    reproduce.main(args)

    plan = plan_runs(settings)
    records = saved_records(settings, output, plan)
    assert len(records) == 2
    assert all(record["timing"]["hardware"]["device"].startswith("cuda") for record in records.values())
    assert all(len(record["population"]["population_checkpoints"]) == 9 for record in records.values())
    with (output / "runs.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [int(row["seed"]) for row in rows] == [run.seed for run in plan]
    assert all(float(row["p_reference"]) > 0 and int(row["crude_n"]) > 0 for row in rows)
    assert len(list((output / "figures").rglob("*.png"))) == 2 * len(settings["figures"]["metrics"])


@pytest.fixture
def completed(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    settings = load_settings(PAPER, smoke=True)
    settings["ladder"]["levels"] = [0.0, 0.2, 0.7, 1.3]
    settings["ladder"]["observe"][0]["tilt"] = 0.2
    config = tmp_path / "tiny.yaml"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    output = tmp_path / "experiment"
    args = ["--config", str(config), "--output", str(output)]
    monkeypatch.setattr(pipeline, "Context", lambda config, **_: reproduce.smoke_context(config))
    return settings, output, args


def test_resume_analysis_and_saved_checkpoint_parity(completed, monkeypatch):
    settings, output, args = completed
    real_population = exploratory_population.population
    calls = []

    def interrupt(context, config, study, seed):
        if not study["record_population_checkpoints"]:
            return real_population(context, config, study, seed)
        if calls:
            raise InterruptedError("test interruption")
        calls.append(seed)
        return real_population(context, config, study, seed)

    monkeypatch.setattr(exploratory_population, "population", interrupt)
    with pytest.raises(InterruptedError):
        reproduce.main(args)
    original, = (output / "runs").glob("*.json")
    first_bytes = original.read_bytes()

    def observe(context, config, study, seed):
        if study["record_population_checkpoints"]:
            calls.append(seed)
        return real_population(context, config, study, seed)

    monkeypatch.setattr(exploratory_population, "population", observe)
    reproduce.main([*args, "--mode", "resume"])
    plan = plan_runs(settings)
    assert calls == [r.seed for r in plan]
    assert original.read_bytes() == first_bytes
    rows_csv = (output / "runs.csv").read_bytes()

    def unexpected_model(*args, **kwargs):
        pytest.fail("analysis attempted to load a model")

    monkeypatch.setattr(pipeline, "Context", unexpected_model)
    reproduce.main([*args, "--mode", "crude", "--crude-root", str(output / "crude")])
    assert read_json(output / "reference.json")["samples"] == 1024
    reproduce.main([*args, "--mode", "analysis"])
    assert (output / "runs.csv").read_bytes() == rows_csv
    assert len(list((output / "figures").rglob("*.png"))) == 2 * len(settings["figures"]["metrics"])
    plain_root = output / "figures/without_captions"
    for path in plain_root.rglob("*.png"):
        captioned = output / "figures/with_captions" / path.relative_to(plain_root)
        with Image.open(path) as plain, Image.open(captioned) as caption:
            assert ImageChops.difference(plain.convert("RGB"), caption.crop((0, 0, *plain.size))).getbbox() is None
    records = saved_records(settings, output, plan)
    crude = load_crude(settings, plan, output / "crude", resolved_config(settings, plan[0].particles))
    assert crude.samples == 1024
    rows = list(comparison_rows(settings, records, crude))
    assert len(rows) == len(plan)
    for r in rows:
        run = next(run for run in plan if run.seed == r["seed"])
        checkpoint = records[run]["population"]["population_checkpoints"][1]
        assert r["smc_seconds"] == checkpoint["runtime_seconds_cumulative"]
        smc, snis, hits = checkpoint_log_estimates(checkpoint, r["threshold"])
        assert (r["log_smc"], r["log_snis"], r["event_particles"]) == (smc, snis, hits)
        assert r["crude"] == r["crude_hits"] / r["crude_n"]

    duplicate = output / "runs/duplicate.json"
    duplicate.write_bytes(first_bytes)
    with pytest.raises(ValueError, match="Unexpected run artifacts"):
        reproduce.main([*args, "--mode", "analysis"])


def test_archive_checks_source_seeds_and_actual_population(completed, tmp_path):
    settings, output, args = completed
    reproduce.main(args)
    plan = plan_runs(settings)
    _, later = approved_sources(settings, FROZEN)
    archive = tmp_path / "historical"
    source = archive / "code/src/diffusion_toxicity_smc_experiment"
    source.mkdir(parents=True)
    with tarfile.open(FROZEN) as bundle:
        for name in later:
            (source / name).write_bytes(bundle.extractfile(f"src/diffusion_toxicity_smc_experiment/{name}").read())
    worker = archive / "workers/worker_0"
    timing_device = {"hostname": "fixture", "workers_in_allocation": 1}
    configs = [resolved_config(settings, n) for n in sorted(settings["grid"]["seed_start"])]
    archived_seeds = [r.seed for r in plan]
    archived_seeds[1] += 1
    atomic_write_json(worker / "manifest.json", {"study": {"paper_final_scaling": True, "independent_runs": 2,
                      "recorded_thresholds": settings["ladder"]["recorded_thresholds"]}, "timing_device": timing_device,
                      "resolved_configs": configs, "source_SHA256": later, "planned_seeds": archived_seeds})
    atomic_write_json(worker / "calibration_original.json", {"timing_device": timing_device, "seconds_per_path": .2,
                      "excluded_from_estimates_and_measured_run_times": True})
    for i in (0, 1, 2):
        run = plan[i]
        p = read_json(output / "runs" / run.filename)["population"]
        p["timing_device"] = timing_device
        if i == 2:
            p["summary"]["seed"] += 1
        file = replace(run, seed=archived_seeds[i]).filename
        atomic_write_json(worker / "populations" / file, p)
    report, compatible = audit_archive(settings, plan, archive, FROZEN)
    assert report["counts"] == dict(compatible=1, incompatible=2, missing=1, requires_recomputation=3)
    assert list(compatible) == [plan[0]]
    assert compatible[plan[0]]["timing"]["origin"] == "archive"
    assert compatible[plan[0]]["timing"]["seconds_per_path"] == .2
    assert "numerical seed" in report["runs"][1]["reason"]
    assert "Population identity" in report["runs"][2]["reason"]
    (source / "smc.py").write_text("changed", encoding="utf-8")
    report, compatible = audit_archive(settings, plan, archive, FROZEN)
    assert not compatible and "hashes differ" in report["runs"][0]["reason"]
