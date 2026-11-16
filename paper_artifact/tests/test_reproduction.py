import csv
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageChops

import diffusion_toxicity_smc_experiment as package
from diffusion_toxicity_smc_experiment import exploratory_population, pipeline
from diffusion_toxicity_smc_experiment.crude import load_crude, planned_batches
from diffusion_toxicity_smc_experiment.paper_analysis import checkpoint_log_estimates, comparison_rows
from diffusion_toxicity_smc_experiment.paper_experiment import load_settings, plan_runs, resolved_config, saved_records
from diffusion_toxicity_smc_experiment.potentials import ToxicityPotential
from diffusion_toxicity_smc_experiment.storage import output_lock, read_json


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper.yaml"
spec = importlib.util.spec_from_file_location("reproduce", ROOT / "reproduce.py")
assert spec is not None and spec.loader is not None
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)


def test_frozen_files_and_imports():
    assert Path(package.__file__).resolve().is_relative_to(ROOT / "src")
    manifest = read_json(ROOT / "RELEASE.json")
    for name, expected in manifest["files_sha256"].items():
        path = ROOT / name
        assert not path.is_symlink()
        assert sha256(path.read_bytes()).hexdigest() == expected, name


def test_paper_plan(capsys):
    reproduce.main(["--config", str(PAPER), "--mode", "plan"])
    plan = json.loads(capsys.readouterr().out)
    runs = plan["runs"]
    assert plan["planned_runs"] == len(runs) == 384
    assert {(run["particles"], run["replicate"]) for run in runs} == {
        (n, rank) for n in range(32, 513, 32) for rank in range(24)
    }
    assert len({r["seed"] for r in runs}) == 384
    settings = load_settings(PAPER)
    assert settings["ladder"]["levels"] == [step / 2 for step in range(14)]
    assert settings["ladder"]["moves_per_level"] == 4
    assert sum(batch["samples"] for batch in planned_batches(settings)) == 975775


def test_duplicate_writer_cannot_enter_output_root(tmp_path):
    with output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already writing"):
            with output_lock(tmp_path):
                pytest.fail("two writers acquired the same lock")
    with output_lock(tmp_path):
        pass


@pytest.mark.parametrize("name", ["log_probability", "logit"])
def test_finite_native_rollback_mh_balance_and_both_reverse_identities(name):
    scores = np.array([.05, .9, .99927776, .99955579])
    native = np.array([.7, .28, .015, .005])
    potential = ToxicityPotential(name)
    rewards = np.array([potential.threshold_value(score) for score in scores])
    previous_tilt, previous_pi, previous_z = 0., native, 1.
    for tilt in [.5, 1., 2.5]:
        gamma = native * np.exp(tilt * rewards)
        z = gamma.sum()
        pi = gamma / z
        acceptance = np.minimum(1., np.exp(tilt * (rewards[None, :] - rewards[:, None])))
        transition = native[None, :] * acceptance
        transition[np.diag_indices(len(scores))] += 1. - transition.sum(axis=1)
        assert np.allclose(pi[:, None] * transition, (pi[:, None] * transition).T)
        incremental = np.exp((tilt - previous_tilt) * rewards)
        ratio = previous_pi @ incremental
        assert previous_z * ratio == pytest.approx(z)
        assert np.allclose((previous_pi * incremental / ratio) @ transition, pi)
        result = pipeline.estimates(np.log(pi), rewards, math.log(z), scores, .9995, tilt)
        assert result["smc"] == pytest.approx(.005)
        assert result["snis"] == pytest.approx(.005)
        assert result["reverse_mass"] == pytest.approx(1.)
        previous_tilt, previous_pi, previous_z = tilt, pi, z


@pytest.fixture
def small_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    settings = load_settings(PAPER, smoke=True)
    settings["ladder"]["levels"] = [0.0, 0.2, 0.7, 1.3]
    settings["ladder"]["observe"][0]["tilt"] = 0.2
    config = tmp_path / "tiny.yaml"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    output = tmp_path / "experiment"
    args = ["--config", str(config)]
    monkeypatch.setattr(pipeline, "Context", lambda config, **_: reproduce.smoke_context(config))
    return settings, output, args


def test_resume_analysis_and_saved_checkpoint_parity(small_experiment, tmp_path, monkeypatch):
    settings, output, config_args = small_experiment
    args = [*config_args, "--output", str(output)]
    population = exploratory_population.population
    calls = []

    def interrupt(context, config, study, seed):
        if study["record_population_checkpoints"]:
            if calls:
                raise InterruptedError("test interruption")
            calls.append(seed)
        return population(context, config, study, seed)

    monkeypatch.setattr(exploratory_population, "population", interrupt)
    with pytest.raises(InterruptedError):
        reproduce.main(args)
    original, = (output / "runs").glob("*.json")
    first_bytes = original.read_bytes()

    def observe(context, config, study, seed):
        if study["record_population_checkpoints"]:
            calls.append(seed)
        return population(context, config, study, seed)

    monkeypatch.setattr(exploratory_population, "population", observe)
    reproduce.main([*args, "--mode", "resume"])
    plan = plan_runs(settings)
    assert calls == [run.seed for run in plan]
    assert original.read_bytes() == first_bytes

    def unexpected_model(*args, **kwargs):
        pytest.fail("offline mode attempted to load a model")

    monkeypatch.setattr(pipeline, "Context", unexpected_model)
    reproduce.main([*args, "--mode", "crude", "--crude-root", str(output / "crude")])
    assert read_json(output / "reference.json")["samples"] == 1024
    source = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    report = tmp_path / "report"
    analysis_args = [*config_args, "--mode", "analysis", "--data", str(output)]
    reproduce.main([*analysis_args, "--output", str(report)])
    for name in ("runs.csv", "summary.csv", "reference.json"):
        assert (report / name).read_bytes() == source[Path(name)]
    assert {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()} == source
    with pytest.raises(SystemExit, match="2"):
        reproduce.main([*analysis_args, "--output", str(output / "report")])
    with (report / "runs.csv").open(newline="", encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == len(plan)

    assert len(list((report / "figures").rglob("*.png"))) == 2 * len(settings["figures"]["metrics"])
    plain_root = report / "figures/without_captions"
    for path in plain_root.rglob("*.png"):
        captioned = report / "figures/with_captions" / path.relative_to(plain_root)
        with Image.open(path) as plain, Image.open(captioned) as caption:
            assert ImageChops.difference(plain.convert("RGB"), caption.crop((0, 0, *plain.size))).getbbox() is None

    records = saved_records(settings, output, plan)
    crude = load_crude(settings, plan, output / "crude", resolved_config(settings, plan[0].particles))
    assert crude.samples == 1024
    for row in comparison_rows(settings, records, crude):
        run = next(run for run in plan if run.seed == row["seed"])
        checkpoint = records[run]["population"]["population_checkpoints"][1]
        assert row["smc_seconds"] == checkpoint["runtime_seconds_cumulative"]
        smc, snis, hits = checkpoint_log_estimates(checkpoint, row["threshold"])
        assert (row["log_smc"], row["log_snis"], row["event_particles"]) == (smc, snis, hits)
        assert row["crude"] == row["crude_hits"] / row["crude_n"]

    (output / "runs/duplicate.json").write_bytes(first_bytes)
    with pytest.raises(ValueError, match="Unexpected run artifacts"):
        reproduce.main([*analysis_args, "--output", str(report)])
