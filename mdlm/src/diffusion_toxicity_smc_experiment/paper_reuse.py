"""Audit the original seed ranks, frozen code and costs before importing a run."""
from dataclasses import asdict
import hashlib
import json
import tarfile

from .paper_experiment import make_record, resolved_config, run_signature
from .storage import file_hash, read_json


def check_archive_crude(settings, crude):
    expected_count = sum(c["samples"] for c in settings["mc"]["cohorts"])
    hits = [crude.hits[q] for q in sorted(crude.hits)]
    raw = crude.matched_scores.astype("<f8", copy=False).tobytes()
    if (crude.samples != expected_count or hits != settings["archive"]["reference_hits"]
            or hashlib.sha256(raw).hexdigest() != settings["archive"]["matched_scores_sha256"]):
        raise ValueError("Archived wall-clock comparisons require their original reference counts and ordered MC pool")


def calibration_for(root,population):
    device=population['timing_device']
    matches=[]
    for path in sorted(root.glob('workers/worker_*/calibration_*.json')):
        data=json.loads(path.read_text(encoding='utf-8'))
        physical=('hostname','SLURM_STEP_GPUS','CUDA_VISIBLE_DEVICES','SLURM_CPUS_PER_TASK')
        if all(data['timing_device'].get(k)==device.get(k) for k in physical):
            if not data['excluded_from_estimates_and_measured_run_times']:
                raise ValueError('Timing calibration was not independent')
            matches.append((path,data))
    if not matches:raise ValueError('No independent calibration for this GPU and CPU allocation')
    return matches[0]


def approved_sources(settings, bundle):
    if file_hash(bundle) != settings["archive"]["source_bundle_sha256"]:
        raise ValueError("The supplied frozen source bundle differs from the approved bundle")
    with tarfile.open(bundle, "r:gz") as archive:
        index = json.load(archive.extractfile("paper_final_sources.json"))
        original = index["original_sampler_sha256"]
        later = {}
        for name in original:
            path = f"src/diffusion_toxicity_smc_experiment/{name}"
            later[name] = hashlib.sha256(archive.extractfile(path).read()).hexdigest()
            if later[name] != index["bundle_sha256"][path]:
                raise ValueError(f"Frozen source does not match the bundle index for {name}")
            if name not in {"accuracy_budget.py", "same_run_reuse.py"} and later[name] != original[name]:
                raise ValueError(f"The historical populations were produced with different scientific code: {name}")
        return original, later


def archived_runs(root, original, later):
    reuse_path = root / "reuse_manifest.json"
    reuse = read_json(reuse_path) if reuse_path.is_file() else {"workers": {}, "files_sha256": {}}
    candidates = {}
    manifests = sorted(root.glob("workers/worker_*/manifest.json"))
    if not manifests:
        raise ValueError(f"Archive has no worker manifests: {root}")
    for path in manifests:
        manifest = read_json(path)
        worker = int(path.parent.name.removeprefix("worker_"))
        workers = manifest["timing_device"]["workers_in_allocation"]
        study = manifest["study"]
        expected_sources = later if study.get("paper_final_scaling", False) else original
        source_error = None
        if manifest["source_SHA256"] != expected_sources:
            source_error = "Worker source hashes differ from the approved frozen code"
        else:
            source_root = root / "code/src/diffusion_toxicity_smc_experiment"
            if any(not (source_root / name).is_file() or file_hash(source_root / name) != digest
                   for name, digest in expected_sources.items()):
                source_error = "Archived source files are missing or their hashes differ"
        configs = manifest["resolved_configs"]
        keys = ((rank, cfg) for rank in range(worker, study["independent_runs"], workers) for cfg in configs)
        expected_files = set()
        for (rank, cfg), seed in zip(keys, manifest["planned_seeds"], strict=True):
            case = cfg["_exploration"]
            key = (cfg["smc"]["num_particles"], rank)
            if key in candidates:
                raise ValueError(f"Multiple archived manifests claim N/replicate {key}")
            file = path.parent / "populations" / f"{case['id']}_seed{seed}.json"
            expected_files.add(file)
            candidates[key] = (cfg, seed, file, study, source_error)
        extra = set((path.parent / "populations").glob("*.json")) - expected_files
        if extra:
            raise ValueError(f"Unplanned archived population files: {sorted(extra)}")
    return candidates, reuse


def check_archive_run(settings, run, candidate, root, reuse, original):
    config, seed, file, study, source_error = candidate
    if seed != run.seed:
        raise ValueError(f"Archived numerical seed {seed} differs from planned seed {run.seed}")
    recorded = (study["recorded_thresholds"] if study.get("paper_final_scaling", False) else study["thresholds"])
    expected = resolved_config(settings, run.particles)
    if (run_signature(config, run, recorded) != run_signature(expected, run, settings["ladder"]["recorded_thresholds"])
            or config["smc"]["tilted_sampler"]["implementation_version"] !=
               expected["smc"]["tilted_sampler"]["implementation_version"]):
        raise ValueError("Saved schedule, sampler policy, or model differs from the configured run")
    if source_error:
        raise ValueError(source_error)
    relative = file.relative_to(root).as_posix()
    digest = file_hash(file)
    worker = file.parent.parent.name.removeprefix("worker_")
    if relative in reuse["files_sha256"]:
        reused = reuse["workers"][worker]
        if (reused["source_sha256"] != original or reused["populations"][str(seed)] != relative
                or digest != reuse["files_sha256"][relative]):
            raise ValueError("Historical copy differs from its original source manifest or hash")
    population = read_json(file)
    path, calibration = calibration_for(root, population)
    timing = {**calibration, "origin": "archive", "group": "archive_" + settings["archive"]["source_bundle_sha256"][:12],
              "calibration_sha256": file_hash(path)}
    if relative in reuse["files_sha256"]:
        original_calibrations = {digest for filename, digest in reuse["files_sha256"].items()
                                 if filename.rsplit("/", 1)[-1].startswith("calibration_")}
        if timing["calibration_sha256"] not in original_calibrations:
            raise ValueError("Timing calibration differs from the original reused calibration")
    record = make_record(settings, run, population, timing)
    record["archive"] = {"file": relative, "sha256": digest, "seed": seed}
    return record


def audit_archive(settings, runs, root=None, bundle=None):
    if root is None:
        candidates = {}
    else:
        original, later = approved_sources(settings, bundle)
        candidates, reuse = archived_runs(root, original, later)
    rows, compatible = [], {}
    for run in runs:
        candidate = candidates.get((run.particles, run.replicate))
        row = {**asdict(run), "archived_seed": candidate[1] if candidate else None}
        if candidate is None or not candidate[2].is_file():
            reason = ("No archive supplied for verification" if root is None
                      else "No population artifact for this particle count and replicate")
            row.update(status="missing", reason=reason)
        else:
            try:
                compatible[run] = check_archive_run(settings, run, candidate, root, reuse, original)
            except (ValueError, KeyError, AssertionError, OSError) as exc:
                row.update(status="incompatible", reason=str(exc))
            else:
                row.update(status="compatible", reason="Verified frozen code, seed, population and calibration")
        row["requires_recomputation"] = row["status"] != "compatible"
        rows.append(row)
    counts = {status: sum(r["status"] == status for r in rows) for status in ("compatible", "incompatible", "missing")}
    counts["requires_recomputation"] = counts["missing"] + counts["incompatible"]
    planned_keys = {(run.particles, run.replicate) for run in runs}
    outside = [{"particles": n, "replicate": rank, "archived_seed": candidate[1]}
               for (n, rank), candidate in sorted(candidates.items())
               if (n, rank) not in planned_keys]
    return {"archive": root.name if root else None, "counts": counts,
            "runs": rows, "outside_grid": outside}, compatible
