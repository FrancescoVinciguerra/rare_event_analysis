from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _bash_executable() -> str:
    configured = os.environ.get("BASH_EXE")
    candidates = [
        configured,
        r"C:\Program Files\Git\bin\bash.exe",
        shutil.which("bash"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    pytest.skip("Bash is unavailable")


def _dry_run(stage: str) -> str:
    environment = os.environ.copy()
    environment.update(
        {
            "AVON_ACCOUNT": "stats",
            "AVON_PARTITION": "gpu-devel",
            "AVON_GPU_GRES": "gpu:quadro_rtx_6000:1",
            "AVON_MODULES": "GCCcore/13.2.0 Python/3.11.5",
        }
    )
    result = subprocess.run(
        [
            _bash_executable(),
            str(EXPERIMENT_ROOT / "scripts" / "submit_avon_first_experiment.sh"),
            stage,
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _serious_dry_run(stage: str) -> str:
    environment = os.environ.copy()
    environment.update(
        {
            "AVON_ACCOUNT": "stats",
            "AVON_QOS": "normal",
            "AVON_GPU_PARTITION": "gpu",
            "AVON_GPU_DEVEL_PARTITION": "gpu-devel",
            "AVON_CPU_PARTITION": "devel",
            "AVON_GPU_GRES": "gpu:quadro_rtx_6000:1",
            "AVON_MODULES": "GCCcore/13.2.0 Python/3.11.5",
        }
    )
    result = subprocess.run(
        [
            _bash_executable(),
            str(EXPERIMENT_ROOT / "scripts" / "submit_avon_serious_comparison.sh"),
            stage,
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _submission_probability_dry_run(stage: str) -> str:
    environment = os.environ.copy()
    environment.update(
        {
            "AVON_ACCOUNT": "stats",
            "AVON_QOS": "normal",
            "AVON_GPU_PARTITION": "gpu",
            "AVON_CPU_PARTITION": "devel",
            "AVON_GPU_GRES": "gpu:quadro_rtx_6000:1",
            "AVON_MODULES": "GCCcore/13.2.0 Python/3.11.5",
        }
    )
    result = subprocess.run(
        [
            _bash_executable(),
            str(
                EXPERIMENT_ROOT
                / "scripts"
                / "submit_avon_submission_probability.sh"
            ),
            stage,
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_all_avon_shell_scripts_pass_bash_syntax() -> None:
    scripts = sorted((EXPERIMENT_ROOT / "scripts").glob("*.sh"))
    assert scripts
    subprocess.run(
        [_bash_executable(), "-n", *[str(path) for path in scripts]],
        cwd=REPOSITORY_ROOT,
        check=True,
    )


def test_avon_runtime_fails_closed_on_shadowed_experiment_package() -> None:
    common = (
        EXPERIMENT_ROOT / "scripts" / "avon_job_common.sh"
    ).read_text(encoding="utf-8")
    assert 'export PYTHONSAFEPATH=1' in common
    assert 'export PYTHONPATH="${EXPERIMENT_DIR}/src:${PROJECT_DIR}/shared:${PROJECT_DIR}"' in common
    assert 'expected a module below' in common
    assert 'avon_assert_experiment_package' in common


def test_calibration_report_dry_run_is_cpu_only() -> None:
    output = _dry_run("calibration-report")
    assert "partition=devel" in output
    assert "gres=none" in output
    assert "cpus_per_task=2" in output
    assert "memory=4G" in output
    command = next(line for line in output.splitlines() if line.startswith("command="))
    assert "--partition=devel" in command
    assert "--cpus-per-task=2" in command
    assert "--mem=4G" in command
    assert "--time=00:10:00" in command
    assert "--gres" not in command


def test_direct_256_dry_run_retains_gpu_resources() -> None:
    output = _dry_run("direct-256")
    assert "partition=gpu-devel" in output
    assert "gres=gpu:quadro_rtx_6000:1" in output
    assert "cpus_per_task=4" in output
    assert "memory=16G" in output
    command = next(line for line in output.splitlines() if line.startswith("command="))
    assert "--gres=gpu:quadro_rtx_6000:1" in command
    assert "--time=00:45:00" in command


def test_progressive_calibration_dry_run_submits_only_the_smc_job() -> None:
    output = _dry_run("progressive-calibration")
    assert "partition=gpu-devel" in output
    assert "gres=gpu:quadro_rtx_6000:1" in output
    assert "logs/mdlm-lambda-progressive-%j" in output
    command = next(line for line in output.splitlines() if line.startswith("command="))
    assert "slurm_run_smc.sh" in command
    assert "--time=01:00:00" in command
    assert "slurm_generate_direct.sh" not in command
    assert "slurm_score_direct.sh" not in command


def test_100_token_estimation_dry_run_submits_only_one_smc_job() -> None:
    output = _dry_run("estimation-100tok")
    assert "partition=gpu-devel" in output
    assert "gres=gpu:quadro_rtx_6000:1" in output
    assert "memory=20G" in output
    assert "logs/mdlm-estimation-100tok-%j" in output
    command = next(line for line in output.splitlines() if line.startswith("command="))
    assert "slurm_run_smc.sh" in command
    assert "--time=01:00:00" in command
    assert "slurm_generate_direct.sh" not in command
    assert "slurm_score_direct.sh" not in command


def test_trajectory_mh_estimation_dry_run_uses_only_the_generic_smc_job() -> None:
    output = _dry_run("estimation-100tok-trajectory-mh")
    assert "partition=gpu-devel" in output
    assert "gres=gpu:quadro_rtx_6000:1" in output
    assert "memory=20G" in output
    assert "logs/mdlm-estimation-100tok-trajectory-mh-%j" in output
    command = next(line for line in output.splitlines() if line.startswith("command="))
    assert "slurm_run_smc.sh" in command
    assert "--time=01:00:00" in command
    assert "slurm_generate_direct.sh" not in command
    assert "slurm_score_direct.sh" not in command


def test_matched_fk_trajectory_dry_run_uses_only_one_smc_job() -> None:
    output = _dry_run("estimation-100tok-trajectory-mh-matched-fk")
    assert "partition=gpu-devel" in output
    assert "gres=gpu:quadro_rtx_6000:1" in output
    assert "memory=20G" in output
    assert "logs/mdlm-estimation-100tok-trajectory-mh-matched-fk-%j" in output
    command = next(line for line in output.splitlines() if line.startswith("command="))
    assert "slurm_run_smc.sh" in command
    assert "--time=01:00:00" in command
    assert "slurm_generate_direct.sh" not in command
    assert "slurm_score_direct.sh" not in command


@pytest.mark.parametrize(
    ("stage", "partition", "gres", "memory", "walltime", "script"),
    [
        ("method-validation", "gpu-devel", "gpu:quadro_rtx_6000:1", "20G", "01:00:00", "slurm_avon_serious_smc.sh"),
        ("direct-reference", "gpu", "gpu:quadro_rtx_6000:1", "16G", "12:00:00", "slurm_generate_direct.sh"),
        ("score-reference", "gpu", "gpu:quadro_rtx_6000:1", "16G", "04:00:00", "slurm_score_direct.sh"),
        ("smc-grid", "gpu", "gpu:quadro_rtx_6000:1", "20G", "2-00:00:00", "slurm_avon_serious_smc.sh"),
        ("report", "devel", "none", "4G", "00:30:00", "slurm_avon_serious_report.sh"),
    ],
)
def test_serious_comparison_submits_one_stage_at_a_time(
    stage: str,
    partition: str,
    gres: str,
    memory: str,
    walltime: str,
    script: str,
) -> None:
    output = _serious_dry_run(stage)
    assert f"partition={partition}" in output
    assert f"gres={gres}" in output
    assert f"memory={memory}" in output
    assert f"walltime={walltime}" in output
    assert "Dry run only: no job submitted." in output
    command = next(line for line in output.splitlines() if line.startswith("command="))
    assert script in command
    assert command.count("sbatch") == 1
    if stage == "direct-reference":
        assert "--array=0" in command
        assert "mdlm-direct32-200000" in output
    if stage == "method-validation":
        assert "mdlm_owt_unconditional_three_smc_gpu_validation.yaml" in output
    if stage == "report":
        assert "--gres" not in command


@pytest.mark.parametrize(
    ("stage", "partition", "gres", "memory", "walltime", "script"),
    [
        (
            "pilot",
            "devel",
            "none",
            "4G",
            "00:20:00",
            "slurm_avon_submission_probability_cpu.sh",
        ),
        (
            "long-smc",
            "gpu",
            "gpu:quadro_rtx_6000:1",
            "20G",
            "2-00:00:00",
            "slurm_avon_submission_long_smc.sh",
        ),
        (
            "report",
            "devel",
            "none",
            "4G",
            "00:30:00",
            "slurm_avon_submission_probability_cpu.sh",
        ),
    ],
)
def test_submission_probability_submits_one_stage_at_a_time(
    stage: str,
    partition: str,
    gres: str,
    memory: str,
    walltime: str,
    script: str,
) -> None:
    output = _submission_probability_dry_run(stage)
    assert f"partition={partition}" in output
    assert f"gres={gres}" in output
    assert f"memory={memory}" in output
    assert f"walltime={walltime}" in output
    assert "Dry run only: no job submitted." in output
    command = next(line for line in output.splitlines() if line.startswith("command="))
    assert script in command
    assert command.count("sbatch") == 1
    assert "--dependency" not in command
    if stage in {"pilot", "report"}:
        assert "--gres" not in command
