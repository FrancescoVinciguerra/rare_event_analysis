from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from rea_llms.official_direct_data import (
    analyze_direct_ari,
    archive_manifest,
    boxplot_statistics,
    classify_ari_representation,
    ensure_raw_max_available,
    extract_member_safely,
    identify_direct_ari_member,
    rule_of_three_upper_bound,
    tail_statistics,
    validate_checksum,
    weighted_quantile,
    wilson_interval,
    write_json_deterministic,
)


def _write_zip(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, contents in members.items():
            archive.writestr(name, contents)


def test_checksum_validation_accepts_exact_digest_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"official data")
    assert validate_checksum(path, "3e77057fdab0d764db0b62814e6cc13d") == (
        "3e77057fdab0d764db0b62814e6cc13d"
    )
    with pytest.raises(ValueError, match="MD5 mismatch"):
        validate_checksum(path, "0" * 32)


def test_safe_zip_extraction_rejects_traversal_before_writing(tmp_path: Path) -> None:
    safe_archive = tmp_path / "safe.zip"
    _write_zip(safe_archive, {"ari_unbiased_samples.json": "[1, 2, 3]"})
    manifest = archive_manifest(safe_archive)
    member = identify_direct_ari_member([row["path"] for row in manifest])
    extracted = extract_member_safely(safe_archive, member, tmp_path / "out")
    assert json.loads(extracted.read_text(encoding="utf-8")) == [1, 2, 3]

    unsafe_archive = tmp_path / "unsafe.zip"
    _write_zip(
        unsafe_archive,
        {
            "ari_unbiased_samples.json": "[1]",
            "../outside.txt": "must not be written",
        },
    )
    with pytest.raises(ValueError, match="Unsafe ZIP entries"):
        extract_member_safely(
            unsafe_archive,
            "ari_unbiased_samples.json",
            tmp_path / "unsafe-out",
        )
    assert not (tmp_path / "outside.txt").exists()


def test_direct_ari_member_identification_is_exact_and_unambiguous() -> None:
    assert identify_direct_ari_member(
        ["ari_trajectories.json", "ari_unbiased_samples.json"]
    ) == "ari_unbiased_samples.json"
    with pytest.raises(ValueError, match="exactly one"):
        identify_direct_ari_member(["ari_trajectories.json"])
    with pytest.raises(ValueError, match="exactly one"):
        identify_direct_ari_member(
            ["ari_unbiased_samples.json", "nested/ari_unbiased_samples.json"]
        )


def test_tail_counts_and_probabilities_on_toy_data() -> None:
    values = np.asarray([0.0, 2.0, 2.0, 5.0], dtype=np.float64)
    rows = tail_statistics(values, [2.0, 6.0])
    assert rows[0]["count"] == 3
    assert rows[0]["empirical_probability"] == pytest.approx(0.75)
    assert rows[0]["rule_of_three_upper"] is None
    assert rows[1]["count"] == 0
    assert rows[1]["empirical_probability"] == 0.0
    assert rows[1]["rule_of_three_upper"] == pytest.approx(0.75)


def test_wilson_interval_matches_known_cases() -> None:
    lower, upper = wilson_interval(50, 100)
    assert lower == pytest.approx(0.40383153, abs=1e-8)
    assert upper == pytest.approx(0.59616847, abs=1e-8)
    zero_lower, zero_upper = wilson_interval(0, 100)
    assert zero_lower == pytest.approx(0.0, abs=1e-15)
    assert zero_upper == pytest.approx(0.03699350, abs=1e-8)


def test_rule_of_three_is_reported_only_for_zero_count() -> None:
    assert rule_of_three_upper_bound(0, 100) == pytest.approx(0.03)
    assert rule_of_three_upper_bound(1, 100) is None


def test_quantiles_and_boxplot_statistics_are_deterministic() -> None:
    values = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    box = boxplot_statistics(values)
    assert box == {
        "q1": 0.75,
        "median": 1.5,
        "q3": 2.25,
        "iqr": 1.5,
        "whisker_low": 0.0,
        "whisker_high": 3.0,
        "outlier_count": 0,
    }
    weighted = weighted_quantile(values[[0, 3]], [0.5, 0.5], [0.25, 0.5, 0.75])
    assert weighted.tolist() == pytest.approx([0.0, 1.5, 3.0])


def test_raw_max_request_fails_for_capped_or_ambiguous_data() -> None:
    capped = np.asarray([1.0, 5.0, 15.0, 15.0])
    representation = classify_ari_representation(capped)
    assert representation == "capped_or_truncated"
    with pytest.raises(ValueError, match="raw maximum"):
        ensure_raw_max_available(representation)
    ensure_raw_max_available(classify_ari_representation(np.asarray([1.0, 16.0])))


def test_summary_json_output_is_byte_deterministic(tmp_path: Path) -> None:
    statistics = analyze_direct_ari(
        np.asarray([0.0, 1.0, 2.0, 20.0]),
        [2.0, 30.0],
    )
    payload = {"statistics": statistics, "source": {"doi": "test"}}
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_json_deterministic(first, payload)
    write_json_deterministic(second, payload)
    assert first.read_bytes() == second.read_bytes()


def test_official_data_colab_notebook_is_valid_and_analysis_only() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "REA_Official_Direct_Data_Quickstart.ipynb"
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells: list[str] = []
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
    assert "rea_llms.official_direct_data" in joined
    for parameter in (
        "DATA_ROOT",
        "DOWNLOAD_IF_MISSING",
        "VERIFY_CHECKSUM",
        "TAIL_THRESHOLDS",
        "OUTPUT_ROOT",
    ):
        assert parameter in joined
    assert "run_suffix_direct_mh_smc" not in joined
    assert "official_annealed_tps" not in joined
    assert "AutoModelForCausalLM" not in joined


def test_official_demo_decoding_and_ari_input_are_audited_from_source() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "external"
        / "rea_of_llms_official"
        / "sampling_examples.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "tokenizer.decode(output_tokens[0], skip_special_tokens=True)" in source
    assert "completion_text = output_text[len(prompt):]" in source
    assert "automated_readability_index(prompt + c)" in source
    assert "clean_up_tokenization_spaces" not in source

    summary = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "results"
            / "reference_direct_sampling_summary.json"
        ).read_text(encoding="utf-8")
    )
    official = summary["official_configuration"]
    assert official["ari_input_expression"] == (
        "prompt + decoded_full_text[len(prompt):]"
    )
    assert official["decode_clean_up_tokenization_spaces"].startswith(
        "not passed explicitly"
    )
    assert official["source_cell_indices_zero_based"] == {
        "direct_sampling": 5,
        "generate_tokens_and_compute_ari": 3,
    }
    assert official["source_fragments"]["prompt_slice"] in source
    assert summary["comparability"]["strict_observable_match_verified"] is False
