"""Acquire and analyze the official direct-sampling ARI reference data.

This module never generates language-model samples. It downloads the published
Zenodo archive, verifies it before extraction, extracts only the direct ARI
array, and computes deterministic descriptive and tail statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import stat
import urllib.request
import zlib
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Sequence

import numpy as np


ZENODO_RECORD = "20395631"
ZENODO_DOI = "10.5281/zenodo.20395631"
ARCHIVE_NAME = "paper_data.zip"
ARCHIVE_URL = (
    "https://zenodo.org/api/records/20395631/files/paper_data.zip/content"
)
EXPECTED_ARCHIVE_BYTES = 145_029_767
EXPECTED_MD5 = "83ad4e8f245edf5f45d879ae08450c3e"
DIRECT_ARI_MEMBER = "ari_unbiased_samples.json"
OFFICIAL_REPOSITORY_COMMIT = "9a2ae06323db98496d179cefb294db60a6cd9535"
DEFAULT_THRESHOLDS = (6.0, 8.0, 10.0, 12.0, 15.0)
RAW_ARI_CAP = 15.0
WILSON_Z_95 = 1.959963984540054


def file_md5(path: Path, *, block_size: int = 1024 * 1024) -> str:
    """Return the lowercase MD5 digest of one local file."""

    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checksum(path: Path, expected_md5: str = EXPECTED_MD5) -> str:
    """Validate an archive checksum and return its observed digest."""

    observed = file_md5(path)
    if observed.lower() != expected_md5.lower():
        raise ValueError(
            f"MD5 mismatch for {path}: expected {expected_md5}, observed {observed}"
        )
    return observed


def _unsafe_zip_reasons(info: zipfile.ZipInfo) -> list[str]:
    posix_path = PurePosixPath(info.filename)
    windows_path = PureWindowsPath(info.filename)
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    reasons: list[str] = []
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        reasons.append("absolute path or drive prefix")
    if ".." in posix_path.parts or ".." in windows_path.parts:
        reasons.append("parent-directory traversal")
    if stat.S_ISLNK(unix_mode):
        reasons.append("symbolic link")
    if "\x00" in info.filename:
        reasons.append("NUL byte")
    return reasons


def validate_zip_members(infos: Iterable[zipfile.ZipInfo]) -> None:
    """Reject absolute, traversing, drive-prefixed, or symlink ZIP entries."""

    unsafe = [
        (info.filename, _unsafe_zip_reasons(info))
        for info in infos
        if _unsafe_zip_reasons(info)
    ]
    if unsafe:
        details = "; ".join(f"{name}: {', '.join(reasons)}" for name, reasons in unsafe)
        raise ValueError(f"Unsafe ZIP entries: {details}")


def archive_manifest(archive_path: Path) -> list[dict[str, Any]]:
    """Return the verified central-directory inventory without extracting data."""

    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        validate_zip_members(infos)
        return [
            {
                "path": info.filename,
                "format": "JSON" if info.filename.lower().endswith(".json") else "unknown",
                "compressed_bytes": int(info.compress_size),
                "uncompressed_bytes": int(info.file_size),
                "crc32": f"{info.CRC:08x}",
            }
            for info in infos
        ]


def identify_direct_ari_member(names: Sequence[str]) -> str:
    """Identify the official IID ARI array by its documented archive basename."""

    matches = [
        name
        for name in names
        if PurePosixPath(name).name == DIRECT_ARI_MEMBER
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {DIRECT_ARI_MEMBER!r} archive member; found {matches}"
        )
    return matches[0]


def _crc32_file(path: Path, *, block_size: int = 1024 * 1024) -> int:
    checksum = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xFFFFFFFF


def extract_member_safely(
    archive_path: Path,
    member_name: str,
    output_root: Path,
) -> Path:
    """Extract one validated regular file and reuse an identical existing copy."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        validate_zip_members(infos)
        matches = [info for info in infos if info.filename == member_name]
        if len(matches) != 1 or matches[0].is_dir():
            raise ValueError(f"Archive member is missing or not a regular file: {member_name}")
        info = matches[0]

        relative = Path(*PurePosixPath(member_name).parts)
        destination = output_root / relative
        resolved_root = output_root.resolve()
        resolved_destination = destination.resolve()
        try:
            resolved_destination.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"Extraction target escapes output root: {member_name}") from exc

        if (
            destination.is_file()
            and destination.stat().st_size == info.file_size
            and _crc32_file(destination) == info.CRC
        ):
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with archive.open(info) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            if temporary.stat().st_size != info.file_size:
                raise ValueError(f"Extracted size mismatch for {member_name}")
            if _crc32_file(temporary) != info.CRC:
                raise ValueError(f"Extracted CRC mismatch for {member_name}")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination


def download_archive(url: str, destination: Path) -> Path:
    """Stream an archive to a temporary file and atomically publish it."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "rea-reference-data/1"})
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as target:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header else None
            copied = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                target.write(block)
                copied += len(block)
                if total:
                    print(f"download: {copied / total:6.1%}", end="\r", flush=True)
        print(f"downloaded {copied} bytes{' ' * 20}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def acquire_archive(
    data_root: Path,
    *,
    download_if_missing: bool,
    verify_checksum: bool,
    url: str = ARCHIVE_URL,
    expected_md5: str = EXPECTED_MD5,
) -> Path:
    """Reuse a local archive or download it once, then validate it."""

    archive_path = Path(data_root) / ARCHIVE_NAME
    if not archive_path.is_file():
        if not download_if_missing:
            raise FileNotFoundError(
                f"Missing {archive_path}; enable --download-if-missing to fetch Zenodo data"
            )
        download_archive(url, archive_path)
    if verify_checksum:
        validate_checksum(archive_path, expected_md5)
    return archive_path


def load_numeric_json_array(path: Path) -> np.ndarray:
    """Load one flat JSON numeric array once and validate every observation."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a non-empty JSON list in {path}")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in payload
    ):
        raise ValueError(f"Non-numeric value in {path}")
    values = np.asarray(payload, dtype=np.float64)
    del payload
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"Expected a finite one-dimensional numeric array in {path}")
    return values


def classify_ari_representation(values: np.ndarray, *, cap: float = RAW_ARI_CAP) -> str:
    """Classify whether an ARI array demonstrably retains values above the cap."""

    maximum = float(np.max(values))
    if maximum > cap:
        return "raw"
    if bool(np.any(np.isclose(values, cap, rtol=0.0, atol=1e-12))):
        return "capped_or_truncated"
    return "indeterminate_below_cap"


def ensure_raw_max_available(representation: str) -> None:
    """Fail when a requested raw maximum cannot be recovered from the dataset."""

    if representation != "raw":
        raise ValueError(
            "A raw maximum was requested, but the dataset is capped or does not "
            "contain evidence that values above the cap were retained"
        )


def wilson_interval(
    count: int,
    total: int,
    *,
    z: float = WILSON_Z_95,
) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a binomial proportion."""

    if total <= 0 or count < 0 or count > total:
        raise ValueError("Wilson interval requires 0 <= count <= total and total > 0")
    proportion = count / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def rule_of_three_upper_bound(count: int, total: int) -> float | None:
    """Return ``3/N`` only for a zero-count event."""

    if total <= 0 or count < 0:
        raise ValueError("count must be non-negative and total must be positive")
    return min(1.0, 3.0 / total) if count == 0 else None


def tail_statistics(
    values: np.ndarray,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    """Compute empirical upper tails and 95% Wilson intervals."""

    total = int(values.size)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        count = int(np.count_nonzero(values >= float(threshold)))
        lower, upper = wilson_interval(count, total)
        rows.append(
            {
                "threshold": float(threshold),
                "count": count,
                "empirical_probability": count / total,
                "wilson_95_lower": lower,
                "wilson_95_upper": upper,
                "rule_of_three_upper": rule_of_three_upper_bound(count, total),
            }
        )
    return rows


def weighted_quantile(
    values: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    quantiles: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Compute weighted quantiles from normalized final-particle mass.

    The interpolation positions are centered within each sorted particle's
    weight interval. This function is intended for future weighted SMC boxplots;
    it never treats particle weights as sample multiplicities.
    """

    value_array = np.asarray(values, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    quantile_array = np.asarray(quantiles, dtype=np.float64)
    if value_array.ndim != 1 or weight_array.shape != value_array.shape:
        raise ValueError("values and weights must be one-dimensional and equally sized")
    if value_array.size == 0 or not np.isfinite(value_array).all():
        raise ValueError("values must be non-empty and finite")
    if not np.isfinite(weight_array).all() or np.any(weight_array < 0.0):
        raise ValueError("weights must be finite and non-negative")
    if not np.any(weight_array > 0.0):
        raise ValueError("at least one weight must be positive")
    if np.any((quantile_array < 0.0) | (quantile_array > 1.0)):
        raise ValueError("quantiles must lie in [0, 1]")

    positive = weight_array > 0.0
    order = np.argsort(value_array[positive], kind="stable")
    sorted_values = value_array[positive][order]
    sorted_weights = weight_array[positive][order]
    cumulative = np.cumsum(sorted_weights)
    positions = (cumulative - 0.5 * sorted_weights) / cumulative[-1]
    return np.interp(
        quantile_array,
        positions,
        sorted_values,
        left=sorted_values[0],
        right=sorted_values[-1],
    )


def boxplot_statistics(
    values: Sequence[float] | np.ndarray,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
) -> dict[str, float | int]:
    """Return quartiles and Tukey whiskers for direct or weighted SMC data."""

    value_array = np.asarray(values, dtype=np.float64)
    if value_array.ndim != 1 or value_array.size == 0 or not np.isfinite(value_array).all():
        raise ValueError("values must be a non-empty finite vector")
    if weights is None:
        q1, median, q3 = np.quantile(value_array, [0.25, 0.5, 0.75], method="linear")
    else:
        q1, median, q3 = weighted_quantile(
            value_array,
            weights,
            [0.25, 0.5, 0.75],
        )
    iqr = float(q3 - q1)
    low_fence = float(q1 - 1.5 * iqr)
    high_fence = float(q3 + 1.5 * iqr)
    inlier_mask = (value_array >= low_fence) & (value_array <= high_fence)
    inliers = value_array[inlier_mask]
    return {
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "iqr": iqr,
        "whisker_low": float(np.min(inliers)),
        "whisker_high": float(np.max(inliers)),
        "outlier_count": int(value_array.size - inliers.size),
    }


def _numeric_tree_stats(value: Any) -> dict[str, Any]:
    count = nonfinite = nonnumeric = 0
    minimum = math.inf
    maximum = -math.inf
    types: set[str] = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            types.add(type(item).__name__)
            count += 1
            if math.isfinite(item):
                minimum = min(minimum, float(item))
                maximum = max(maximum, float(item))
            else:
                nonfinite += 1
        else:
            types.add(type(item).__name__)
            nonnumeric += 1
    return {
        "count": count,
        "value_types": sorted(types),
        "min": minimum if count > nonfinite else None,
        "max": maximum if count > nonfinite else None,
        "nonfinite": nonfinite,
        "nonnumeric": nonnumeric,
    }


def _archive_member_role(name: str) -> tuple[str, str]:
    observable = "ari" if name.startswith("ari_") else "log_probability"
    role = "direct_iid" if "unbiased_samples" in name else "tps_markov_chain_states"
    return observable, role


def inspect_json_member(archive_path: Path, member_name: str) -> dict[str, Any]:
    """Inspect one official JSON member without extracting it to disk."""

    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(member_name)
        with archive.open(info) as raw, io.TextIOWrapper(raw, encoding="utf-8") as text:
            payload = json.load(text)
    observable, role = _archive_member_role(PurePosixPath(member_name).name)
    result: dict[str, Any] = {
        "path": member_name,
        "format": "JSON",
        "compressed_bytes": int(info.compress_size),
        "uncompressed_bytes": int(info.file_size),
        "observable": observable,
        "role": role,
        "contains_completions": False,
        "contains_metadata": role == "tps_markov_chain_states",
    }
    if role == "direct_iid":
        result.update(_numeric_tree_stats(payload))
        result["shape"] = [len(payload)]
    else:
        if not isinstance(payload, list) or len(payload) != 3:
            raise ValueError(f"Unexpected trajectory structure in {member_name}")
        samples, biases, steps_per_bias = payload
        result.update(_numeric_tree_stats(samples))
        result.update(
            shape=[
                len(samples),
                sorted({len(group) for group in samples}),
                sorted({len(chain) for group in samples for chain in group}),
            ],
            biases=biases,
            steps_per_bias=int(steps_per_bias),
        )
    return result


def analyze_direct_ari(
    values: np.ndarray,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    """Build the deterministic statistical summary for official direct ARI."""

    quantile_levels = (0.90, 0.95, 0.99, 0.999, 0.9999)
    quantile_values = np.quantile(values, quantile_levels, method="linear")
    representation = classify_ari_representation(values)
    return {
        "count": int(values.size),
        "stored_type": "JSON number",
        "analysis_dtype": str(values.dtype),
        "nonfinite": int(np.count_nonzero(~np.isfinite(values))),
        "nonnumeric": 0,
        "ari_representation": representation,
        "cap_applied_in_file": False if representation == "raw" else None,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values, dtype=np.float64)),
        "standard_deviation_population": float(np.std(values, ddof=0, dtype=np.float64)),
        "quartiles": {
            "q25": float(np.quantile(values, 0.25, method="linear")),
            "q50": float(np.quantile(values, 0.50, method="linear")),
            "q75": float(np.quantile(values, 0.75, method="linear")),
        },
        "quantiles": {
            f"q{level:g}": float(value)
            for level, value in zip(quantile_levels, quantile_values)
        },
        "quantile_method": "numpy linear interpolation",
        "boxplot": boxplot_statistics(values),
        "tails": tail_statistics(values, thresholds),
        "values_equal_15": int(np.count_nonzero(values == RAW_ARI_CAP)),
        "values_above_15": int(np.count_nonzero(values > RAW_ARI_CAP)),
    }


def _official_configuration() -> dict[str, Any]:
    return {
        "model": "roneneldan/TinyStories-8M",
        "model_revision": "not published",
        "tokenizer": "EleutherAI/gpt-neo-125M",
        "tokenizer_revision": "not published",
        "prompt": "Once upon a time, in a big forest, there lived a rhinoc",
        "prompt_repr": "'Once upon a time, in a big forest, there lived a rhinoc'",
        "prompt_sha256": (
            "b4b7f60343cd3d1e24d8f2e312810afdaaa28034b9a194c66865b47483f70f59"
        ),
        "prompt_token_ids": [
            7454, 2402, 257, 640, 11, 287, 257, 1263,
            8222, 11, 612, 5615, 257, 9529, 259, 420,
        ],
        "prompt_token_count": 16,
        "completion_tokens": 100,
        "temperature": 1.0,
        "do_sample": True,
        "top_k": None,
        "top_p": "not explicitly set; Hugging Face default",
        "eos_token_id": None,
        "pad_token_id": None,
        "fixed_length_asserted": True,
        "public_demo_batching": "one completion at a time",
        "paper_generation_implementation": "not published",
        "decode_convention": (
            "decode the full prompt-token plus completion-token sequence with "
            "skip_special_tokens=True, then define completion_text as "
            "decoded_full_text[len(prompt):]"
        ),
        "decode_clean_up_tokenization_spaces": (
            "not passed explicitly; tokenizer/version default"
        ),
        "ari_input_expression": "prompt + decoded_full_text[len(prompt):]",
        "source_notebook": "external/rea_of_llms_official/sampling_examples.ipynb",
        "source_cell_indices_zero_based": {
            "generate_tokens_and_compute_ari": 3,
            "direct_sampling": 5,
        },
        "source_fragments": {
            "decode": (
                "output_text = tokenizer.decode(output_tokens[0], "
                "skip_special_tokens=True)"
            ),
            "prompt_slice": "completion_text = output_text[len(prompt):]",
            "ari": "automated_readability_index(prompt + c)",
        },
        "observable_scope": (
            "the original prompt string concatenated with the character-sliced "
            "completion from the decoded full token sequence"
        ),
        "ari_preprocessing": (
            "no strip, whitespace normalization, case normalization, or newline "
            "processing is explicit in the public demo"
        ),
        "ari_formula": "4.71 * characters/words + 0.5 * words/sentences - 21.43",
        "paper_ari_cap": 15.0,
        "paper_cap_operation": (
            "described as a hard cap at 15 and consistent with min(raw_ari, 15), "
            "but the implementation and paired raw/capped TPS values are unavailable"
        ),
        "paper_ari_implementation": (
            "custom py-readability implementation with edge-case handling; "
            "implementation not published"
        ),
        "demo_ari_implementation": (
            "textstat.automated_readability_index, uncapped; public environment "
            "declares textstat>=0.7.12 without an exact lock"
        ),
        "demo_edge_case_handling": (
            "delegated to the installed textstat version; no additional handling "
            "appears in the public notebook"
        ),
        "demo_torch_seed": 42,
        "demo_python_seed": 1234,
        "dataset_seed": "not published",
        "log_probability_observable": (
            "re-tokenize decoded prompt plus completion, compute model log-softmax, "
            "zero prompt-token contributions, gather observed token IDs, and sum"
        ),
        "paper_data_use": (
            "Figure 3b direct ARI/log-probability curves, direct data incorporated "
            "into MBAR, and Figure 4 direct-sampling Wilson error analysis"
        ),
    }


def build_reference_summary(
    *,
    archive_path: Path,
    manifest: list[dict[str, Any]],
    dataset_inventory: list[dict[str, Any]],
    direct_statistics: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the stable machine-readable reference summary."""

    return {
        "schema_version": 1,
        "source": {
            "paper": "Rare Event Analysis of Large Language Models",
            "paper_url": "https://arxiv.org/abs/2602.06791",
            "official_repository": (
                "https://github.com/MLiS-Research/rea_of_llms_supplementary_code"
            ),
            "official_repository_commit_inspected": OFFICIAL_REPOSITORY_COMMIT,
            "zenodo_record": ZENODO_RECORD,
            "zenodo_url": f"https://zenodo.org/records/{ZENODO_RECORD}",
            "doi": ZENODO_DOI,
            "zenodo_publication_date": "2026-05-26",
            "zenodo_creators": [
                "Jake McAllister Dorman",
                "Edward Gillman",
                "Dominic C. Rose",
                "Jamie F. Mair",
                "Juan P. Garrahan",
            ],
            "zenodo_relation": "isSupplementTo arXiv:2602.06791",
            "archive": ARCHIVE_NAME,
            "archive_bytes": int(Path(archive_path).stat().st_size),
            "archive_md5": file_md5(archive_path),
            "published_archive_md5": EXPECTED_MD5,
            "direct_ari_member": DIRECT_ARI_MEMBER,
        },
        "official_configuration": _official_configuration(),
        "archive_manifest": manifest,
        "dataset_inventory": dataset_inventory,
        "direct_ari": direct_statistics,
        "published_count_discrepancy": {
            "paper_setup_text": "4.2 million direct completions",
            "figure_3b_caption": "4.1 million direct completions",
            "sampling_notebook_markdown": (
                "10^6 completions plus the ARI cap gives approximate paper results"
            ),
            "sampling_notebook_comment": 4_200_000,
            "reweighting_notebook_current_histogram_formula": 2_200_000,
            "zenodo_array_count": direct_statistics["count"],
        },
        "documentation_discrepancies": [
            (
                "The data README describes both log-probability files as containing "
                "ARI scores; their numeric ranges and filenames show they contain "
                "log-probabilities."
            ),
            (
                "The repository README says USE_EXAMPLE_DATA=True selects extracted "
                "paper data, while reweighting_example.ipynb uses True for data/example "
                "and False for data/paper."
            ),
            (
                "The paper states a cap of 15 for ARI, but the Zenodo direct ARI array "
                "contains values above 15; only the TPS ARI array is capped."
            ),
        ],
        "comparability": {
            "sampling_setup_match": (
                "Model name, tokenizer name, prompt, 100-token fixed length, and "
                "temperature-one ancestral sampling match the current experiment."
            ),
            "strict_observable_match_verified": False,
            "reason_observable_match_unverified": (
                "The archive has no completions or token IDs, and the paper's custom "
                "ARI edge-case implementation is not published. The official demo also "
                "decodes the full sequence and removes the prompt by character slicing, "
                "whereas this repository decodes completion token IDs separately. "
                "Recalculation with the repository textstat backend is impossible."
            ),
            "thresholds_6_8_10_12_15_preserved_under_min_cap": True,
            "raw_maximum_available": direct_statistics["ari_representation"] == "raw",
            "central_boxplot_usable": True,
            "lost_information_due_to_cap_in_direct_file": "none; direct ARI is raw",
            "ground_truth_assessment": (
                "High-precision empirical baseline for the official direct-sampling "
                "observable, but not exact ground truth for this repository's textstat "
                "raw-ARI convention until observable equivalence is established."
            ),
        },
    }


def write_json_deterministic(path: Path, payload: dict[str, Any]) -> None:
    """Write stable UTF-8 JSON with sorted keys and no runtime timestamp."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_reference_figure(
    path: Path,
    values: np.ndarray,
    summary: dict[str, Any],
) -> None:
    """Write one compact raw-ARI boxplot and empirical-tail panel."""

    import matplotlib.pyplot as plt

    box = summary["direct_ari"]["boxplot"]
    tails = summary["direct_ari"]["tails"]
    quantiles = summary["direct_ari"]["quantiles"]
    figure, (box_axis, tail_axis) = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.4),
        constrained_layout=True,
    )
    box_axis.bxp(
        [
            {
                "label": "official direct",
                "q1": box["q1"],
                "med": box["median"],
                "q3": box["q3"],
                "whislo": box["whisker_low"],
                "whishi": box["whisker_high"],
                "fliers": [],
            }
        ],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#4C78A8", "alpha": 0.65},
        medianprops={"color": "#B22222", "linewidth": 1.8},
    )
    box_axis.set_ylabel("ARI")
    box_axis.set_title("Official direct ARI (raw, uncapped)")
    box_axis.grid(axis="y", alpha=0.25)
    annotation = "\n".join(
        [
            f"q99 = {quantiles['q0.99']:.3f}",
            f"q99.9 = {quantiles['q0.999']:.3f}",
            f"q99.99 = {quantiles['q0.9999']:.3f}",
            f"raw max = {summary['direct_ari']['max']:.3f}",
        ]
    )
    box_axis.text(
        0.97,
        0.97,
        annotation,
        transform=box_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
    )

    thresholds = np.asarray([row["threshold"] for row in tails], dtype=np.float64)
    probabilities = np.asarray(
        [row["empirical_probability"] for row in tails], dtype=np.float64
    )
    lower = np.asarray([row["wilson_95_lower"] for row in tails], dtype=np.float64)
    upper = np.asarray([row["wilson_95_upper"] for row in tails], dtype=np.float64)
    positive = probabilities > 0.0
    tail_axis.errorbar(
        thresholds[positive],
        probabilities[positive],
        yerr=np.vstack(
            (
                probabilities[positive] - lower[positive],
                upper[positive] - probabilities[positive],
            )
        ),
        marker="o",
        color="#E45756",
        capsize=3,
        linewidth=1.2,
    )
    tail_axis.set_yscale("log")
    tail_axis.set_xlabel("threshold a")
    tail_axis.set_ylabel("P(ARI >= a)")
    tail_axis.set_title("Empirical upper tail (Wilson 95% CI)")
    tail_axis.grid(alpha=0.25, which="both")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=180,
        metadata={"Software": "rea_llms.official_direct_data"},
    )
    plt.close(figure)


def _format_probability(value: float | None) -> str:
    return "-" if value is None else f"{value:.8g}"


def render_reference_report(summary: dict[str, Any]) -> str:
    """Render the concise provenance, comparability, and result report."""

    source = summary["source"]
    config = summary["official_configuration"]
    direct = summary["direct_ari"]
    discrepancy = summary["published_count_discrepancy"]
    comparison = summary["comparability"]
    lines = [
        "# Official direct-sampling ARI reference",
        "",
        "## Provenance",
        "",
        f"- Paper: [{source['paper']}]({source['paper_url']})",
        f"- Dataset: [Zenodo {source['zenodo_record']}]({source['zenodo_url']})",
        f"- DOI: `{source['doi']}`",
        f"- Zenodo publication date: {source['zenodo_publication_date']}",
        f"- Archive: `{source['archive']}` ({source['archive_bytes']:,} bytes)",
        f"- Verified MD5: `{source['archive_md5']}`",
        f"- Direct ARI member: `{source['direct_ari_member']}`",
        "",
        "The archive checksum exactly matches Zenodo. It contains four JSON files and",
        "no completion text, token IDs, generation metadata, or separate README.",
        "",
        "## Official configuration",
        "",
        f"- Model: `{config['model']}` (revision not published)",
        f"- Tokenizer: `{config['tokenizer']}` (revision not published)",
        f"- Prompt ({config['prompt_token_count']} tokens): `{config['prompt']}`",
        f"- `repr(prompt)`: `{config['prompt_repr']}`",
        f"- Prompt SHA-256: `{config['prompt_sha256']}`",
        f"- Prompt token IDs: `{config['prompt_token_ids']}`",
        f"- Completion: {config['completion_tokens']} fixed tokens",
        f"- Generation: `do_sample={config['do_sample']}`, temperature "
        f"{config['temperature']}, `top_k=None`, top-p not explicitly set",
        "- EOS and padding IDs are passed as `None`; the demo asserts exact length",
        f"- Dataset seed: {config['dataset_seed']}; notebook demo seeds are "
        f"Torch={config['demo_torch_seed']} and Python={config['demo_python_seed']}",
        f"- Demo batching: {config['public_demo_batching']}; the paper generation "
        "implementation is not published",
        f"- Decode convention: {config['decode_convention']}",
        f"- Decode cleanup: {config['decode_clean_up_tokenization_spaces']}",
        f"- Exact public-demo ARI input: `{config['ari_input_expression']}`",
        "- Source: `external/rea_of_llms_official/sampling_examples.ipynb`, "
        "zero-based JSON cell 3 (`generate_tokens`, `compute_ari`) and cell 5 "
        "(`direct_sampling`)",
        f"- Decode fragment: `{config['source_fragments']['decode']}`",
        f"- Prompt-slice fragment: `{config['source_fragments']['prompt_slice']}`",
        f"- ARI fragment: `{config['source_fragments']['ari']}`",
        f"- ARI scope: {config['observable_scope']}",
        f"- Explicit preprocessing: {config['ari_preprocessing']}",
        f"- ARI formula: `{config['ari_formula']}`",
        f"- Paper method: {config['paper_ari_implementation']}",
        f"- Cap: {config['paper_cap_operation']}",
        f"- Public notebook demo: {config['demo_ari_implementation']}",
        f"- Public-demo edge cases: {config['demo_edge_case_handling']}",
        f"- Log-probability: {config['log_probability_observable']}",
        f"- Published use: {config['paper_data_use']}",
        "",
        "## Archive inventory",
        "",
        "| member | role | observable | observations | shape | min | max | invalid |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in summary["dataset_inventory"]:
        shape = json.dumps(row.get("shape"))
        invalid = int(row.get("nonfinite", 0)) + int(row.get("nonnumeric", 0))
        lines.append(
            f"| `{row['path']}` | {row['role']} | {row['observable']} | "
            f"{row['count']:,} | `{shape}` | {row['min']:.6g} | "
            f"{row['max']:.6g} | {invalid} |"
        )
    lines.extend(
        [
            "",
            "The trajectory files are TPS Markov-chain states, including repeated",
            "current states after rejected proposals; they are not IID direct samples or",
            "unaccepted proposal collections. The direct ARI member is the only official",
            "IID ARI array.",
            "",
            "## Count and representation discrepancies",
            "",
            f"- Experimental-setup text: {discrepancy['paper_setup_text']}.",
            f"- Figure 3b caption: {discrepancy['figure_3b_caption']}.",
            f"- Notebook direct-sampling markdown: {discrepancy['sampling_notebook_markdown']}.",
            f"- Notebook reconstruction comment: {discrepancy['sampling_notebook_comment']:,}.",
            "- Current reweighting-notebook formula with the archived trajectory "
            f"metadata selects {discrepancy['reweighting_notebook_current_histogram_formula']:,} "
            "direct values for its histogram.",
            f"- Actual Zenodo direct array: **{discrepancy['zenodo_array_count']:,}**.",
            "- The paper says ARI is capped at 15, and the TPS array is capped at 15.",
            f"  The direct array is demonstrably **raw**: max={direct['max']:.6g}, "
            f"{direct['values_above_15']} values exceed 15, and "
            f"{direct['values_equal_15']} equal 15 exactly.",
            "",
            "The dataset content is primary for this analysis. No values above 15 were",
            "lost from the direct array; the published cap applies to the TPS target/data.",
            "",
            "Two documentation errors are also visible: the official data README calls",
            "the log-probability arrays ARI arrays, and the repository README reverses",
            "the notebook's `USE_EXAMPLE_DATA` setting (`False` selects `data/paper`).",
            "",
            "## Direct ARI statistics",
            "",
            f"- N: **{direct['count']:,}**",
            f"- Mean: {direct['mean']:.8f}",
            f"- Population standard deviation: {direct['standard_deviation_population']:.8f}",
            f"- Min / max raw: {direct['min']:.8f} / {direct['max']:.8f}",
            f"- Q1 / median / Q3: {direct['quartiles']['q25']:.8f} / "
            f"{direct['quartiles']['q50']:.8f} / {direct['quartiles']['q75']:.8f}",
            "",
            "| quantile | ARI |",
            "|---:|---:|",
        ]
    )
    for level, value in direct["quantiles"].items():
        lines.append(f"| {level[1:]} | {value:.8f} |")
    lines.extend(
        [
            "",
            "## Empirical upper tails",
            "",
            "| threshold | count | probability | Wilson 95% lower | Wilson 95% upper | rule of three |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in direct["tails"]:
        lines.append(
            f"| {row['threshold']:g} | {row['count']:,} | "
            f"{row['empirical_probability']:.8g} | "
            f"{row['wilson_95_lower']:.8g} | {row['wilson_95_upper']:.8g} | "
            f"{_format_probability(row['rule_of_three_upper'])} |"
        )
    lines.extend(
        [
            "",
            "For thresholds up to and including 15, replacing raw ARI by",
            "`min(raw_ari, 15)` preserves every event indicator `ARI >= threshold`.",
            "It does not preserve raw maxima, upper-tail magnitudes, or distributions",
            "conditioned on values above 15.",
            "",
            "## Comparability with suffix direct-MH SMC",
            "",
            f"- {comparison['sampling_setup_match']}",
            f"- Strict observable equivalence: **not verified**. "
            f"{comparison['reason_observable_match_unverified']}",
            "- The central direct boxplot is usable as an official reference. A future",
            "  SMC boxplot must use weighted quantiles from final normalized weights.",
            "- Model/tokenizer revisions, the exact dataset-generation environment, and",
            "  dataset RNG seeds are not published.",
            "",
            "## Ground-truth assessment",
            "",
            comparison["ground_truth_assessment"],
            "",
            "It is appropriate as a large IID baseline for probabilities under the",
            "official observable at thresholds 6, 8, 10, 12, and 15. It should not be",
            "described as exact ground truth for our raw `textstat` ARI until the custom",
            "edge-case convention or original completions become available.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_thresholds(value: str) -> tuple[float, ...]:
    """Parse a non-empty comma-separated threshold list."""

    try:
        thresholds = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid threshold list: {value!r}") from exc
    if not thresholds or any(not math.isfinite(item) for item in thresholds):
        raise ValueError("thresholds must be a non-empty finite list")
    return tuple(dict.fromkeys(thresholds))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the official Zenodo direct-sampling ARI array without sampling."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--thresholds",
        default=",".join(f"{value:g}" for value in DEFAULT_THRESHOLDS),
    )
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument(
        "--verify-checksum",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--inspect-all-members",
        action="store_true",
        help="Also parse the three large non-ARI-direct arrays for a full inventory.",
    )
    parser.add_argument("--require-raw-maximum", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    thresholds = parse_thresholds(args.thresholds)
    archive_path = acquire_archive(
        args.data_root,
        download_if_missing=args.download_if_missing,
        verify_checksum=args.verify_checksum,
    )
    print(f"archive: {archive_path}")
    print(f"bytes: {archive_path.stat().st_size}")
    print(f"md5: {file_md5(archive_path)}")

    manifest = archive_manifest(archive_path)
    print("archive contents (before extraction):")
    for row in manifest:
        print(
            f"  {row['path']}: compressed={row['compressed_bytes']}, "
            f"uncompressed={row['uncompressed_bytes']}"
        )
    member = identify_direct_ari_member([row["path"] for row in manifest])
    extracted_path = extract_member_safely(
        archive_path,
        member,
        Path(args.data_root) / "extracted",
    )
    print(f"direct ARI file: {extracted_path}")

    values = load_numeric_json_array(extracted_path)
    direct_statistics = analyze_direct_ari(values, thresholds)
    if args.require_raw_maximum:
        ensure_raw_max_available(direct_statistics["ari_representation"])

    if args.inspect_all_members:
        dataset_inventory = [
            inspect_json_member(archive_path, row["path"])
            for row in manifest
        ]
    else:
        direct_manifest = next(row for row in manifest if row["path"] == member)
        dataset_inventory = [
            {
                **direct_manifest,
                "observable": "ari",
                "role": "direct_iid",
                "contains_completions": False,
                "contains_metadata": False,
                "shape": [direct_statistics["count"]],
                "count": direct_statistics["count"],
                "value_types": ["float"],
                "min": direct_statistics["min"],
                "max": direct_statistics["max"],
                "nonfinite": direct_statistics["nonfinite"],
                "nonnumeric": direct_statistics["nonnumeric"],
            }
        ]

    summary = build_reference_summary(
        archive_path=archive_path,
        manifest=manifest,
        dataset_inventory=dataset_inventory,
        direct_statistics=direct_statistics,
    )
    output_dir = Path(args.output_dir)
    summary_path = output_dir / "reference_direct_sampling_summary.json"
    figure_path = output_dir / "reference_direct_sampling_boxplot.png"
    report_path = output_dir / "REFERENCE_DIRECT_SAMPLING.md"
    write_json_deterministic(summary_path, summary)
    write_reference_figure(figure_path, values, summary)
    report_path.write_text(render_reference_report(summary), encoding="utf-8")
    print(f"summary: {summary_path.resolve()}")
    print(f"figure: {figure_path.resolve()}")
    print(f"report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
