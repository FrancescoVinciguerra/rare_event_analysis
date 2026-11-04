# Artifact layout

Artifacts are deliberately separated from source files. Scientific results
are versioned in Git following the publication request; machine-local logs,
credentials, caches, and private Avon connection notes remain excluded.

- `current/`: destination for new direct, SMC, evaluation, and comparison
  artifacts.
- `logs/`: stdout and stderr from Avon jobs.
- `archive/avon_jobs/`: read-only local copies of earlier Avon results.
- `archive/transfer_packages/`: historical source-transfer archives.

See the [experiment results index](../../RESULTS_INDEX.md) for saved studies,
Gaussian toy experiments, and the status of ongoing jobs. A running job's
launch metadata is not a completed scientific result.

Archived Avon jobs currently present locally:

| Job | Contents | Completeness |
| --- | --- | --- |
| `2159528` | lambda calibration, manifest, summary, components and selected terminal examples | downloaded copy complete for the files originally collected |
| `2159674` | progressive-lambda manifest and summary | summary-only copy |
| `2159764` | 100-token four-component estimation | partial download; the remote originals remain authoritative |

Never use either `job_2159764_*partial_download` directory as a complete
scientific artifact. The reliable aggregate values are in its `summary.json`;
some component JSON files were interrupted during transfer.

Moving files into this layout does not alter `decoder_id`, `evaluator_id`, or
`smc_id`: filesystem paths are not part of those fingerprints.
