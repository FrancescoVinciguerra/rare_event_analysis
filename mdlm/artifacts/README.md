# Artifact layout

Artifacts are deliberately separated from source files. Scientific results
are versioned in Git following the publication request; machine-local logs,
credentials, caches, and private Avon connection notes remain excluded.

- `current/`: run data, CSVs and reports, grouped by campaign. The
  `camera_ready_*` campaigns keep generated images in their own `plots/`.
- `logs/`: stdout and stderr from Avon jobs.
- `archive/avon_jobs/`: read-only local copies of earlier Avon results.
- `archive/transfer_packages/`: historical source-transfer archives.

Extracted campaign files are the working copies. Transport archives are kept
when they contain unique files or original versions of reports.

Within a campaign, `workers/*/populations/` holds SMC populations and their
weights and checkpoints. `baselines/` holds saved crude Monte Carlo draws.
`runs.csv` and `summary.csv` are the analysis tables. A `code/` directory is
the frozen source snapshot of that campaign.

See the [experiment results index](../../shared/RESULTS_INDEX.md) for saved studies,
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
