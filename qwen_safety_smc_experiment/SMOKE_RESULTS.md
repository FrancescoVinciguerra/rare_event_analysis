# Local smoke results

Run date: 2026-08-01. These are infrastructure checks, not a scientific
probability estimate or a benchmark.

## Hardware and models

- GPU: NVIDIA GeForce MX570 A, 4 GB VRAM.
- Generator used locally: `Qwen/Qwen3-0.6B` in FP16 on CUDA. This is a
  hardware fallback for the smoke only; the Colab T4 default remains the
  selected `Qwen/Qwen3-1.7B`.
- Guardrail: official `Qwen/Qwen3Guard-Stream-0.6B`, pinned at revision
  `419364a715de9840d47b1457982f64ff37f90ed4`.

The local 4 GB GPU cannot safely hold the selected 1.7B generator and the
0.6B guard together. A 15 GB Colab T4 has sufficient weight-memory headroom
for both FP16 checkpoints; the notebook still performs only the tiny smoke.

## Observed results

- Generator load, including first download: 311.1 s.
- Batched generator probe: 2 prompts, 12 tokens each, 2.39 s; peak CUDA
  allocation 1,157.7 MiB.
- Guard load on CPU, including first download: 291.1 s.
- Guard terminal smoke: 35.4 s for the checks. The three safe, abstract test
  pairs were classified `Safe`, with terminal `P(Unsafe)` equal to 0.00296,
  0.00082, and 0.00570.
- A separate cached guard-only CUDA probe used 1,149.1 MiB peak allocation;
  three sequential prompt/completion scores took 1.05 s total.
- Tiny central SMC: 4 particles, 8 completion tokens, lambdas `[0, -0.25]`,
  one suffix mutation step. Runtime was 10.88 s, acceptance was 1.0, no
  resampling occurred, and final ESS/N was approximately 1.
- The prefix trajectory, terminal scoring, batching, exact cache, output
  serialization, and official-terminal-class agreement checks passed.
- The complete test suite passed: 81 tests.

Model downloads dominate the first run. Later runs reuse the Hugging Face
cache. The generated `outputs/smoke_<timestamp>/` artifacts are intentionally
ignored by Git; rerun the command in the experiment README to reproduce them.
