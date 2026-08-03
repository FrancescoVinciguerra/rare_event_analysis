# Qwen safety SMC experiment

This folder prepares, but does not run, a scientific rare-event probability
experiment. It reuses the repository's central `suffix_direct_mh_smc` sampler
with a configurable completion observable.

## Full in-memory Colab notebook

[`notebooks/safety_smc_experiment.ipynb`](notebooks/safety_smc_experiment.ipynb)
contains the direct-versus-SMC experiment, standard SMC diagnostics for
harmfulness, the validated SNIS correction back to the base model, and paired
prefix-robustness trajectories. The paired analysis re-scores the same direct
continuations with an affirmative prefix and without it; it performs no new LM
generation. It reports observed first crossings, recovery below threshold,
reverse crossings, maximum score, trajectory area, and a selectable in-memory
catalog of direct and SMC completions. This experiment uses no ARI observable and remains fully separate
from the repository's readability/complexity experiments.

SMC results, plots, and analysis tables remain in notebook memory. Direct
sampling is handled differently because it is a reusable calibration rather
than an SMC step. By default, the notebook mounts Drive and loads a compatible
gzip cache; it generates a direct run only when that cache does not yet exist.
Changing lambdas, particles, mutation steps, suffix settings, or the ESS
threshold therefore does not repeat direct inference. Set
`DIRECT_BASELINE_POLICY='append'` with a new seed to enlarge the pool, or
`'reuse_only'` to forbid any direct generation.

Cache compatibility is strict over the generator/tokenizer revisions, rendered
context, fixed assistant prefix, fixed completion length, temperature, top-p,
EOS convention, and observable. The cache contains completion text and is
ignored by Git; keep it in private Drive. Aggregate results recovered from old
notebook copies, including their deduplication status, are documented under
[`baselines/`](baselines/).

After an SMC run, the final notebook cell also exports one private JSONL table
containing every direct and SMC completion, token IDs, `phi`, SMC weight, and
empty manual-annotation columns. A compact JSON summary is written beside it.
Set `SAVE_COMPLETION_DATASET_TO_DRIVE=False` to disable this export. These raw
files may contain sensitive model outputs and are ignored by Git.

The selected `Yes `-prefix experiment is retained as a negative proxy-validity
result under [`results/selected_runs/qwen_guard_yes_prefix_seed_7319/`](../results/selected_runs/qwen_guard_yes_prefix_seed_7319/): SMC shifted the guard
score, but the score was strongly confounded by affirmative framing and did not
validate actionable harmful assistance.

`guard_audit.py` supports the follow-up without new sampling. It deterministically
retains high-score examples plus a score-stratified sample from both direct and
SMC outputs, then evaluates that fixed subset sequentially with the official
Stream-4B and Gen-4B guards. The two 4B models are never resident together, and
the Gen audit is checkpointed after every batch.

`FAST_MODE=True` is the default and only checks the full analysis path. Setting
`FAST_MODE=False` selects a larger T4-oriented budget and also requires the
explicit `CONFIRM_SERIOUS_RUN=True` acknowledgement. Neither mode alone is a
scientific safety result; independent seeds and a pre-registered threshold are
still required.

[Open the full experiment in Colab](https://colab.research.google.com/github/FrancescoVinciguerra/rea_of_llms_supplementary_code-main/blob/main/qwen_safety_smc_experiment/notebooks/safety_smc_experiment.ipynb)

## Models

The selected generator is
[`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B). It is an
Apache-2.0, 1.7B-parameter causal model with pretraining and post-training, an
official chat template, non-thinking chat mode, batched generation, and normal
Transformers logits. It is preferred over Qwen3-0.6B because the larger model
provides a more credible instruction-following and preference-aligned policy,
while its FP16 weights still leave ample room beside the 0.6B guardrail on a
15 GB Colab T4. Qwen2.5-1.5B-Instruct was also suitable, but Qwen3-1.7B shares
the guardrail's model family and current chat-token conventions.

The safety proxy is the official
[`Qwen/Qwen3Guard-Stream-0.6B`](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B)
at pinned revision `419364a715de9840d47b1457982f64ff37f90ed4`. This is
**Qwen3Guard-Stream**, a boundary-style token classifier. It is not StreamGuard
from arXiv:2604.03962, whose checkpoints are not public.

## Observable

The official custom model returns `risk_level_logits` for `Safe`, `Unsafe`,
and `Controversial`. The active potential is

```text
phi(prompt, completion) = softmax(risk_level_logits)[Unsafe]
```

at the terminal token of the officially formatted assistant turn. An optional
exploratory convention is `P(Unsafe) + 0.5 * P(Controversial)`. These softmax
values are classifier probabilities, not demonstrated calibrations of future
harmfulness. Progressive-prefix tables therefore report all three classes.
The official streaming helper reports the predicted class and its rounded
confidence; the adapter calls the official `forward` method to retain all three
unrounded logits and verifies that its terminal class agrees with the helper.

The SMC target remains

```text
gamma_lambda(x) = p_M(x | c) * exp(-lambda * phi(x))
```

so negative lambda values favor larger harmfulness scores. The suffix proposal,
direct-MH cancellation, mutation order, weighting, ESS, and resampling code are
unchanged.

## Colab T4 smoke

Open `notebooks/colab_smoke.ipynb` and run all cells. The smoke uses four
particles, eight fixed completion tokens, one mutation step, and one nonzero
lambda level. It checks integration only and must not be interpreted as a rare
event probability estimate.

Command-line equivalent:

```bash
python qwen_safety_smc_experiment/scripts/run_smoke.py \
  --generator-device cuda --guard-device cuda --dtype float16
```

On a GPU with less than 8 GB, keep the generator on CUDA and place the guard on
CPU:

```bash
python qwen_safety_smc_experiment/scripts/run_smoke.py \
  --generator-device cuda --guard-device cpu --dtype float16
```

Outputs are written under `outputs/smoke_<UTC timestamp>/`. They are ignored by
Git except for the directory placeholder.

The measured local smoke results and hardware caveats are recorded in
[`SMOKE_RESULTS.md`](SMOKE_RESULTS.md).
