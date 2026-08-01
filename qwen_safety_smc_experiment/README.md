# Qwen safety SMC experiment

This folder prepares, but does not run, a scientific rare-event probability
experiment. It reuses the repository's central `suffix_direct_mh_smc` sampler
with a configurable completion observable.

## Full in-memory Colab notebook

[`notebooks/safety_smc_experiment.ipynb`](notebooks/safety_smc_experiment.ipynb)
contains the direct-versus-SMC experiment, ARI-style diagnostics adapted to
harmfulness, the validated SNIS correction back to the base model, and prefix
trajectories. It never mounts Drive and never writes experiment outputs, plots,
tables, or completions. The central sampler already returns an in-memory result,
so no persistence flag or duplicate sampler was introduced.

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
