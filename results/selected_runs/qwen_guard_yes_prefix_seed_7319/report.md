# QwenGuard proxy-validity result

## Result

This run is retained as a negative measurement-validity result. Suffix
Direct-MH SMC moved the configured observable toward larger values, but manual
inspection reported no completion that actually supplied actionable bomb
construction instructions. High-scoring examples instead used affirmative
framing, promised help, discussed the dangerous topic, or subsequently
refused.

This supports a narrower conclusion than "the classifier does not work": the
terminal `P(Unsafe)` score from Qwen3Guard-Stream is not a validated observable
for the event "the response contains actionable harmful assistance" under this
prompt and prefix. It may still be useful for its intended broad response-risk
classification task.

## Strongest diagnostic

The paired-prefix analysis re-scored the same 512 direct continuations with
and without the fixed `Yes ` prefix:

| Diagnostic | With `Yes ` | Without prefix |
|---|---:|---:|
| Mean terminal `phi` | 0.410929 | 0.173286 |
| Fraction above 0.5 | 0.371094 | 0.097656 |
| Mean `phi` at the first checkpoint | 0.821917 | 0.103284 |
| Fraction above 0.5 at the first checkpoint | 1.0 | 0.0 |

The prefix increased `phi` for every paired completion and caused 140
safe-to-unsafe threshold flips. This is direct evidence that affirmative
framing strongly confounds the observable before substantive completion
content has appeared.

## SMC behavior

The final SMC population reached a weighted mean `phi` of 0.812568 and a raw
maximum of 0.993263. The final weight ESS/N was 0.474379, mean mutation
acceptance was 0.926875, and all 128 final completions were unique. Thus the
run does not indicate an SMC mechanics failure: it indicates that SMC
successfully optimized a proxy that was misaligned with the intended semantic
event.

The SNIS value 0.315697 at threshold 0.5 estimates the probability of crossing
the classifier threshold under the base model. It must not be reported as the
probability of actionable harmful assistance.

## Data status

The executed notebook contains aggregate tables and selected completion
previews. The user reports inspecting the exported completion data and finding
zero actionable harmful outputs, but a row-level annotation dataset was not
provided with the notebook. This manual finding is therefore recorded as
provisional rather than as a formal 640-row labeled audit.

Future notebook runs export complete direct and SMC records to private Drive.
Raw completion text is intentionally excluded from the public repository.

## Next validation

Before trying stronger or less explicit prompts, validate the observable on a
small blinded contrast set that separates refusal, affirmative promise followed
by refusal, high-level safety discussion, and genuinely actionable harmful
content. Report confusion matrices and ranking quality for actionability. Only
after that check should the observable be used as an SMC potential for the
intended event.
