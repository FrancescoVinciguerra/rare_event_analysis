# Rare-event estimation for language models

The repository is organized into three independent experiment areas.

| Area | Start here | Contents |
| --- | --- | --- |
| **Gaussian toy model** | [`experiments/toy_gaussian/`](experiments/toy_gaussian/README.md) | Full-trajectory RollBack SMC, self-contained Colab notebooks, and saved Gaussian runs |
| **Autoregressive LLMs** | [`experiments/llm/`](experiments/llm/README.md) | Suffix MH, annealed TPS, ARI and QwenGuard notebooks, and tail validation |
| **Masked diffusion (MDLM)** | [`experiments/mdlm_toxicity_rare_event/`](experiments/mdlm_toxicity_rare_event/README.md) | Toxicity-tail estimator, Avon campaigns, analysis, and figure renderers |

The [results index](experiments/RESULTS_INDEX.md) links to saved runs and the
main figures. Shared Python utilities live in [`rea_llms/`](rea_llms/).
The original supplementary code is pinned in [`external/`](external/),
and paper sources live in [`paper/`](paper/).

To install and test the Python code from the repository root:

```bash
python -m pip install -e .
python -m pytest -q
```

The Gaussian notebooks run directly in Colab. For MDLM, see the
[quick start](experiments/mdlm_toxicity_rare_event/README.md#quick-start).
To export the standalone MDLM source, run
`python tools/export_mdlm_source.py` from an installed checkout.
