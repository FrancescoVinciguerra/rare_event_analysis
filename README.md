# Rare-event estimation for language models

Start in one of the three research directories at the repository root.

| Area | Start here | Contents |
| --- | --- | --- |
| **Gaussian toy model** | [`toy_gaussian/`](toy_gaussian/README.md) | Full-trajectory RollBack SMC, self-contained Colab notebooks, and saved Gaussian runs |
| **Autoregressive LLMs** | [`llm/`](llm/README.md) | Suffix MH, annealed TPS, ARI and QwenGuard notebooks, and tail validation |
| **Masked diffusion (MDLM)** | [`mdlm/`](mdlm/README.md) | Toxicity-tail estimator, Avon campaigns, analysis, and figure renderers |

The [results index](shared/RESULTS_INDEX.md) links to saved runs and figures.
[`shared/`](shared/README.md) holds common Python utilities, manuscript files,
and publication exports.

To install and test the Python code from the repository root:

```bash
python -m pip install -e .
python -m pytest -q
```

The Gaussian notebooks run directly in Colab. For MDLM, see the
[quick start](mdlm/README.md#quick-start).
To export the standalone MDLM source, run
`python mdlm/tools/export_mdlm_source.py` from an installed checkout.
