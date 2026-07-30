from __future__ import annotations

from typing import Any

import torch


CURRENT_DECODING_CONVENTION = (
    "decode completion token IDs separately with skip_special_tokens=True and "
    "the tokenizer default for clean_up_tokenization_spaces; prepend the exact "
    "prompt string before textstat ARI"
)


def score_completion_tokens(
    tokenizer,
    observable,
    *,
    prompt: str,
    prompt_ids: torch.Tensor,
    completions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[Any]]:
    """Score completion-token batches with the shared current-pipeline convention."""

    prompt_batch = prompt_ids.unsqueeze(0).expand(completions.shape[0], -1)
    full_ids = torch.cat((prompt_batch, completions), dim=1).detach().cpu().tolist()
    score_batch = getattr(observable, "score_token_sequences", None)
    if callable(score_batch):
        scores = score_batch(
            tokenizer=tokenizer,
            prompt=prompt,
            token_sequences=full_ids,
            prompt_token_count=int(prompt_ids.numel()),
        )
    else:
        completion_ids = completions.detach().cpu().tolist()
        texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        scores = [observable.score_completion(prompt, text) for text in texts]
    phi = torch.tensor(
        [float(score.value) for score in scores],
        dtype=torch.float32,
        device=completions.device,
    )
    raw_ari = torch.tensor(
        [float(score.metadata["raw_ari"]) for score in scores],
        dtype=torch.float32,
        device=completions.device,
    )
    if not bool(torch.isfinite(phi).all().item()) or not bool(
        torch.isfinite(raw_ari).all().item()
    ):
        raise ValueError("Observable returned a non-finite value")
    return phi, raw_ari, scores
