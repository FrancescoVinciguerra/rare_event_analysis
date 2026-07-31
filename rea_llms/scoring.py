from __future__ import annotations

from typing import Any

import torch


ARI_CONVENTION_CURRENT = "current"
ARI_CONVENTION_OFFICIAL_PUBLIC = "official_public"
ARI_CONVENTIONS = (
    ARI_CONVENTION_CURRENT,
    ARI_CONVENTION_OFFICIAL_PUBLIC,
)

CURRENT_DECODING_CONVENTION = (
    "decode completion token IDs separately with skip_special_tokens=True and "
    "the tokenizer default for clean_up_tokenization_spaces; prepend the exact "
    "prompt string before textstat ARI"
)
OFFICIAL_PUBLIC_DECODING_CONVENTION = (
    "decode the full prompt-token plus completion-token sequence with "
    "skip_special_tokens=True and the tokenizer default for "
    "clean_up_tokenization_spaces; remove the first len(prompt) characters, "
    "then prepend the exact prompt string before textstat ARI"
)


def validate_ari_convention(convention: str) -> str:
    if convention not in ARI_CONVENTIONS:
        raise ValueError(
            f"ARI convention must be one of {ARI_CONVENTIONS}, got {convention!r}"
        )
    return convention


def ari_decoding_description(convention: str) -> str:
    validate_ari_convention(convention)
    if convention == ARI_CONVENTION_CURRENT:
        return CURRENT_DECODING_CONVENTION
    return OFFICIAL_PUBLIC_DECODING_CONVENTION


def text_for_ari_from_token_ids(
    tokenizer,
    *,
    prompt: str,
    token_ids: list[int],
    prompt_token_count: int,
    convention: str,
) -> str:
    """Reconstruct the exact ARI input text for one declared convention."""

    validate_ari_convention(convention)
    if convention == ARI_CONVENTION_OFFICIAL_PUBLIC:
        decoded_full = tokenizer.decode(token_ids, skip_special_tokens=True)
        completion = decoded_full[len(prompt) :]
    else:
        completion = tokenizer.decode(
            token_ids[prompt_token_count:],
            skip_special_tokens=True,
        )
    return prompt + completion


def score_completion_tokens(
    tokenizer,
    observable,
    *,
    prompt: str,
    prompt_ids: torch.Tensor,
    completions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[Any]]:
    """Score completion-token batches with the observable's declared convention."""

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
