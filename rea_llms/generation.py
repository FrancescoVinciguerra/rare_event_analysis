from __future__ import annotations

from typing import Any

import torch


GENERATION_ENGINE_VERSION = "fixed_length_categorical_v1"


def prompt_token_ids(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    """Tokenize the exact prompt using the shared SMC/direct convention."""

    encoded = tokenizer(prompt, return_tensors="pt")["input_ids"]
    if not isinstance(encoded, torch.Tensor):
        encoded = torch.tensor(encoded, dtype=torch.long)
    return encoded[0].to(device=device, dtype=torch.long)


def _base_model_parts(model) -> tuple[Any | None, Any | None]:
    base_model = getattr(model, "base_model", None)
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    output_embeddings = get_output_embeddings() if callable(get_output_embeddings) else None
    if base_model is None or base_model is model or output_embeddings is None:
        return None, None
    return base_model, output_embeddings


def _project_hidden(model, output_embeddings, hidden: torch.Tensor) -> torch.Tensor:
    logits = output_embeddings(hidden)
    final_bias = getattr(model, "final_logits_bias", None)
    return logits if final_bias is None else logits + final_bias


def _selected_next_token_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    base_model, output_embeddings = _base_model_parts(model)
    if base_model is not None:
        outputs = base_model(input_ids=input_ids, use_cache=False, return_dict=True)
        return _project_hidden(model, output_embeddings, outputs.last_hidden_state[:, -1])
    return model(input_ids=input_ids).logits[:, -1]


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _generate_suffix_chunk(
    model,
    context_ids: torch.Tensor,
    *,
    suffix_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate an unrestricted fixed-length suffix for an equal-length batch."""

    base_model, output_embeddings = _base_model_parts(model)
    generated = torch.empty(
        (context_ids.shape[0], suffix_size),
        dtype=context_ids.dtype,
        device=context_ids.device,
    )
    if base_model is not None:
        outputs = base_model(input_ids=context_ids, use_cache=True, return_dict=True)
        hidden = outputs.last_hidden_state[:, -1]
        past = getattr(outputs, "past_key_values", None)
        work = context_ids
        for step in range(suffix_size):
            logits = _project_hidden(model, output_embeddings, hidden)
            token = torch.multinomial(
                torch.softmax(logits, dim=-1),
                num_samples=1,
                generator=generator,
            )
            generated[:, step : step + 1].copy_(token)
            if step + 1 == suffix_size:
                break
            if past is not None:
                outputs = base_model(
                    input_ids=token,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                hidden = outputs.last_hidden_state[:, -1]
                past = getattr(outputs, "past_key_values", None)
            else:
                work = torch.cat((work, token), dim=1)
                hidden = base_model(
                    input_ids=work,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state[:, -1]
        return generated

    work = context_ids
    for step in range(suffix_size):
        logits = _selected_next_token_logits(model, work)
        token = torch.multinomial(
            torch.softmax(logits, dim=-1),
            num_samples=1,
            generator=generator,
        )
        generated[:, step : step + 1].copy_(token)
        work = torch.cat((work, token), dim=1)
    return generated


def generate_fixed_length_suffixes(
    model,
    context_ids: torch.Tensor,
    *,
    suffix_size: int,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Generate suffixes in GPU batches with safe CUDA OOM backoff.

    Sampling is unrestricted categorical sampling at temperature one. EOS is
    not interpreted as a stopping condition, and no top-k/top-p filter,
    penalties, beams, or length normalization are applied.
    """

    if context_ids.ndim != 2 or context_ids.shape[0] <= 0:
        raise ValueError("context_ids must be a non-empty rank-two tensor")
    if suffix_size <= 0 or batch_size <= 0:
        raise ValueError("suffix_size and batch_size must be positive")
    chunks: list[torch.Tensor] = []
    start = 0
    active_batch_size = min(batch_size, context_ids.shape[0])
    while start < context_ids.shape[0]:
        stop = min(start + active_batch_size, context_ids.shape[0])
        try:
            chunks.append(
                _generate_suffix_chunk(
                    model,
                    context_ids[start:stop],
                    suffix_size=suffix_size,
                    generator=generator,
                )
            )
            start = stop
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            if not _is_cuda_oom(exc) or active_batch_size == 1:
                raise RuntimeError(
                    f"CUDA OOM while generating suffix size {suffix_size} at batch "
                    f"size {active_batch_size}; reduce --batch-size"
                ) from exc
            active_batch_size = max(1, active_batch_size // 2)
            if context_ids.device.type == "cuda":
                torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def generate_fixed_length_completions(
    model,
    prompt_ids: torch.Tensor,
    *,
    num_sequences: int,
    max_new_tokens: int,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw IID fixed-length completions from the base language model."""

    if num_sequences <= 0:
        raise ValueError("num_sequences must be positive")
    contexts = prompt_ids.unsqueeze(0).expand(num_sequences, -1)
    return generate_fixed_length_suffixes(
        model,
        contexts,
        suffix_size=max_new_tokens,
        batch_size=batch_size,
        generator=generator,
    )
