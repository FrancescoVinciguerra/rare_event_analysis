from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Callable

import torch


GENERATION_ENGINE_VERSION = "fixed_length_categorical_v1"


@dataclass
class GenerationDiagnostics:
    """Successful batched autoregressive work performed by shared generation."""

    batch_calls: int = 0
    model_forward_calls: int = 0
    generated_token_transitions: int = 0


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
    diagnostics: GenerationDiagnostics | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
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
            with torch.inference_mode():
                chunk = _generate_suffix_chunk(
                    model,
                    context_ids[start:stop],
                    suffix_size=suffix_size,
                    generator=generator,
                )
            chunks.append(chunk)
            if diagnostics is not None:
                diagnostics.batch_calls += 1
                diagnostics.model_forward_calls += suffix_size
                diagnostics.generated_token_transitions += (stop - start) * suffix_size
            start = stop
            if progress_callback is not None:
                progress_callback(start, context_ids.shape[0])
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            if not _is_cuda_oom(exc) or active_batch_size == 1:
                raise RuntimeError(
                    f"CUDA OOM while generating suffix size {suffix_size} at batch "
                    f"size {active_batch_size}; reduce --batch-size"
                ) from exc
            active_batch_size = max(1, active_batch_size // 2)
            # The traceback owns the failed generation frame and its CUDA tensors.
            # Detach it before asking the allocator to release cached blocks.
            exc.__traceback__ = None
            gc.collect()
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
    diagnostics: GenerationDiagnostics | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
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
        diagnostics=diagnostics,
        progress_callback=progress_callback,
    )


def fixed_completion_log_probabilities(
    model,
    prompt_ids: torch.Tensor,
    completions: torch.Tensor,
    *,
    batch_size: int,
    position_batch_size: int = 4,
) -> torch.Tensor:
    """Teacher-force fixed completions and sum their base-model log probabilities.

    For standard Hugging Face causal models, hidden states are projected only
    for the completion positions in small chunks. This avoids materializing a
    full ``batch x sequence x vocabulary`` logits tensor for large vocabularies.
    """

    if completions.ndim != 2 or completions.shape[0] <= 0:
        raise ValueError("completions must be a non-empty rank-two tensor")
    if batch_size <= 0 or position_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if prompt_ids.ndim != 1 or prompt_ids.numel() <= 0:
        raise ValueError("prompt_ids must be a non-empty vector")
    if prompt_ids.device != completions.device:
        raise ValueError("prompt_ids and completions must share a device")

    base_model, output_embeddings = _base_model_parts(model)
    rows: list[torch.Tensor] = []
    prompt_length = int(prompt_ids.numel())
    with torch.inference_mode():
        for start in range(0, completions.shape[0], batch_size):
            completion = completions[start : start + batch_size]
            prompt = prompt_ids.unsqueeze(0).expand(completion.shape[0], -1)
            full_ids = torch.cat((prompt, completion), dim=1)
            targets = completion
            if base_model is not None:
                hidden = base_model(
                    input_ids=full_ids,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state[:, prompt_length - 1 : -1]
                row_log_probability = torch.zeros(
                    completion.shape[0],
                    dtype=torch.float64,
                    device=completion.device,
                )
                for position_start in range(0, completion.shape[1], position_batch_size):
                    position_stop = min(
                        position_start + position_batch_size,
                        completion.shape[1],
                    )
                    logits = _project_hidden(
                        model,
                        output_embeddings,
                        hidden[:, position_start:position_stop],
                    ).float()
                    selected = torch.gather(
                        logits,
                        dim=-1,
                        index=targets[:, position_start:position_stop].unsqueeze(-1),
                    ).squeeze(-1)
                    row_log_probability += (
                        selected - torch.logsumexp(logits, dim=-1)
                    ).to(torch.float64).sum(dim=1)
            else:
                logits = model(input_ids=full_ids).logits[
                    :, prompt_length - 1 : -1
                ].float()
                selected = torch.gather(
                    logits,
                    dim=-1,
                    index=targets.unsqueeze(-1),
                ).squeeze(-1)
                row_log_probability = (
                    selected - torch.logsumexp(logits, dim=-1)
                ).to(torch.float64).sum(dim=1)
            rows.append(row_log_probability)
    return torch.cat(rows, dim=0)
