from __future__ import annotations

import torch

from rea_llms import generation


def test_fixed_length_generation_releases_oom_traceback_before_retry(
    monkeypatch,
) -> None:
    attempted_batch_sizes: list[int] = []
    grad_enabled_during_attempts: list[bool] = []
    oom = torch.OutOfMemoryError("synthetic CUDA out of memory")

    def fake_generate_chunk(
        model,
        context_ids: torch.Tensor,
        *,
        suffix_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        del model, generator
        attempted_batch_sizes.append(context_ids.shape[0])
        grad_enabled_during_attempts.append(torch.is_grad_enabled())
        if context_ids.shape[0] > 1:
            raise oom
        return torch.zeros(
            (context_ids.shape[0], suffix_size),
            dtype=context_ids.dtype,
        )

    monkeypatch.setattr(generation, "_generate_suffix_chunk", fake_generate_chunk)
    result = generation.generate_fixed_length_suffixes(
        object(),
        torch.ones((4, 3), dtype=torch.long),
        suffix_size=2,
        batch_size=4,
        generator=torch.Generator().manual_seed(7),
    )

    assert attempted_batch_sizes == [4, 2, 1, 1, 1, 1]
    assert grad_enabled_during_attempts == [False] * len(attempted_batch_sizes)
    assert result.shape == (4, 2)
    assert oom.__traceback__ is None
