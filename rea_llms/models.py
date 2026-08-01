from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import transformers


DEFAULT_PROMPT = "Once upon a time, in a big forest, there lived a rhinoc"
DEFAULT_MODEL_NAME = "roneneldan/TinyStories-8M"
DEFAULT_TOKENIZER_NAME = "EleutherAI/gpt-neo-125M"
TORCH_DTYPES = ("auto", "float16", "bfloat16", "float32")


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a CLI device request and reject unavailable explicit CUDA."""

    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be one of {'auto', 'cuda', 'cpu'}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is False")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, CUDA, and Transformers RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


def resolve_torch_dtype(dtype: str, device: torch.device) -> torch.dtype | None:
    """Resolve a portable model dtype and reject BF16 on unsupported CUDA."""

    if dtype not in TORCH_DTYPES:
        raise ValueError(f"dtype must be one of {TORCH_DTYPES}")
    if dtype == "auto":
        return None
    resolved = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    if (
        device.type == "cuda"
        and resolved == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("bfloat16 was requested, but the selected CUDA device lacks BF16")
    return resolved


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    tokenizer_name: str | None = None,
    *,
    local_files_only: bool = False,
    device: str = "auto",
    dtype: str = "auto",
) -> tuple[Any, Any]:
    """Load the shared Hugging Face model/tokenizer pair on one device."""

    selected_device = resolve_device(device)
    selected_dtype = resolve_torch_dtype(dtype, selected_device)
    tokenizer_name = tokenizer_name or model_name
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name,
        local_files_only=local_files_only,
    )
    model_kwargs = {"local_files_only": local_files_only}
    if selected_dtype is not None:
        model_kwargs["torch_dtype"] = selected_dtype
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(selected_device)
    model.eval()
    if (
        getattr(tokenizer, "pad_token_id", None) is None
        and getattr(tokenizer, "eos_token_id", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
