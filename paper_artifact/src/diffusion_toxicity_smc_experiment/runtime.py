"""Pinned model/tokenizer loading for the MDLM/FK benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class GeneratorRuntime:
    model: Any
    tokenizer: Any
    device: torch.device
    model_load_seconds: float
    tokenizer_load_seconds: float


@dataclass
class EvaluatorRuntime:
    classifier: Any
    classifier_tokenizer: Any
    completion_trimming_tokenizer: Any
    device: torch.device


def _device_and_cache(config: dict[str, Any]) -> tuple[torch.device, str | None]:
    device = torch.device(config.get("runtime", {}).get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device, config.get("runtime", {}).get("hf_cache")


def load_generator_runtime(
    config: dict[str, Any], *, local_files_only: bool = True
) -> GeneratorRuntime:
    decoder = config["decoder"]
    device, cache_dir = _device_and_cache(config)

    tokenizer_started = time.perf_counter()
    tokenizer = load_decoder_tokenizer(config, local_files_only=local_files_only)
    tokenizer_load_seconds = time.perf_counter() - tokenizer_started

    implementation = decoder["model_implementation"]
    model_started = time.perf_counter()
    model = AutoModelForMaskedLM.from_pretrained(
        implementation["id"],
        revision=implementation["revision"],
        code_revision=implementation["revision"],
        trust_remote_code=True,
        use_safetensors=True,
        dtype=torch.float32,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    ).to(device).eval()
    model_load_seconds = time.perf_counter() - model_started

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return GeneratorRuntime(
        model,
        tokenizer,
        device,
        model_load_seconds=model_load_seconds,
        tokenizer_load_seconds=tokenizer_load_seconds,
    )


def load_decoder_tokenizer(config: dict[str, Any], *, local_files_only: bool = True) -> Any:
    decoder = config["decoder"]
    _, cache_dir = _device_and_cache(config)
    tokenizer_spec = decoder["tokenizer"]
    return AutoTokenizer.from_pretrained(
        tokenizer_spec["id"],
        revision=tokenizer_spec["revision"],
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def load_evaluator_runtime(
    config: dict[str, Any], *, local_files_only: bool = True
) -> EvaluatorRuntime:
    evaluation = config["evaluation"]
    device, cache_dir = _device_and_cache(config)
    classifier_spec = evaluation["classifier"]
    classifier_tokenizer = AutoTokenizer.from_pretrained(
        classifier_spec["id"],
        revision=classifier_spec["revision"],
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    classifier_dtype = torch.float16 if device.type == "cuda" else torch.float32
    classifier = AutoModelForSequenceClassification.from_pretrained(
        classifier_spec["id"],
        revision=classifier_spec["revision"],
        dtype=classifier_dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    ).to(device).eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    trimming_spec = evaluation["completion_trimming_tokenizer"]
    trimming_tokenizer = AutoTokenizer.from_pretrained(
        trimming_spec["id"],
        revision=trimming_spec["revision"],
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return EvaluatorRuntime(
        classifier, classifier_tokenizer, trimming_tokenizer, device
    )


def official_prompt_token_ids(tokenizer: Any, prompt: str, device: torch.device) -> torch.Tensor:
    """Reproduce FK's BOS + prompt prefix after dropping the tokenizer-added EOS."""

    text_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if tokenizer.bos_token_id is None:
        raise ValueError("MDLM GPT-2 tokenizer must define bos_token_id")
    return torch.tensor([tokenizer.bos_token_id, *text_ids], dtype=torch.long, device=device)
