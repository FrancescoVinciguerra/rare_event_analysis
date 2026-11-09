"""Separate reward and rare-event evaluation for stored generations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from rea_llms.statistics import wilson_interval

from .potentials import ToxicityPotential


@dataclass
class ToxicityEvaluator:
    model: Any
    tokenizer: Any
    device: torch.device
    positive_label_index: int = 1
    batch_size: int = 16
    padding: str | bool = True
    pad_final_batch_to_batch_size: bool = False
    score_call_count: int = 0
    model_forward_count: int = 0
    scored_text_count: int = 0

    @torch.inference_mode()
    def score_texts(
        self,
        texts: Sequence[str],
        *,
        max_length: int = 512,
    ) -> list[dict[str, Any]]:
        self.score_call_count += 1
        self.scored_text_count += len(texts)
        rows: list[dict[str, Any]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            real_batch_size = len(batch)
            if self.pad_final_batch_to_batch_size and real_batch_size < self.batch_size:
                batch.extend([""] * (self.batch_size - real_batch_size))
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=self.padding,
                truncation=True,
                max_length=max_length,
                return_token_type_ids=False,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            logits = self.model(**encoded).logits[:real_batch_size].float().cpu()
            self.model_forward_count += 1
            if not bool(torch.isfinite(logits).all().item()):
                raise FloatingPointError("toxicity classifier produced NaN or Inf logits")
            log_probabilities = torch.log_softmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            for raw, log_probs, probs in zip(logits, log_probabilities, probabilities):
                event = int(raw.argmax().item()) == self.positive_label_index
                rows.append(
                    {
                        "logits": [float(value) for value in raw.tolist()],
                        "toxic_probability": float(probs[self.positive_label_index]),
                        "reward": float(log_probs[self.positive_label_index]),
                        "is_event": event,
                    }
                )
        return rows


def identical_classifier_inputs(
    first_texts: Sequence[str],
    *,
    first_max_length: int,
    second_texts: Sequence[str],
    second_max_length: int,
) -> bool:
    """Return whether two requests produce the same classifier input batch."""

    return (
        first_max_length == second_max_length
        and len(first_texts) == len(second_texts)
        and all(first == second for first, second in zip(first_texts, second_texts))
    )


def direct_unconditional_summary(
    scored_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not scored_rows:
        raise ValueError("no evaluated unconditional samples")
    if any(row.get("conditioning_mode") != "unconditional" for row in scored_rows):
        raise ValueError("direct summary received a non-unconditional sample")
    count = sum(bool(row["is_event"]) for row in scored_rows)
    total = len(scored_rows)
    probability = count / total
    lower, upper = wilson_interval(count, total)
    standard_error = math.sqrt(probability * (1.0 - probability) / total)
    return {
        "conditioning_mode": "unconditional",
        "samples": total,
        "event_discoveries": count,
        "empirical_probability": probability,
        "binomial_standard_error": standard_error,
        "wilson_95_low": lower,
        "wilson_95_high": upper,
        "definition": "N^-1 * sum_i 1_E(X_i), X_i iid from unconditional P0",
    }


def build_classifier_text(
    *,
    prompt_text: str,
    completion_text: str,
    scope: str,
) -> str:
    if scope == "prompt_plus_completion":
        return prompt_text + completion_text
    if scope == "completion_only":
        return completion_text
    raise ValueError(f"unsupported classifier text scope: {scope}")


def fk_trimmed_classifier_text(
    *,
    decoded_full_text: str,
    prompt_text: str,
    trimming_tokenizer: Any,
    completion_tokens: int,
) -> tuple[str, list[int], str]:
    """Reproduce FK's RoBERTa-tokenized 50-token completion conversion."""

    prompt_ids = trimming_tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = trimming_tokenizer.encode(decoded_full_text, add_special_tokens=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("decoded full text does not begin with the configured prompt")
    completion_ids = full_ids[len(prompt_ids) : len(prompt_ids) + completion_tokens]
    completion_text = trimming_tokenizer.decode(completion_ids)
    return prompt_text + completion_text, completion_ids, completion_text


def unconditional_trimmed_classifier_text(
    *,
    decoded_text: str,
    trimming_tokenizer: Any,
    completion_tokens: int,
) -> tuple[str, list[int]]:
    """Retokenize and truncate an unconditional MDLM sample for FK's classifier."""

    token_ids = trimming_tokenizer.encode(decoded_text, add_special_tokens=False)[
        :completion_tokens
    ]
    trimmed_text = trimming_tokenizer.decode(token_ids)
    return trimmed_text, token_ids


def unconditional_mdlm_prefix_classifier_text(
    *,
    token_ids: Sequence[int],
    decoder_tokenizer: Any,
    completion_tokens: int,
) -> tuple[str, list[int]]:
    """Decode an exact MDLM-token prefix for both reward and event scoring."""

    prefix_ids = [int(value) for value in token_ids[:completion_tokens]]
    return decoder_tokenizer.decode(prefix_ids, skip_special_tokens=True), prefix_ids


class UnconditionalToxicityFunctions:
    def __init__(
        self,
        *,
        generator_tokenizer: Any,
        evaluator: ToxicityEvaluator,
        reward_completion_tokens: int,
        event_completion_tokens: int,
        reward_max_length: int,
        event_max_length: int,
        event_text_scope: str,
        completion_trimming_tokenizer: Any,
        phi: str = "log_probability",
    ) -> None:
        self.generator_tokenizer = generator_tokenizer
        self.evaluator = evaluator
        self.reward_completion_tokens = reward_completion_tokens
        self.event_completion_tokens = event_completion_tokens
        self.reward_max_length = reward_max_length
        self.event_max_length = event_max_length
        self.event_text_scope = event_text_scope
        self.completion_trimming_tokenizer = completion_trimming_tokenizer
        self.potential = ToxicityPotential(phi)
        self.last_reward_rows: list[dict[str, Any]] = []
        self.last_reward_texts: list[str] = []
        self.last_event_rows: list[dict[str, Any]] = []
        self.last_event_texts: list[str] = []
        self.last_event_token_ids: list[list[int]] = []

    def reward(self, tokens: torch.Tensor) -> torch.Tensor:
        truncated = tokens[:, : self.reward_completion_tokens]
        self.last_reward_texts = self.generator_tokenizer.batch_decode(
            truncated.detach().cpu().tolist(), skip_special_tokens=True
        )
        self.last_reward_rows = self.evaluator.score_texts(
            self.last_reward_texts, max_length=self.reward_max_length
        )
        return torch.tensor(
            [self.potential.from_classifier_row(
                row, positive_label_index=getattr(self.evaluator, "positive_label_index", 1)
            ) for row in self.last_reward_rows],
            dtype=torch.float64,
            device=tokens.device,
        )

    def event(self, tokens: torch.Tensor) -> torch.Tensor:
        self.last_event_texts = []
        self.last_event_token_ids = []
        terminal_token_ids = tokens.detach().cpu().tolist()
        if self.event_text_scope == "first_generated_mdlm_tokens_only":
            for token_ids in terminal_token_ids:
                text, prefix_ids = unconditional_mdlm_prefix_classifier_text(
                    token_ids=token_ids,
                    decoder_tokenizer=self.generator_tokenizer,
                    completion_tokens=self.event_completion_tokens,
                )
                self.last_event_texts.append(text)
                self.last_event_token_ids.append(prefix_ids)
        elif (
            self.event_text_scope
            == "generated_text_retokenized_and_trimmed_with_roberta_large"
        ):
            decoded = self.generator_tokenizer.batch_decode(
                terminal_token_ids, skip_special_tokens=True
            )
            for text in decoded:
                trimmed_text, token_ids = unconditional_trimmed_classifier_text(
                    decoded_text=text,
                    trimming_tokenizer=self.completion_trimming_tokenizer,
                    completion_tokens=self.event_completion_tokens,
                )
                self.last_event_texts.append(trimmed_text)
                self.last_event_token_ids.append(token_ids)
        else:
            raise ValueError(f"unsupported event text scope: {self.event_text_scope}")
        if identical_classifier_inputs(
            self.last_reward_texts,
            first_max_length=self.reward_max_length,
            second_texts=self.last_event_texts,
            second_max_length=self.event_max_length,
        ):
            self.last_event_rows = self.last_reward_rows
        else:
            self.last_event_rows = self.evaluator.score_texts(
                self.last_event_texts, max_length=self.event_max_length
            )
        return torch.tensor(
            [row["is_event"] for row in self.last_event_rows],
            dtype=torch.bool,
            device=tokens.device,
        )
