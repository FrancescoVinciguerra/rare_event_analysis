"""Native MDLM SUBS reverse-ancestral kernel shared by direct and SMC runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(getattr(output, "logits", None), torch.Tensor):
        return output.logits
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("MDLM model did not return logits")


def loglinear_sigma(t: torch.Tensor, *, noise_epsilon: float = 1e-3) -> torch.Tensor:
    return -torch.log1p(-(1.0 - noise_epsilon) * t)


def sample_categorical_race(
    unnormalized_probabilities: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Official MDLM exponential-race categorical sampler with explicit RNG."""

    uniform = torch.rand(
        unnormalized_probabilities.shape,
        dtype=unnormalized_probabilities.dtype,
        device=unnormalized_probabilities.device,
        generator=generator,
    )
    exponential = 1e-10 - torch.log(uniform + 1e-10)
    return (unnormalized_probabilities / exponential).argmax(dim=-1)


def subs_log_probabilities(
    raw_logits: torch.Tensor,
    states: torch.Tensor,
    *,
    mask_token_id: int,
) -> torch.Tensor:
    if (
        raw_logits.ndim != states.ndim + 1
        or raw_logits.shape[:-1] != states.shape
    ):
        raise ValueError(
            "MDLM logits must have shape states.shape + (vocab_size,); "
            f"received logits shape {tuple(raw_logits.shape)} for states shape "
            f"{tuple(states.shape)}"
        )
    logits = raw_logits.float().clone()
    logits[..., mask_token_id] = -torch.inf
    logits = torch.log_softmax(logits, dim=-1)
    revealed = states != mask_token_id
    if bool(revealed.any().item()):
        revealed_rows = logits[revealed]
        revealed_rows.fill_(-torch.inf)
        revealed_token_ids = states[revealed].unsqueeze(1)
        revealed_rows.scatter_(1, revealed_token_ids, 0.0)
        logits[revealed] = revealed_rows
    return logits


@dataclass
class NativeMDLMSampler:
    model: Any
    mask_token_id: int = 50257
    predictor: str = "ddpm"
    noise_epsilon: float = 1e-3
    sampling_epsilon: float = 1e-5
    final_noise_removal: bool = True
    final_noise_removal_method: str = "stochastic_categorical"

    def __post_init__(self) -> None:
        if self.predictor not in {"ddpm", "ddpm_cache"}:
            raise ValueError("predictor must be ddpm or ddpm_cache")
        if not 0.0 < self.noise_epsilon < 1.0:
            raise ValueError("noise_epsilon must be in (0, 1)")
        if not 0.0 < self.sampling_epsilon < 1.0:
            raise ValueError("sampling_epsilon must be in (0, 1)")
        if self.final_noise_removal_method not in {
            "stochastic_categorical",
            "argmax",
        }:
            raise ValueError("invalid final noise removal method")
        self.nfe = 0

    def reset_nfe(self) -> None:
        self.nfe = 0

    def log_p_x0(self, states: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sigma = loglinear_sigma(t, noise_epsilon=self.noise_epsilon)
        output = self.model(input_ids=states, timesteps=sigma, return_dict=False)
        self.nfe += 1
        log_probabilities = subs_log_probabilities(
            extract_logits(output), states, mask_token_id=self.mask_token_id
        )
        if bool(torch.isnan(log_probabilities).any().item()) or bool(
            torch.isposinf(log_probabilities).any().item()
        ):
            raise FloatingPointError("MDLM produced NaN or +Inf log-probabilities")
        return log_probabilities

    def ddpm_step(
        self,
        states: torch.Tensor,
        *,
        t_scalar: float,
        dt: float,
        generator: torch.Generator,
        cached_p_x0: torch.Tensor | None = None,
        microbatch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch = states.shape[0]
        if microbatch_size is not None:
            if microbatch_size <= 0:
                raise ValueError("microbatch_size must be positive")
            if microbatch_size < batch:
                next_state_chunks: list[torch.Tensor] = []
                next_cache_chunks: list[torch.Tensor] = []
                cache_is_complete = True
                for start in range(0, batch, microbatch_size):
                    stop = min(start + microbatch_size, batch)
                    cache_chunk = (
                        None if cached_p_x0 is None else cached_p_x0[start:stop]
                    )
                    next_states, next_cache = self.ddpm_step(
                        states[start:stop],
                        t_scalar=t_scalar,
                        dt=dt,
                        generator=generator,
                        cached_p_x0=cache_chunk,
                        microbatch_size=None,
                    )
                    next_state_chunks.append(next_states)
                    if next_cache is None:
                        cache_is_complete = False
                    else:
                        next_cache_chunks.append(next_cache)
                combined_cache = (
                    torch.cat(next_cache_chunks, dim=0)
                    if cache_is_complete and next_cache_chunks
                    else None
                )
                return torch.cat(next_state_chunks, dim=0), combined_cache
        t = torch.full((batch,), t_scalar, dtype=torch.float32, device=states.device)
        s = torch.clamp(t - dt, min=0.0)
        move_t = 1.0 - torch.exp(-loglinear_sigma(t, noise_epsilon=self.noise_epsilon))
        move_s = 1.0 - torch.exp(-loglinear_sigma(s, noise_epsilon=self.noise_epsilon))
        if cached_p_x0 is None:
            p_x0 = self.log_p_x0(states, t).exp()
        else:
            p_x0 = cached_p_x0
        q = p_x0 * (move_t - move_s)[:, None, None]
        q[..., self.mask_token_id] = move_s[:, None]
        proposed = sample_categorical_race(q, generator=generator)
        revealed = states != self.mask_token_id
        next_states = torch.where(revealed, states, proposed)
        next_cache = p_x0 if self.predictor == "ddpm_cache" else None
        if self.predictor == "ddpm_cache" and not torch.equal(next_states, states):
            next_cache = None
        return next_states, next_cache

    def ddpm_step_with_x0_samples(
        self,
        states: torch.Tensor,
        *,
        t_scalar: float,
        dt: float,
        num_x0_samples: int,
        generator: torch.Generator,
        cached_p_x0: torch.Tensor | None = None,
        microbatch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Apply one native DDPM step and sample FK's intermediate texts.

        This matches the MDLM implementation in FK Steering: the model's
        current ``p(x_0 | x_t)`` is evaluated once, ``num_x0_samples``
        independent categorical predictions are drawn from it, already
        revealed tokens are copied from ``x_t``, and the ordinary stochastic
        DDPM transition is then sampled from the same model prediction.

        The returned samples have shape ``[batch, num_x0_samples, length]``.
        They are reward-only predictions and do not replace the propagated
        DDPM state.
        """

        if not isinstance(num_x0_samples, int) or num_x0_samples <= 0:
            raise ValueError("num_x0_samples must be a positive integer")
        batch = states.shape[0]
        if microbatch_size is not None:
            if microbatch_size <= 0:
                raise ValueError("microbatch_size must be positive")
            if microbatch_size < batch:
                next_state_chunks: list[torch.Tensor] = []
                next_cache_chunks: list[torch.Tensor] = []
                x0_sample_chunks: list[torch.Tensor] = []
                cache_is_complete = True
                for start in range(0, batch, microbatch_size):
                    stop = min(start + microbatch_size, batch)
                    cache_chunk = (
                        None if cached_p_x0 is None else cached_p_x0[start:stop]
                    )
                    next_states, next_cache, x0_samples = (
                        self.ddpm_step_with_x0_samples(
                            states[start:stop],
                            t_scalar=t_scalar,
                            dt=dt,
                            num_x0_samples=num_x0_samples,
                            generator=generator,
                            cached_p_x0=cache_chunk,
                            microbatch_size=None,
                        )
                    )
                    next_state_chunks.append(next_states)
                    x0_sample_chunks.append(x0_samples)
                    if next_cache is None:
                        cache_is_complete = False
                    else:
                        next_cache_chunks.append(next_cache)
                combined_cache = (
                    torch.cat(next_cache_chunks, dim=0)
                    if cache_is_complete and next_cache_chunks
                    else None
                )
                return (
                    torch.cat(next_state_chunks, dim=0),
                    combined_cache,
                    torch.cat(x0_sample_chunks, dim=0),
                )

        t = torch.full((batch,), t_scalar, dtype=torch.float32, device=states.device)
        s = torch.clamp(t - dt, min=0.0)
        move_t = 1.0 - torch.exp(-loglinear_sigma(t, noise_epsilon=self.noise_epsilon))
        move_s = 1.0 - torch.exp(-loglinear_sigma(s, noise_epsilon=self.noise_epsilon))
        if cached_p_x0 is None:
            p_x0 = self.log_p_x0(states, t).exp()
        else:
            p_x0 = cached_p_x0

        revealed = states != self.mask_token_id
        x0_samples = torch.stack(
            [
                torch.where(
                    revealed,
                    states,
                    sample_categorical_race(p_x0, generator=generator),
                )
                for _ in range(num_x0_samples)
            ],
            dim=1,
        )

        q = p_x0 * (move_t - move_s)[:, None, None]
        q[..., self.mask_token_id] = move_s[:, None]
        proposed = sample_categorical_race(q, generator=generator)
        next_states = torch.where(revealed, states, proposed)
        next_cache = p_x0 if self.predictor == "ddpm_cache" else None
        if self.predictor == "ddpm_cache" and not torch.equal(next_states, states):
            next_cache = None
        return next_states, next_cache, x0_samples

    def deterministic_completion(
        self,
        states: torch.Tensor,
        *,
        t_scalar: float,
        microbatch_size: int | None = None,
    ) -> torch.Tensor:
        batch = states.shape[0]
        if microbatch_size is not None:
            if microbatch_size <= 0:
                raise ValueError("microbatch_size must be positive")
            if microbatch_size < batch:
                return torch.cat(
                    [
                        self.deterministic_completion(
                            states[start : start + microbatch_size],
                            t_scalar=t_scalar,
                            microbatch_size=None,
                        )
                        for start in range(0, batch, microbatch_size)
                    ],
                    dim=0,
                )
        t = torch.full((batch,), t_scalar, dtype=torch.float32, device=states.device)
        predicted = self.log_p_x0(states, t).argmax(dim=-1)
        return torch.where(states != self.mask_token_id, states, predicted)

    def terminal_completion(
        self,
        states: torch.Tensor,
        *,
        t_scalar: float,
        generator: torch.Generator,
        microbatch_size: int | None = None,
    ) -> torch.Tensor:
        """Remove terminal masks using the decoder-configured base policy."""

        if self.final_noise_removal_method == "argmax":
            return self.deterministic_completion(
                states,
                t_scalar=t_scalar,
                microbatch_size=microbatch_size,
            )
        batch = states.shape[0]
        if microbatch_size is not None:
            if microbatch_size <= 0:
                raise ValueError("microbatch_size must be positive")
            if microbatch_size < batch:
                return torch.cat(
                    [
                        self.terminal_completion(
                            states[start : start + microbatch_size],
                            t_scalar=t_scalar,
                            generator=generator,
                            microbatch_size=None,
                        )
                        for start in range(0, batch, microbatch_size)
                    ],
                    dim=0,
                )
        t = torch.full((batch,), t_scalar, dtype=torch.float32, device=states.device)
        probabilities = self.log_p_x0(states, t).exp()
        sampled = sample_categorical_race(probabilities, generator=generator)
        return torch.where(states != self.mask_token_id, states, sampled)

    def initial_states(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        device: torch.device | str,
        fixed_prefix_token_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create the absorbing prior; unconditional mode has an empty prefix."""

        device = torch.device(device)
        states = torch.full(
            (batch_size, sequence_length),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        prefix = (
            torch.empty(0, dtype=torch.long, device=device)
            if fixed_prefix_token_ids is None
            else fixed_prefix_token_ids.to(device=device, dtype=torch.long)
        )
        if prefix.numel() >= sequence_length:
            raise ValueError("fixed prefix must leave at least one generated position")
        if prefix.numel():
            states[:, : prefix.numel()] = prefix
        return states, prefix

    def rollout_trajectory_suffix(
        self,
        checkpoint_states: torch.Tensor,
        *,
        start_step: int,
        num_steps: int,
        generator: torch.Generator,
        microbatch_size: int | None = None,
    ) -> torch.Tensor:
        """Roll out the unchanged base decoder from a real chain checkpoint.

        The returned tensor contains the checkpoint, every subsequent DDPM
        state, and the state after configured terminal noise removal.  Keeping
        the pre-terminal state is important: it makes this the complete path
        suffix whose base-kernel factors cancel in trajectory Metropolis-Hastings.
        """

        if checkpoint_states.ndim != 2:
            raise ValueError("checkpoint_states must have shape [batch, length]")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if not 0 <= start_step < num_steps:
            raise ValueError("start_step must be in [0, num_steps)")
        if self.predictor != "ddpm":
            raise ValueError(
                "trajectory rollback currently requires predictor='ddpm'; "
                "ddpm_cache has history-dependent cached state"
            )
        dt = (1.0 - self.sampling_epsilon) / num_steps
        times = torch.linspace(1.0, self.sampling_epsilon, num_steps + 1).tolist()
        states = checkpoint_states
        suffix = [states.clone()]
        for t_scalar in times[start_step:-1]:
            states, _ = self.ddpm_step(
                states,
                t_scalar=float(t_scalar),
                dt=dt,
                generator=generator,
                cached_p_x0=None,
                microbatch_size=microbatch_size,
            )
            suffix.append(states.clone())
        if self.final_noise_removal:
            states = self.terminal_completion(
                states,
                t_scalar=self.sampling_epsilon,
                generator=generator,
                microbatch_size=microbatch_size,
            )
        if bool((states == self.mask_token_id).any().item()):
            raise RuntimeError("trajectory rollout ended with [MASK] tokens")
        suffix.append(states.clone())
        return torch.stack(suffix, dim=1)

    def sample_trajectory(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        num_steps: int,
        fixed_prefix_token_ids: torch.Tensor | None,
        device: torch.device | str,
        generator: torch.Generator,
        microbatch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample and retain one complete native MDLM reverse trajectory."""

        states, prefix = self.initial_states(
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=device,
            fixed_prefix_token_ids=fixed_prefix_token_ids,
        )
        trajectory = self.rollout_trajectory_suffix(
            states,
            start_step=0,
            num_steps=num_steps,
            generator=generator,
            microbatch_size=microbatch_size,
        )
        expected_shape = (batch_size, num_steps + 2, sequence_length)
        if tuple(trajectory.shape) != expected_shape:
            raise RuntimeError(
                f"trajectory shape {tuple(trajectory.shape)} != {expected_shape}"
            )
        if prefix.numel() and not torch.equal(
            trajectory[:, -1, : prefix.numel()], prefix.expand(batch_size, -1)
        ):
            raise RuntimeError("fixed prefix changed during trajectory rollout")
        return trajectory, prefix

    @torch.inference_mode()
    def sample(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        num_steps: int,
        fixed_prefix_token_ids: torch.Tensor | None,
        device: torch.device | str,
        generator: torch.Generator,
    ) -> torch.Tensor:
        device = torch.device(device)
        states, prefix = self.initial_states(
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=device,
            fixed_prefix_token_ids=fixed_prefix_token_ids,
        )
        dt = (1.0 - self.sampling_epsilon) / num_steps
        times = torch.linspace(1.0, self.sampling_epsilon, num_steps + 1).tolist()
        cache = None
        for t_scalar in times[:-1]:
            states, cache = self.ddpm_step(
                states,
                t_scalar=float(t_scalar),
                dt=dt,
                generator=generator,
                cached_p_x0=cache,
            )
        if self.final_noise_removal:
            states = self.terminal_completion(
                states,
                t_scalar=self.sampling_epsilon,
                generator=generator,
            )
        if prefix.numel() and not torch.equal(
            states[:, : prefix.numel()], prefix.expand(batch_size, -1)
        ):
            raise RuntimeError("fixed prefix changed during native MDLM sampling")
        if bool((states == self.mask_token_id).any().item()):
            raise RuntimeError("terminal native MDLM samples still contain [MASK] tokens")
        return states
