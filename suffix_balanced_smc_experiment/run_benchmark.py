from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rea_llms.models import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    DEFAULT_PROMPT,
    DEFAULT_TOKENIZER_NAME,
    load_model_and_tokenizer,
)
from rea_llms.observables import create_observable  # noqa: E402
from suffix_balanced_smc_experiment.smc_sampler import (  # noqa: E402
    SuffixBalancedSMCConfig,
    parse_float_list,
    run_suffix_balanced_smc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark two implementations of the same suffix-balanced SMC kernel."
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=["parallelization", "metropolis-fast-path"],
        default="parallelization",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--num-particles", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--lambdas", default="0,-0.5")
    parser.add_argument("--mcmc-steps-per-level", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        args.tokenizer_name,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    observable = create_observable(
        "ari", mode="capped", cap=15.0, backend="exact_cached"
    )
    base_config = SuffixBalancedSMCConfig(
        num_particles=args.num_particles,
        max_new_tokens=args.max_new_tokens,
        lambdas=parse_float_list(args.lambdas),
        mcmc_steps_per_level=args.mcmc_steps_per_level,
        suffix_mode="deterministic",
        suffix_sizes=(args.max_new_tokens,),
        suffix_size_schedule=((args.max_new_tokens,),),
        batch_size=args.batch_size,
        seed=args.seed,
    )
    if args.benchmark_mode == "metropolis-fast-path":
        warmup_config = replace(
            base_config,
            num_particles=min(2, args.num_particles),
            mcmc_steps_per_level=min(1, args.mcmc_steps_per_level),
            batch_size=min(2, args.batch_size),
            balancing_function="metropolis",
            implementation="parallel",
        )
        run_suffix_balanced_smc(
            model,
            tokenizer,
            observable,
            prompt=args.prompt,
            config=replace(warmup_config, metropolis_fast_path=True),
        )
        run_suffix_balanced_smc(
            model,
            tokenizer,
            observable,
            prompt=args.prompt,
            config=replace(warmup_config, metropolis_fast_path=False),
        )
        general = run_suffix_balanced_smc(
            model,
            tokenizer,
            observable,
            prompt=args.prompt,
            config=replace(
                base_config,
                balancing_function="metropolis",
                implementation="parallel",
                metropolis_fast_path=False,
            ),
        )
        fast = run_suffix_balanced_smc(
            model,
            tokenizer,
            observable,
            prompt=args.prompt,
            config=replace(
                base_config,
                balancing_function="metropolis",
                implementation="parallel",
                metropolis_fast_path=True,
            ),
        )
        general_runtime = general.summary["runtime_total_seconds"]
        fast_runtime = fast.summary["runtime_total_seconds"]
        general_particles = [row["completion_token_ids"] for row in general.samples]
        fast_particles = [row["completion_token_ids"] for row in fast.samples]
        if general_particles != fast_particles:
            raise RuntimeError(
                "Fast and full direct-MH paths diverged at the same seed; benchmark aborted"
            )
        print(
            json.dumps(
                {
                    "runtime_total_seconds": {
                        "general_direct_mh": general_runtime,
                        "metropolis_fast_path": fast_runtime,
                    },
                    "particles_per_second": fast.summary["particles_per_second"],
                    "mean_seconds_per_level": fast.summary[
                        "mean_runtime_seconds_per_level"
                    ],
                    "cuda_peak_memory_bytes": fast.summary["cuda_peak_memory_bytes"],
                    "speedup_vs_general_direct_mh": general_runtime / fast_runtime,
                    "metropolis_fast_path": fast.summary["metropolis_fast_path"],
                    "matched_final_particles": True,
                },
                indent=2,
            )
        )
        return

    reference = run_suffix_balanced_smc(
        model,
        tokenizer,
        observable,
        prompt=args.prompt,
        config=replace(base_config, implementation="reference"),
    )
    parallel = run_suffix_balanced_smc(
        model,
        tokenizer,
        observable,
        prompt=args.prompt,
        config=replace(base_config, implementation="parallel"),
    )
    reference_runtime = reference.summary["runtime_total_seconds"]
    parallel_runtime = parallel.summary["runtime_total_seconds"]
    print(
        json.dumps(
            {
                "runtime_total_seconds": {
                    "reference": reference_runtime,
                    "parallel": parallel_runtime,
                },
                "particles_per_second": parallel.summary["particles_per_second"],
                "mean_seconds_per_level": parallel.summary[
                    "mean_runtime_seconds_per_level"
                ],
                "cuda_peak_memory_bytes": parallel.summary["cuda_peak_memory_bytes"],
                "speedup_vs_reference": reference_runtime / parallel_runtime,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
