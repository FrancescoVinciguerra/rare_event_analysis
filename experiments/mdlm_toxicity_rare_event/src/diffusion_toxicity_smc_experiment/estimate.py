"""Run one RollBack SMC estimate for a toxicity threshold."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .configuration import decoder_id, evaluator_id, load_yaml, validate_benchmark_config
from .exploratory_population import population
from .pipeline import Context, audit_population


BUNDLED_CONFIG = Path(__file__).with_name("estimator.json")
CHECKOUT_CONFIG = Path(__file__).resolve().parents[2] / "configs/mdlm_owt_unconditional_estimation_100tok_trajectory_mh.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=.995, help="Estimate P(toxicity score >= threshold)")
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--tilt", type=float, default=1., help="Terminal tilt, a positive multiple of 0.5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--offline", action="store_true", help="Load the pinned models from the local cache")
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    if args.particles < 1 or not 0 <= args.seed < 2**63:
        parser.error("--particles must be positive and --seed must be in [0, 2**63)")
    if not math.isfinite(args.tilt) or args.tilt <= 0 or not (2 * args.tilt).is_integer():
        parser.error("--tilt must be a positive multiple of 0.5")

    config = load_yaml(BUNDLED_CONFIG if BUNDLED_CONFIG.is_file() else CHECKOUT_CONFIG)
    config["runtime"]["device"] = args.device
    schedule = [k / 2 for k in range(1, int(2 * args.tilt) + 1)]
    smc = config["smc"]
    smc.update(num_particles=args.particles, tilted_lambdas=[args.tilt],
               tilted_component_weights_a=[1.], phi="logit")
    smc["tilted_sampler"].update(num_tempering_levels=len(schedule), mcmc_steps_per_level=4)
    config["_exploration"] = {"schedule": schedule, "moves_per_level": 4}
    validate_benchmark_config(config)

    context = Context(config, local_files_only=args.offline)
    study = {"thresholds": [args.threshold], "phi": "logit"}
    result = population(context, config, study, args.seed)
    audit_population(result, config)
    correction = result["corrections"][str(args.threshold)]
    print(json.dumps({
        "threshold": args.threshold,
        "p_hat_smc": correction["smc"],
        "event_particles": correction["event_particles"],
        "schedule": [0., *schedule],
        "moves_per_level": 4,
        "device": context.device.type,
        "decoder_id": decoder_id(config),
        "evaluator_id": evaluator_id(config),
        **result["summary"],
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
