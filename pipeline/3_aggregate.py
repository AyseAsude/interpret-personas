#!/usr/bin/env python3
"""Aggregate features per-role.

This script loads per-response .npz files from step 2 and aggregates them
to produce per-role feature vectors.

Usage:
    python pipeline/3_aggregate.py --config configs/aggregation.yaml

Output:
    outputs/aggregated/{strategy}/per_role.npz
"""

import argparse
from pathlib import Path

import numpy as np

from interpret_personas.config import AggregationConfig
from interpret_personas.aggregation.aggregator import AGGREGATION_FUNCTIONS
from interpret_personas.utils import ensure_dir, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Aggregate features at role level")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to aggregation config YAML file",
    )
    args = parser.parse_args()

    # Load and validate config
    logger = setup_logging("aggregation")
    logger.info(f"Loading config from {args.config}")

    config = AggregationConfig.from_yaml(args.config)
    config.validate()

    logger.info(f"Features directory: {config.features_dir}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Aggregation strategies: {list(AGGREGATION_FUNCTIONS.keys())}")

    # Get feature files (.npz from step 2)
    npz_files = sorted(config.features_dir.glob("*.npz"))
    logger.info(f"Found {len(npz_files)} role .npz files to process")

    # Aggregate per-role for each strategy
    for strategy_name, func in AGGREGATION_FUNCTIONS.items():
        logger.info(f"Aggregating with strategy: {strategy_name}")

        role_names = []
        role_vectors = []
        strategy_key = f"{strategy_name}_features"  # "mean_features" or "max_features"

        for npz_file in npz_files:
            role_name = npz_file.stem
            data = np.load(npz_file)

            if strategy_key not in data:
                logger.warning(f"Key '{strategy_key}' not found in {npz_file}, skipping")
                continue

            # data[strategy_key] shape: [n_responses, sae_dim]
            response_features = data[strategy_key]
            n_responses = response_features.shape[0]

            # Aggregate across responses for this role
            role_vector = func(response_features)  # [sae_dim]

            role_names.append(role_name)
            role_vectors.append(role_vector)

            logger.info(f"  {role_name}: {n_responses} responses -> 1 role vector")

        if not role_vectors:
            logger.warning(f"No role vectors for strategy '{strategy_name}', skipping")
            continue

        # Save all role vectors for this strategy
        output_dir = ensure_dir(config.output_dir / strategy_name)
        output_file = output_dir / "per_role.npz"

        np.savez_compressed(
            output_file,
            features=np.array(role_vectors),       # [n_roles, sae_dim]
            role_names=np.array(role_names),
        )

        logger.info(f"  Saved {len(role_vectors)} role vectors to {output_file}")

    logger.info("Done! Aggregation complete.")


if __name__ == "__main__":
    main()
