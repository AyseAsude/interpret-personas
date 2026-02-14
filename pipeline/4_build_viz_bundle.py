#!/usr/bin/env python3
"""Build visualization bundle for the SAE Persona Feature Explorer.

Usage:
    python pipeline/4_build_viz_bundle.py --config configs/visualization.yaml

Output:
    outputs/viz_bundle/{dataset_name}/bundle.json
    outputs/viz_bundle/{dataset_name}/features.csv
"""

import argparse
from pathlib import Path

from interpret_personas.config import VisualizationConfig
from interpret_personas.utils import setup_logging
from interpret_personas.visualization import build_visualization_bundle


def main():
    parser = argparse.ArgumentParser(description="Build visualization data bundle")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to visualization config YAML file",
    )
    args = parser.parse_args()

    logger = setup_logging("visualization")
    logger.info(f"Loading config from {args.config}")

    config = VisualizationConfig.from_yaml(args.config)
    config.validate()

    logger.info(f"Dataset name: {config.dataset_name}")
    logger.info(f"Aggregated file: {config.aggregated_file}")
    logger.info(f"Features directory: {config.features_dir}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Aggregation strategy: {config.strategy}")
    logger.info(f"Question-centering: {config.question_centering}")
    logger.info(f"Top-K selected features: {config.top_k}")

    build_visualization_bundle(config=config, logger=logger)
    logger.info("Done! Visualization bundle build complete.")


if __name__ == "__main__":
    main()
