#!/usr/bin/env python3
"""Extract SAE features from responses.

This script extracts SAE features from generated responses using a specified SAE model.
Features are aggregated per-response (mean and max across tokens) and saved as compressed
.npz files. Metadata is saved separately as JSONL (no feature data in JSONL).

Usage:
    python pipeline/2_extract_features.py --config configs/extraction.yaml [--skip-existing]

Output:
    outputs/features/{model_name}_{sae_id}/{role_name}.npz   (binary features)
    outputs/features/{model_name}_{sae_id}/{role_name}.jsonl  (metadata only)
"""

import argparse
from pathlib import Path

import jsonlines
import numpy as np

from interpret_personas.config import ExtractionConfig
from interpret_personas.extraction.sae_loader import load_sae_model
from interpret_personas.extraction.feature_extractor import FeatureExtractor
from interpret_personas.utils import get_completed_roles, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Extract SAE features from responses")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to extraction config YAML file",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip roles with existing output files",
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        help="Only extract for these roles (matching .jsonl filenames without extension)",
    )
    args = parser.parse_args()

    # Load and validate config
    logger = setup_logging("extraction")
    logger.info(f"Loading config from {args.config}")

    config = ExtractionConfig.from_yaml(args.config)
    config.validate()

    logger.info(f"Model: {config.model_name}")
    logger.info(f"SAE: {config.sae_release}/{config.sae_id}")
    logger.info(f"Token selection: {config.token_selection}")
    logger.info(f"Responses directory: {config.responses_dir}")
    logger.info(f"Output directory: {config.output_dir}")

    logger.info("Loading SAE model...")
    model, sae, tokenizer = load_sae_model(
        model_name=config.model_name,
        sae_release=config.sae_release,
        sae_id=config.sae_id,
    )

    extractor = FeatureExtractor(
        model=model,
        sae=sae,
        tokenizer=tokenizer,
    )

    # Get completed roles (for resumption)
    completed = get_completed_roles(config.output_dir) if args.skip_existing else set()
    if completed:
        logger.info(f"Skipping {len(completed)} existing roles")

    role_filter = set(args.roles) if args.roles else None
    if role_filter:
        response_files = [config.responses_dir / f"{r}.jsonl" for r in role_filter]
        response_files = [f for f in response_files if f.exists()]
        missing = role_filter - {f.stem for f in response_files}
        if missing:
            logger.warning(f"Response files not found for: {sorted(missing)}")
    else:
        response_files = list(config.responses_dir.glob("*.jsonl"))
    logger.info(f"Found {len(response_files)} role files to process")

    total_roles = 0
    processed_roles = 0

    for response_file in sorted(response_files):
        role_name = response_file.stem
        total_roles += 1

        if role_name in completed:
            logger.info(f"Skipping {role_name} (already exists)")
            continue

        logger.info(f"Processing {role_name}...")

        # Read responses
        with jsonlines.open(response_file, "r") as reader:
            responses = list(reader)

        logger.info(f"Extracting features from {len(responses)} responses...")

        mean_list = []
        max_list = []
        metadata_records = []

        for i, resp in enumerate(responses):

            aggregated = extractor.extract_from_conversation(
                conversation=resp["conversation"],
                token_selection=config.token_selection,
            )

            mean_list.append(aggregated["mean"])
            max_list.append(aggregated["max"])

            metadata_records.append({
                "role_name": role_name,
                "system_prompt": resp["system_prompt"],
                "question_id": resp["question_id"],
                "question": resp["question"],
                "response_text": resp["response"],
                "metadata": {
                    "model_name": config.model_name,
                    "sae_release": config.sae_release,
                    "sae_id": config.sae_id,
                    "token_selection": config.token_selection,
                },
            })

            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(responses)} responses")

        npz_file = config.output_dir / f"{role_name}.npz"
        np.savez_compressed(
            npz_file,
            mean_features=np.array(mean_list),   # [n_responses, sae_dim]
            max_features=np.array(max_list),      # [n_responses, sae_dim]
        )

        metadata_file = config.output_dir / f"{role_name}.jsonl"
        with jsonlines.open(metadata_file, "w") as writer:
            writer.write_all(metadata_records)

        logger.info(f"Saved {len(metadata_records)} responses to {npz_file} + {metadata_file}")
        processed_roles += 1

    logger.info(f"Processed {processed_roles}/{total_roles} roles")


if __name__ == "__main__":
    main()
