#!/usr/bin/env python3
"""Generate responses using vLLM.

This script generates model responses for all roles using vLLM batch inference.

Usage:
    python pipeline/1_generate.py --config configs/generation.yaml [--skip-existing]

Output:
    outputs/responses/{question_mode}/{model_name}/{role_name}.jsonl
"""

import argparse
from pathlib import Path

import jsonlines

from interpret_personas.config import GenerationConfig
from interpret_personas.generation.data_loader import (
    iter_roles,
    load_general_questions,
)
from interpret_personas.generation.generator import VLLMGenerator
from interpret_personas.utils import get_completed_roles, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Generate role-based responses using vLLM")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to generation config YAML file",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip roles with existing output files",
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        help="Only generate for these roles (matching .json filenames without extension)",
    )
    args = parser.parse_args()

    logger = setup_logging("generation")
    logger.info(f"Loading config from {args.config}")

    config = GenerationConfig.from_yaml(args.config)
    config.validate()

    output_dir = config.get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model: {config.model_name}")
    logger.info(f"Question mode: {config.question_mode}")
    logger.info(f"Roles directory: {config.roles_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Prompt indices: {config.prompt_indices}")

    generator = VLLMGenerator(config)

    logger.info(f"Loading general questions from {config.general_questions_file}")
    general_questions = load_general_questions(config.general_questions_file)
    logger.info(f"Loaded {len(general_questions)} general questions")

    # Get completed roles (for resumption)
    completed = get_completed_roles(output_dir, extension="jsonl") if args.skip_existing else set()
    if completed:
        logger.info(f"Skipping {len(completed)} existing roles")

    total_roles = 0
    processed_roles = 0

    role_filter = set(args.roles) if args.roles else None
    if role_filter:
        logger.info(f"Filtering to {len(role_filter)} roles: {sorted(role_filter)}")

    for role_data in iter_roles(config.roles_dir, general_questions):
        total_roles += 1

        if role_filter and role_data.name not in role_filter:
            continue

        if role_data.name in completed:
            logger.info(f"Skipping {role_data.name} (already exists)")
            continue

        # Replace {model_name} placeholder in default role instructions
        if role_data.name == "default":
            role_data.instructions = [
                inst.replace("{model_name}", config.model_short_name)
                for inst in role_data.instructions
            ]

        if config.question_mode == "general":
            questions = role_data.general_questions
        elif config.question_mode == "role_specific":
            questions = role_data.role_questions
        else:
            raise ValueError(f"Invalid question_mode: {config.question_mode}")

        logger.info(
            f"Generating for {role_data.name} "
            f"({len(role_data.instructions)} instructions × {len(questions)} questions)"
        )

        results = generator.generate_for_role(
            instructions=role_data.instructions,
            questions=questions,
            prompt_indices=config.prompt_indices,
        )

        if not results:
            logger.warning(f"No results generated for {role_data.name}")
            continue

        output_file = output_dir / f"{role_data.name}.jsonl"
        with jsonlines.open(output_file, "w") as writer:
            for result in results:
                writer.write(result)

        logger.info(f"Saved {len(results)} responses to {output_file}")
        processed_roles += 1

    logger.info(f"Processed {processed_roles}/{total_roles} roles")


if __name__ == "__main__":
    main()
