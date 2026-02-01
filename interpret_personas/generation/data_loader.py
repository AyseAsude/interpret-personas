"""Data loading utilities for role-based generation."""

import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from typing import Any

import jsonlines


@dataclass
class RoleData:
    """Container for role data."""

    name: str
    instructions: list[str]
    general_questions: list[dict[str, Any]]  # List of {"question": str, "id": int}
    role_questions: list[dict[str, Any]]  # List of {"question": str, "id": int}, empty if none


def load_general_questions(questions_file: Path) -> list[dict[str, Any]]:
    """
    Load general questions from JSONL file.

    Args:
        questions_file: Path to extraction_questions.jsonl

    Returns:
        List of question dicts with "question" and "id" fields
    """
    questions = []
    with jsonlines.open(questions_file, "r") as reader:
        for entry in reader:
            questions.append({"question": entry["question"], "id": entry["id"]})

    return questions


def load_role(role_file: Path, general_questions: list[dict[str, Any]]) -> RoleData:
    """
    Load role from JSON file.

    Returns both general questions and role-specific questions.

    Args:
        role_file: Path to role JSON file
        general_questions: General questions (same for all roles)

    Returns:
        RoleData with name, instructions, general_questions, and role_questions
    """
    with open(role_file, "r") as f:
        data = json.load(f)

    role_name = role_file.stem

    raw_instructions = data.get("instruction", [])
    instructions = [inst.get("pos", "") for inst in raw_instructions]
    raw_role_questions = data.get("questions", [])
    if raw_role_questions:
        role_questions = [{"question": q, "id": i} for i, q in enumerate(raw_role_questions)]
    else:
        role_questions = []

    return RoleData(
        name=role_name,
        instructions=instructions,
        general_questions=general_questions,
        role_questions=role_questions,
    )


def iter_roles(
    roles_dir: Path, general_questions: list[dict[str, Any]]
) -> Iterator[RoleData]:
    """
    Iterate over all role JSON files in directory.

    Raises exceptions on malformed data for fail-fast behavior.

    Args:
        roles_dir: Directory containing role JSON files
        general_questions: General questions (same for all roles)

    Yields:
        RoleData for each role file
    """
    for role_file in sorted(roles_dir.glob("*.json")):
        role_data = load_role(role_file, general_questions)
        yield role_data
