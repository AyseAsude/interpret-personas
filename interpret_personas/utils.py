"""Utility functions for the pipeline."""

import logging
from pathlib import Path


def setup_logging(name: str) -> logging.Logger:
    """
    Setup consistent logging format.

    Args:
        name: Logger name

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Only add handler if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def ensure_dir(path: Path) -> Path:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path to create

    Returns:
        The input path
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


MODEL_SHORT_NAMES = {
    "google/gemma-2-27b-it": "Gemma",
}


def get_model_short_name(model_name: str) -> str:
    """
    Get short display name for a HuggingFace model.

    Args:
        model_name: Full HuggingFace model name (e.g., "google/gemma-3-27b-it")

    Returns:
        Short name (e.g., "Gemma")
    """
    if model_name in MODEL_SHORT_NAMES:
        return MODEL_SHORT_NAMES[model_name]
    # Fallback: use first part of model name after '/'
    return model_name.split("/")[-1].split("-")[0].capitalize()


def get_completed_roles(output_dir: Path) -> set[str]:
    """
    Return set of role names with existing output files.

    Args:
        output_dir: Directory containing output files

    Returns:
        Set of role names (without extension)
    """
    if not output_dir.exists():
        return set()

    return {f.stem for f in output_dir.glob("*.npz")}
