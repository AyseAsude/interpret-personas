"""SAE model loading with raw HuggingFace transformers."""

import logging

import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_sae_model(
    model_name: str,
    sae_release: str,
    sae_id: str,
) -> tuple:
    """
    Load HuggingFace model and SAE for feature extraction.

    Args:
        model_name: HuggingFace model name
        sae_release: SAE release name (e.g., "gemma-scope-2-27b-it-res")
        sae_id: SAE ID (e.g., "layer_40_width_65k_l0_medium")

    Returns:
        Tuple of (model, sae, tokenizer)
    """
    logger.info(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info(f"Loading SAE: {sae_release}/{sae_id}")
    sae = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device="cuda",
    )

    logger.info("Model and SAE loaded successfully")
    return model, sae, tokenizer
