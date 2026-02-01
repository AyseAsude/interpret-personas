"""SAE model loading with auto-detection for bridge architecture."""

import logging

logger = logging.getLogger(__name__)


def load_sae_model(
    model_name: str,
    sae_release: str,
    sae_id: str,
) -> tuple:
    """
    Load SAE model with auto-detection for bridge vs hooked transformer.

    For Gemma 3 models, uses SAETransformerBridge (beta feature).
    For other models, uses HookedTransformer + SAE separately.

    Models are loaded on CUDA.

    Args:
        model_name: HuggingFace model name
        sae_release: SAE release name (e.g., "gemma-scope-27b-pt-res")
        sae_id: SAE ID (e.g., "layer_20/width_16k/average_l0_82")

    Returns:
        Tuple of (model, sae, tokenizer)
    """
    needs_bridge = "gemma-3" in model_name.lower()

    if needs_bridge:
        logger.info(f"Detected Gemma 3 model, using SAETransformerBridge for {model_name}")
        return _load_with_bridge(model_name, sae_release, sae_id)
    else:
        logger.info(f"Using standard HookedTransformer for {model_name}")
        return _load_with_hooked_transformer(model_name, sae_release, sae_id)


def _load_with_bridge(
    model_name: str,
    sae_release: str,
    sae_id: str,
) -> tuple:
    """
    Load model using SAETransformerBridge (for Gemma 3 models).

    Args:
        model_name: HuggingFace model name
        sae_release: SAE release name
        sae_id: SAE ID

    Returns:
        Tuple of (model, sae, tokenizer)
    """
    from sae_lens import SAE
    from sae_lens.analysis.sae_transformer_bridge import SAETransformerBridge

    logger.info(f"Loading {model_name} with SAETransformerBridge...")
    logger.info(f"SAE: {sae_release}/{sae_id}")

    model = SAETransformerBridge.boot_transformers(
        model_name,
        device="cuda",
    )

    sae = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device="cuda",
    )

    tokenizer = model.tokenizer

    logger.info("Model and SAE loaded successfully with bridge")

    return model, sae, tokenizer


def _load_with_hooked_transformer(
    model_name: str,
    sae_release: str,
    sae_id: str,
) -> tuple:
    """
    Load model using HookedTransformer + SAE separately.

    Args:
        model_name: HuggingFace model name
        sae_release: SAE release name
        sae_id: SAE ID

    Returns:
        Tuple of (model, sae, tokenizer)
    """
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    logger.info(f"Loading {model_name} with HookedTransformer...")

    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        device="cuda",
    )

    logger.info(f"Loading SAE: {sae_release}/{sae_id}")

    sae = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device="cuda",
    )

    tokenizer = model.tokenizer

    logger.info("Model and SAE loaded successfully")

    return model, sae, tokenizer
