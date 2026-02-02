"""SAE feature extraction with token selection support."""

import logging
import re
from functools import partial

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _gather_acts_hook(module, input, output, cache, key):
    """Store layer output activations in cache dict."""
    hidden_states = output[0] if isinstance(output, tuple) else output
    cache[key] = hidden_states.detach()


def _get_decoder_layer(model, layer_idx):
    """Resolve the decoder layer module for different model architectures."""
    inner = model.model
    # Gemma 3 nests layers under language_model
    if hasattr(inner, "language_model"):
        return inner.language_model.layers[layer_idx]
    return inner.layers[layer_idx]


def _gather_residual_activations(model, target_layer, inputs):
    """Run forward pass and capture residual stream at target_layer via hook."""
    cache = {}
    handle = _get_decoder_layer(model, target_layer).register_forward_hook(
        partial(_gather_acts_hook, cache=cache, key="resid_post")
    )
    try:
        with torch.no_grad():
            model(inputs)
    finally:
        handle.remove()
    return cache["resid_post"]


class FeatureExtractor:
    """
    Extract SAE features from text with token selection support.

    Supports two token selection modes:
    - "response_only": Extract features only from assistant response tokens
    - "all": Extract features from all tokens in the conversation
    """

    def __init__(self, model, sae, tokenizer):
        self.model = model
        self.sae = sae
        self.tokenizer = tokenizer
        self.device = "cuda"
        self._target_layer = self._parse_layer_index(sae.cfg.metadata.hook_name)

    @staticmethod
    def _parse_layer_index(hook_name: str) -> int:
        """Extract layer index from SAE hook name (e.g. 'blocks.40.hook_resid_post' -> 40)."""
        match = re.search(r"(?:blocks|layers)\.(\d+)\.", hook_name)
        if not match:
            raise ValueError(f"Cannot parse layer index from hook name: {hook_name}")
        return int(match.group(1))

    def extract_from_conversation(
        self,
        conversation: list[dict[str, str]],
        token_selection: str = "response_only",
    ) -> dict[str, np.ndarray]:
        """
        Extract SAE features from a conversation, aggregated across tokens.

        Args:
            conversation: List of message dicts
            token_selection: "response_only" or "all"

        Returns:
            Dict with "mean" and "max" aggregated feature vectors of shape (n_features,)
        """
        if token_selection not in ["response_only", "all"]:
            raise ValueError(f"token_selection must be 'response_only' or 'all', got {token_selection}")

        formatted_text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

        tokens = self.tokenizer.encode(formatted_text, return_tensors="pt", add_special_tokens=False).to(self.device)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        activations = _gather_residual_activations(self.model, self._target_layer, tokens)
        sae_features = self.sae.encode(activations.to(self.sae.dtype).to(self.sae.device))

        features = sae_features.squeeze(0).cpu().numpy()

        if token_selection == "response_only":
            response_start_idx = self._find_response_start(conversation)
            if response_start_idx is not None:
                features = features[response_start_idx:]
            else:
                logger.warning("Could not find response start, using all tokens")

        return {
            "mean": np.mean(features, axis=0),
            "max": np.max(features, axis=0),
        }

    def _find_response_start(
        self,
        conversation: list[dict[str, str]],
    ) -> int:
        """Find the token index where the assistant response starts."""
        if conversation[-1]["role"] != "assistant":
            logger.warning("No assistant message at end of conversation")
            return None

        prompt_only = conversation[:-1]

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_only,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompt_tokens = self.tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False).to(self.device)
        if prompt_tokens.dim() == 1:
            prompt_tokens = prompt_tokens.unsqueeze(0)

        return prompt_tokens.shape[1]

    def extract_batch(
        self,
        conversations: list[list[dict[str, str]]],
        token_selection: str = "response_only",
    ) -> list[dict[str, np.ndarray]]:
        """
        Extract features from a batch of conversations.

        Note: Processes one at a time (no batching) to handle variable-length responses.
        """
        features_list = []
        for conversation in conversations:
            features = self.extract_from_conversation(conversation, token_selection)
            features_list.append(features)
        return features_list
