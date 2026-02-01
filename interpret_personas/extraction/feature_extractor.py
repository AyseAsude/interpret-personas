"""SAE feature extraction with token selection support."""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract SAE features from text with token selection support.

    Supports two token selection modes:
    - "response_only": Extract features only from assistant response tokens
    - "all": Extract features from all tokens in the conversation
    """

    def __init__(self, model, sae, tokenizer):
        """
        Initialize feature extractor.

        Args:
            model: HookedTransformer or SAETransformerBridge model
            sae: SAE model
            tokenizer: Model tokenizer
        """
        self.model = model
        self.sae = sae
        self.tokenizer = tokenizer
        self.device = "cuda"

        # Bridge resolves alias hook names (e.g. hook_resid_post -> hook_out) to
        # canonical names, and cache keys use the canonical form. Resolve here so
        # the cache lookup matches.
        hook_name = sae.cfg.metadata.hook_name
        if hasattr(model, "_resolve_hook_name"):
            hook_name = model._resolve_hook_name(hook_name)
        self._sae_acts_key = f"{hook_name}.hook_sae_acts_post"

    def extract_from_conversation(
        self,
        conversation: list[dict[str, str]],
        token_selection: str = "response_only",
    ) -> dict[str, np.ndarray]:
        """
        Extract SAE features from a conversation, aggregated across tokens.

        Args:
            conversation: List of message dicts (e.g., [{"role": "user", "content": "..."}])
            token_selection: "response_only" or "all"

        Returns:
            Dict with aggregated feature vectors:
                "mean": mean-pooled vector of shape (n_features,)
                "max": max-pooled vector of shape (n_features,)
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

        # Run forward pass with SAE attached at its hook point
        with torch.no_grad():
            _, cache = self.model.run_with_cache_with_saes(
                tokens, saes=[self.sae], use_error_term=True
            )

        # Get SAE feature activations from cache
        sae_features = cache[self._sae_acts_key]  # Shape: (batch=1, seq_len, n_features)

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
        """
        Find the token index where the assistant response starts.

        Strategy: Apply chat template to conversation without assistant response,
        tokenize, and return the length as the response start index.

        Args:
            conversation: Full conversation including assistant response

        Returns:
            Index where response starts, or None if not found
        """
        # Remove the last assistant message to get the prompt up to the response
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

        # The response starts at the end of the prompt tokens
        response_start_idx = prompt_tokens.shape[1]

        return response_start_idx

    def extract_batch(
        self,
        conversations: list[list[dict[str, str]]],
        token_selection: str = "response_only",
    ) -> list[dict[str, np.ndarray]]:
        """
        Extract features from a batch of conversations.

        Note: Processes one at a time (no batching) to handle variable-length responses.

        Args:
            conversations: List of conversations
            token_selection: "response_only" or "all"

        Returns:
            List of dicts, each with "mean" and "max" aggregated vectors
        """
        features_list = []

        for conversation in conversations:
            features = self.extract_from_conversation(conversation, token_selection)
            features_list.append(features)

        return features_list
