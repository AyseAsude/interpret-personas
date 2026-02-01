"""vLLM-based response generation."""

import logging

logger = logging.getLogger(__name__)


class VLLMGenerator:
    """
    Generator for batch inference using vLLM.

    Adapted from assistant-axis for the interpret-personas pipeline.

    Example:
        config = GenerationConfig.from_yaml("configs/generation.yaml")
        generator = VLLMGenerator(config)
        results = generator.generate_for_role(instructions, questions, prompt_indices)
    """

    def __init__(self, config):
        """
        Initialize vLLM generator and load model.

        Args:
            config: GenerationConfig instance
        """
        self.config = config

        from vllm import LLM, SamplingParams

        logger.info(f"Loading vLLM model: {config.model_name}")

        self.llm = LLM(
            model=config.model_name,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            trust_remote_code=True,
        )

        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
        )

        logger.info("Model loaded successfully")

    def generate_batch(
        self,
        conversations: list[list[dict[str, str]]],
    ) -> list[str]:
        """
        Generate responses for a batch of conversations.

        Args:
            conversations: List of conversations (each is a list of message dicts)

        Returns:
            List of generated response texts
        """
        tokenizer = self.llm.get_tokenizer()

        prompts = []
        for conv in conversations:
            prompt = tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

        logger.info(f"Running batch inference for {len(prompts)} prompts...")
        outputs = self.llm.generate(prompts, self.sampling_params)

        responses = [output.outputs[0].text for output in outputs]
        return responses

    def generate_for_role(
        self,
        instructions: list[str],
        questions: list[dict],
        prompt_indices: list[int] | None = None,
    ) -> list[dict]:
        """
        Generate responses for a role across all instruction variants and questions.

        Args:
            instructions: List of system prompt variants
            questions: List of question dicts with "question" and "id" fields
            prompt_indices: Which instruction indices to use (default: config.prompt_indices)

        Returns:
            List of result dicts with response, system_prompt, question_id, etc.
        """
        if prompt_indices is None:
            prompt_indices = self.config.prompt_indices

        # Build all conversations
        all_conversations = []
        all_metadata = []

        for prompt_idx in prompt_indices:
            if prompt_idx >= len(instructions):
                logger.warning(
                    f"Skipping prompt_idx {prompt_idx}: only {len(instructions)} instructions"
                )
                continue

            instruction = instructions[prompt_idx]

            for question_dict in questions:
                question_text = question_dict["question"]
                question_id = question_dict["id"]

                conversation = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": question_text},
                ]

                all_conversations.append(conversation)
                all_metadata.append({
                    "system_prompt": instruction,
                    "question_id": question_id,
                    "question": question_text,
                })

        if not all_conversations:
            return []

        responses = self.generate_batch(all_conversations)

        results = []
        for conv, meta, response in zip(all_conversations, all_metadata, responses):
            result = {
                "system_prompt": meta["system_prompt"],
                "question_id": meta["question_id"],
                "question": meta["question"],
                "response": response,
                "conversation": conv + [{"role": "assistant", "content": response}],
            }
            results.append(result)

        return results
