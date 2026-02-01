"""Configuration dataclasses for pipeline stages."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from interpret_personas.utils import get_model_short_name


@dataclass
class GenerationConfig:
    """Configuration for response generation stage."""

    model_name: str
    roles_dir: Path
    general_questions_file: Path
    output_dir: Path
    question_mode: str = "general"  # "general" or "role_specific"
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.95
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    prompt_indices: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    model_short_name: str = field(init=False)

    def __post_init__(self):
        self.model_short_name = get_model_short_name(self.model_name)

    @classmethod
    def from_yaml(cls, path: Path) -> "GenerationConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        data["roles_dir"] = Path(data["roles_dir"])
        data["general_questions_file"] = Path(data["general_questions_file"])
        data["output_dir"] = Path(data["output_dir"])

        return cls(**data)

    def validate(self):
        """Validate configuration parameters."""
        if not self.roles_dir.exists():
            raise ValueError(f"Roles directory does not exist: {self.roles_dir}")
        if not self.roles_dir.is_dir():
            raise ValueError(f"Roles path is not a directory: {self.roles_dir}")
        if not self.general_questions_file.exists():
            raise ValueError(
                f"General questions file does not exist: {self.general_questions_file}"
            )
        if not self.general_questions_file.is_file():
            raise ValueError(
                f"General questions path is not a file: {self.general_questions_file}"
            )

        if self.question_mode not in ["general", "role_specific"]:
            raise ValueError(
                f"question_mode must be 'general' or 'role_specific': {self.question_mode}"
            )

        if self.max_model_len <= 0:
            raise ValueError(f"max_model_len must be positive: {self.max_model_len}")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1]: {self.gpu_memory_utilization}"
            )
        if not 0 <= self.temperature <= 2:
            raise ValueError(f"temperature must be in [0, 2]: {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1]: {self.top_p}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive: {self.max_tokens}")
        if not self.prompt_indices:
            raise ValueError("prompt_indices cannot be empty")
        if any(i < 0 for i in self.prompt_indices):
            raise ValueError(f"prompt_indices must be non-negative: {self.prompt_indices}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_output_dir(self) -> Path:
        """
        Get the full output directory path with question_mode and model_name.

        Returns:
            Path: {output_dir}/{question_mode}/{model_name}
        """
        # Extract model short name for path (e.g., "gemma-3-27b-it" from full path)
        model_short = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
        return self.output_dir / self.question_mode / model_short


@dataclass
class ExtractionConfig:
    """Configuration for SAE feature extraction stage."""

    model_name: str
    sae_release: str
    sae_id: str
    responses_dir: Path
    output_dir: Path
    token_selection: str = "response_only"

    @classmethod
    def from_yaml(cls, path: Path) -> "ExtractionConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        data["responses_dir"] = Path(data["responses_dir"])
        data["output_dir"] = Path(data["output_dir"])

        return cls(**data)

    def validate(self):
        """Validate configuration parameters."""
        if not self.responses_dir.exists():
            raise ValueError(f"Responses directory does not exist: {self.responses_dir}")
        if not self.responses_dir.is_dir():
            raise ValueError(f"Responses path is not a directory: {self.responses_dir}")

        if self.token_selection not in ["response_only", "all"]:
            raise ValueError(
                f"token_selection must be 'response_only' or 'all': {self.token_selection}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class AggregationConfig:
    """Configuration for feature aggregation stage."""

    features_dir: Path
    output_dir: Path

    @classmethod
    def from_yaml(cls, path: Path) -> "AggregationConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        data["features_dir"] = Path(data["features_dir"])
        data["output_dir"] = Path(data["output_dir"])

        return cls(**data)

    def validate(self):
        """Validate configuration parameters."""
        if not self.features_dir.exists():
            raise ValueError(f"Features directory does not exist: {self.features_dir}")
        if not self.features_dir.is_dir():
            raise ValueError(f"Features path is not a directory: {self.features_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
