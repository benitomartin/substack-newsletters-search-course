from enum import Enum

from pydantic import BaseModel, Field


# OpenRouter priority sort options
class ProviderSort(str, Enum):
    latency = "latency"


class ModelConfig(BaseModel):
    # The "entry point" model — required by OpenRouter API
    primary_model: str = Field(default="", description="The initial model requested")
    # Optional fallback / routing models
    candidate_models: list[str] = Field(
        default_factory=list, description="List of candidate models for fallback or routing"
    )
    provider_sort: ProviderSort = Field(
        default=ProviderSort.latency, description="How to sort candidate models"
    )
    stream: bool = Field(default=False, description="Whether to stream responses")
    max_completion_tokens: int = Field(
        default=5000, description="Maximum number of tokens for completion"
    )
    temperature: float = Field(default=0.0, description="Sampling temperature")


class ModelRegistry(BaseModel):
    models: dict[str, ModelConfig] = Field(default_factory=dict)

    def get_config(self, provider: str) -> ModelConfig:
        """Retrieve the ModelConfig for the specified provider.

        Args:
            provider (str): The name of the provider.

        Returns:resp
            ModelConfig: The ModelConfig instance for the specified provider.

        Raises:
            ValueError: If the provider is not found in the registry.
        """
        provider_lower = provider.lower()
        if provider_lower not in self.models:
            raise ValueError(f"ModelConfig not found for provider: {provider}")
        return self.models[provider_lower]


# -----------------------
# Default registry
# -----------------------

# Default ModelConfigs for models
# OpenRouter models show low latency and are highly ranked by OpenRouter

MODEL_REGISTRY = ModelRegistry(
    models={
        "openrouter": ModelConfig(
            primary_model="openai/gpt-oss-20b:free",
            candidate_models=[
                # "meta-llama/llama-4-scout:free",
                "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
                # "meta-llama/llama-3.3-8b-instruct:free",
                # "openai/gpt-oss-20b:free",
                # "openai/gpt-oss-120b:free",
                "nvidia/nemotron-nano-9b-v2:free",
            ],
        ),
        # "openai": ModelConfig(primary_model="gpt-4o-mini"),
        "huggingface": ModelConfig(primary_model="deepseek-ai/DeepSeek-R1"),
    }
)

# MODELS WITH LOGPROBS SUPPORT

# deepseek/deepseek-r1-0528-qwen3-8b:free
# mistralai/mistral-small-3.2-24b-instruct:free
