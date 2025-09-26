import os
from collections.abc import AsyncGenerator
from typing import Any

import opik
from openai import AsyncOpenAI
from opik.integrations.openai import track_openai

from src.api.models.provider_models import ModelConfig
from src.api.services.providers.utils.messages import build_messages
from src.config import settings
from src.utils.logger_util import setup_logging

logger = setup_logging()


# -----------------------
# OpenRouter client
# -----------------------

openrouter_key = settings.openrouter.api_key
openrouter_url = settings.openrouter.api_url
async_openrouter_client = AsyncOpenAI(base_url=openrouter_url, api_key=openrouter_key)

# -----------------------
# Opik Observability
# -----------------------

os.environ["OPIK_API_KEY"] = settings.opik.api_key
os.environ["OPIK_PROJECT_NAME"] = settings.opik.project_name

async_openrouter_client = track_openai(async_openrouter_client)

# -----------------------
# Helper to build extra body for OpenRouter
# -----------------------


@opik.track(name="build_openrouter_extra")
def build_openrouter_extra(config: ModelConfig) -> dict[str, Any]:
    """Build the extra body for OpenRouter API requests based on the ModelConfig.

    Args:
        config (ModelConfig): The model configuration.

    Returns:
        dict[str, Any]: The extra body for OpenRouter API requests.

    """
    body = {"provider": {"sort": config.provider_sort.value}}
    if config.candidate_models:
        body["models"] = list(config.candidate_models)  # type: ignore
    return body


# -----------------------
# Core OpenRouter functions
# -----------------------


@opik.track(name="generate_openrouter")
async def generate_openrouter(
    prompt: str,
    config: ModelConfig,
    selected_model: str | None = None,
) -> tuple[str, str | None, str | None]:
    """Generate a response from OpenRouter for a given prompt and model configuration.

    Args:
        prompt (str): The input prompt.
        config (ModelConfig): The model configuration.
        selected_model (str | None): Optional specific model to use.

    Returns:
        tuple[str, str | None, str | None]: The generated response, model used, and finish reason.

    """

    model_to_use = selected_model or config.primary_model

    resp = await async_openrouter_client.chat.completions.create(
        model=model_to_use,
        messages=build_messages(prompt),
        temperature=config.temperature,
        max_completion_tokens=config.max_completion_tokens,
        extra_body=build_openrouter_extra(config),
    )
    answer = resp.choices[0].message.content or ""

    # Reasons: tool_calls, stop, length, content_filter, error
    finish_reason = getattr(resp.choices[0], "native_finish_reason", None)
    model_used = getattr(resp.choices[0], "model", None) or getattr(resp, "model", None)

    logger.info(f"OpenRouter non-stream finish_reason: {finish_reason}")
    if finish_reason == "length":
        logger.warning("Response was truncated by token limit.")

    model_used = getattr(resp.choices[0], "model", None) or getattr(resp, "model", None)
    logger.info(f"OpenRouter non-stream finished. Model used: {model_used}")

    return answer, model_used, finish_reason


@opik.track(name="stream_openrouter")
def stream_openrouter(
    prompt: str,
    config: ModelConfig,
    selected_model: str | None = None,
) -> AsyncGenerator[str, None]:
    """Stream a response from OpenRouter for a given prompt and model configuration.

    Args:
        prompt (str): The input prompt.
        config (ModelConfig): The model configuration.
        selected_model (str | None): Optional specific model to use.

    Returns:
        AsyncGenerator[str, None]: An asynchronous generator yielding response chunks.

    """

    async def gen() -> AsyncGenerator[str, None]:
        """Generate response chunks from OpenRouter.

        Yields:
            AsyncGenerator[str, None]: Response chunks.

        """

        model_to_use = selected_model or config.primary_model

        stream = await async_openrouter_client.chat.completions.create(
            model=model_to_use,
            messages=build_messages(prompt),
            temperature=config.temperature,
            max_completion_tokens=config.max_completion_tokens,
            extra_body=build_openrouter_extra(config),
            stream=True,
        )
        try:
            first_chunk = await stream.__anext__()
            model_used = getattr(first_chunk, "model", None)
            if model_used:
                yield f"__model_used__:{model_used}"
            delta_text = getattr(first_chunk.choices[0].delta, "content", None)
            if delta_text:
                yield delta_text
        except StopAsyncIteration:
            return

        last_finish_reason = None
        async for chunk in stream:
            delta_text = getattr(chunk.choices[0].delta, "content", None)
            if delta_text:
                yield delta_text

            # Reasons: tool_calls, stop, length, content_filter, error
            finish_reason = getattr(chunk.choices[0], "finish_reason", None)

            if finish_reason:
                last_finish_reason = finish_reason

        logger.info(f"OpenRouter stream finished. Model used: {model_used}")
        logger.warning(f"Final finish_reason: {last_finish_reason}")

        # Yield a chunk to trigger truncation warning in UI
        if last_finish_reason == "length":
            yield "__truncated__"

    return gen()


# ---------------------------------------
# Test Log Probs and Confidence Visualization
# ---------------------------------------


# import math


# def visualize_token_confidence(token_probs: list[tuple[str, float]]):
#     """Print token probabilities as ASCII bars in the terminal."""
#     for tok, prob in token_probs:
#         bar_length = int(prob * 40)  # scale bar to 40 chars max
#         bar = "#" * bar_length
#         print(f"{tok:>12}: [{bar:<40}] {prob:.2%}")

# async def generate_openrouter(
#     prompt: str,
#     config: ModelConfig,
#     max_tokens: int | None = None) -> tuple[str, str | None, str | None]:
#     """Generate a response from OpenRouter
#     and log token-level statistics with confidence evolution."""

#     resp = await async_openrouter_client.chat.completions.create(
#         model=config.primary_model,
#         messages=build_messages(prompt),
#         temperature=config.temperature,
#         max_completion_tokens=max_tokens or config.max_completion_tokens,
#         extra_body={**build_openrouter_extra(config), "logprobs": True, "top_logprobs": 3},
#     )

#     choice = resp.choices[0]
#     content = choice.message.content or ""
#     finish_reason = getattr(choice, "native_finish_reason", None)
#     model_used = getattr(choice, "model", None) or getattr(resp, "model", None)

#     logger.info(f"OpenRouter non-stream finish_reason: {finish_reason}")
#     if finish_reason == "length":
#         logger.warning("Response was truncated by token limit.")

#     # Extract logprobs
#     token_logprobs_list = choice.logprobs
#     tokens_logprobs = []
#     token_probs = []

#     if token_logprobs_list and hasattr(token_logprobs_list, "content"):
#         for token_info in token_logprobs_list.content:
#             tok = token_info.token
#             logprob = token_info.logprob
#             prob = math.exp(logprob)

#             tokens_logprobs.append(logprob)
#             token_probs.append((tok, prob))


#     if tokens_logprobs:
#         avg_logprob = sum(tokens_logprobs) / len(tokens_logprobs)
#         avg_prob = math.exp(avg_logprob)

#         most_confident = sorted(token_probs, key=lambda x: x[1], reverse=True)[:5]
#         least_confident = sorted(token_probs, key=lambda x: x[1])[:5]

#         logger.info(f"Temperature: {config.temperature}")
#         logger.info(f"Max completion tokens: {config.max_completion_tokens}")
# logger.info(f"Average log probability: {avg_logprob:.4f} "
#             f"(≈ {avg_prob:.2%} avg token prob)")"

#         logger.info("Top 5 most confident tokens:")
#         for tok, prob in most_confident:
#             logger.info(f"  '{tok}' → {prob:.2%}")

#         logger.info("Top 5 least confident tokens:")
#         for tok, prob in least_confident:
#             logger.info(f"  '{tok}' → {prob:.2%}")

#         # Terminal visualization
#         print("\nToken confidence evolution:")
#         visualize_token_confidence(token_probs,)

#     else:
#         logger.warning("No logprob information found in response.")

#     return content, model_used, finish_reason
