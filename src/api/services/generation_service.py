from collections.abc import AsyncGenerator, Callable

import opik

from src.api.models.api_models import SearchResult
from src.api.models.provider_models import MODEL_REGISTRY
from src.api.services.providers.huggingface_service import generate_huggingface, stream_huggingface
from src.api.services.providers.openai_service import generate_openai, stream_openai
from src.api.services.providers.openrouter_service import generate_openrouter, stream_openrouter
from src.api.services.providers.utils.evaluation_metrics import evaluate_metrics
from src.api.services.providers.utils.prompts import build_research_prompt
from src.utils.logger_util import setup_logging

logger = setup_logging()


# -----------------------
# Non-streaming answer generator
# -----------------------
@opik.track(name="generate_answer")
async def generate_answer(
    query: str,
    contexts: list[SearchResult],
    provider: str = "openrouter",
    selected_model: str | None = None,
) -> dict:
    """Generate a non-streaming answer using the specified LLM provider.

    Args:
        query (str): The user's research query.
        contexts (list[SearchResult]): List of context documents with metadata.
        provider (str): The LLM provider to use ("openai", "openrouter", "huggingface").

    Returns:
        dict: {"answer": str, "sources": list[str], "model": Optional[str]}

    """
    prompt = build_research_prompt(contexts, query=query)
    model_used: str | None = None
    finish_reason: str | None = None

    provider_lower = provider.lower()

    config = MODEL_REGISTRY.get_config(provider_lower)

    if provider_lower == "openai":
        answer, model_used = await generate_openai(prompt, config=config)
    elif provider_lower == "openrouter":
        try:
            answer, model_used, finish_reason = await generate_openrouter(
                prompt, config=config, selected_model=selected_model
            )
            metrics_results = await evaluate_metrics(answer, prompt)
            logger.info(f"G-Eval Faithfulness → {metrics_results}")
        except Exception as e:
            logger.error(f"Error occurred while generating answer from {provider_lower}: {e}")
            raise

    elif provider_lower == "huggingface":
        answer, model_used = await generate_huggingface(prompt, config=config)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return {
        "answer": answer,
        "sources": [r.url for r in contexts],
        "model": model_used,
        "finish_reason": finish_reason,
    }


# -----------------------
# Streaming answer generator
# -----------------------
@opik.track(name="get_streaming_function")
def get_streaming_function(
    provider: str,
    query: str,
    contexts: list[SearchResult],
    selected_model: str | None = None,
) -> Callable[[], AsyncGenerator[str, None]]:
    """Get a streaming function for the specified LLM provider.

    Args:
        provider (str): The LLM provider to use ("openai", "openrouter", "huggingface").
        query (str): The user's research query.
        contexts (list[SearchResult]): List of context documents with metadata.

    Returns:
        Callable[[], AsyncGenerator[str, None]]: A function that returns an async generator yielding
        response chunks.

    """
    prompt = build_research_prompt(contexts, query=query)
    provider_lower = provider.lower()
    config = MODEL_REGISTRY.get_config(provider_lower)
    logger.info(f"Using model config: {config}")

    async def stream_gen() -> AsyncGenerator[str, None]:
        """Asynchronous generator that streams response chunks from the specified provider.

        Yields:
            str: The next chunk of the response.

        """
        buffer = []  # collect all chunks here

        if provider_lower == "openai":
            async for chunk in stream_openai(prompt, config=config):
                buffer.append(chunk)
                yield chunk

        elif provider_lower == "openrouter":
            try:
                async for chunk in stream_openrouter(
                    prompt, config=config, selected_model=selected_model
                ):
                    buffer.append(chunk)
                    yield chunk

                full_output = "".join(buffer)
                metrics_results = await evaluate_metrics(full_output, prompt)
                logger.info(f"Metrics results: {metrics_results}")

            except Exception as e:
                logger.error(f"Error occurred while streaming from {provider}: {e}")
                yield "__error__"

        elif provider_lower == "huggingface":
            async for chunk in stream_huggingface(prompt, config=config):
                buffer.append(chunk)
                yield chunk

        else:
            raise ValueError(f"Unknown provider: {provider}")

    return stream_gen
