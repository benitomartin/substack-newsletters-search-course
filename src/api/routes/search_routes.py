import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.api.models.api_models import (
    AskRequest,
    AskResponse,
    AskStreamingResponse,
    SearchResult,
    UniqueTitleRequest,
    UniqueTitleResponse,
)
from src.api.services.generation_service import generate_answer, get_streaming_function
from src.api.services.search_service import query_unique_titles, query_with_filters

router = APIRouter()


@router.post("/unique-titles", response_model=UniqueTitleResponse)
async def search_unique(request: Request, params: UniqueTitleRequest):
    """Returns unique article titles based on a query and optional filters.

    Deduplicates results by article title.

    Args:
        request: FastAPI request object.
        params: UniqueTitleRequest with search parameters.

    Returns:
        UniqueTitleResponse: List of unique titles.

    """
    results = await query_unique_titles(
        request=request,
        query_text=params.query_text,
        feed_author=params.feed_author,
        feed_name=params.feed_name,
        title_keywords=params.title_keywords,
        limit=params.limit,
    )
    return {"results": results}


@router.post("/ask", response_model=AskResponse)
async def ask_with_generation(request: Request, ask: AskRequest):
    """Non-streaming question-answering endpoint using vector search and LLM.

    Workflow:
        1. Retrieve relevant documents (possibly duplicate titles for richer context).
        2. Generate an answer with the selected LLM provider.

    Args:
        request: FastAPI request object.
        ask: AskRequest with query, provider, and limit.

    Returns:
        AskResponse: Generated answer and source documents.

    """
    # Step 1: Retrieve relevant documents with filters
    results: list[SearchResult] = await query_with_filters(
        request,
        query_text=ask.query_text,
        feed_author=ask.feed_author,
        feed_name=ask.feed_name,
        title_keywords=ask.title_keywords,
        limit=ask.limit,
    )

    # Step 2: Generate an answer
    answer_data = await generate_answer(
        query=ask.query_text, contexts=results, provider=ask.provider, selected_model=ask.model
    )

    return AskResponse(
        query=ask.query_text,
        provider=ask.provider,
        answer=answer_data["answer"],
        sources=results,
        model=answer_data.get("model", None),
        finish_reason=answer_data.get("finish_reason", None),
    )


@router.post("/ask/stream", response_model=AskStreamingResponse)
async def ask_with_generation_stream(request: Request, ask: AskRequest):
    """Streaming question-answering endpoint using vector search and LLM.

    Workflow:
        1. Retrieve relevant documents (possibly duplicate titles for richer context).
        2. Stream generated answer with the selected LLM provider.

    Args:
        request: FastAPI request object.
        ask: AskRequest with query, provider, and limit.

    Returns:
        StreamingResponse: Yields text chunks as plain text.

    """
    # Step 1: Retrieve relevant documents with filters
    results: list[SearchResult] = await query_with_filters(
        request,
        query_text=ask.query_text,
        feed_author=ask.feed_author,
        feed_name=ask.feed_name,
        title_keywords=ask.title_keywords,
        limit=ask.limit,
    )

    # Step 2: Get the streaming generator
    stream_func = get_streaming_function(
        provider=ask.provider, query=ask.query_text, contexts=results, selected_model=ask.model
    )

    # Step 3: Wrap streaming generator
    async def stream_generator():
        async for delta in stream_func():
            yield delta
            await asyncio.sleep(0)  # allow event loop to handle other tasks

    return StreamingResponse(stream_generator(), media_type="text/plain")
