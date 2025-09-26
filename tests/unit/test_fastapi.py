import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app


@pytest.mark.asyncio
async def test_lifespan_and_client():
    """Test the application lifespan and verify that the Qdrant client
    is properly initialized and available in app.state during requests.
    """
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")
        assert response.status_code == 200, "Health endpoint did not return 200 OK"


@pytest.mark.asyncio
async def test_search_unique_titles_route():
    """Test that the /search route can access the Qdrant client from app.state
    and return unique titles.
    """
    payload = {
        "query_text": "RAG",
        "feed_author": None,
        "feed_name": None,
        "title_keywords": None,
        "limit": 1,
    }

    # Use lifespan context to ensure app.state.qdrant_client is initialized
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post("/search/unique-titles", json=payload)
            assert response.status_code == 200, "Search endpoint did not return 200 OK"
            assert "results" in response.json(), "Search response missing 'results' key"


@pytest.mark.asyncio
async def test_search_ask():
    """Test that the /search route can access the Qdrant client from app.state
    and return unique titles.
    """
    payload = {"query_text": "RAG", "provider": "OpenRouter", "limit": 1}

    # Use lifespan context to ensure app.state.qdrant_client is initialized
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post("/search/ask", json=payload)
            assert response.status_code == 200, "Ask endpoint did not return 200 OK"
            assert "answer" in response.json(), "Ask response missing 'answer' key"


### ASGI Transport
# You can configure an httpx client to call
# directly into an async Python web application using the ASGI protocol.
