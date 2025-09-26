import time

from fastapi import APIRouter, Request
from qdrant_client.http.exceptions import UnexpectedResponse

router = APIRouter()

start_time = time.time()


@router.get("/")
async def root():
    """Root endpoint.

    Returns a simple JSON response indicating that the API is running.

    Returns:
        dict: {"message": "Hello! API is running."}

    """
    return {"message": "Hello! API is running."}


@router.get("/health")
async def health_check():
    """Liveness check endpoint.

    Returns basic service info, uptime, and environment variables.
    """
    uptime = int(time.time() - start_time)
    return {
        "status": "ok",
        "uptime_seconds": uptime,
    }


@router.get("/ready")
async def readiness_check(request: Request):
    """Readiness check endpoint.

    Verifies whether the service is ready to handle requests by
    checking connectivity to Qdrant.
    """
    try:
        vectorstore = request.app.state.vectorstore
        # a lightweight check: list_collections is cheap
        await vectorstore.client.get_collections()
        return {"status": "ready"}
    except UnexpectedResponse:
        return {"status": "not ready", "reason": "Qdrant unexpected response"}
    except Exception as e:
        return {"status": "not ready", "reason": str(e)}
