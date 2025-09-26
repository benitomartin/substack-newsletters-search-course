import os
from contextlib import asynccontextmanager

import dotenv
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client.http.exceptions import UnexpectedResponse

from src.api.exceptions.exception_handlers import (
    general_exception_handler,
    qdrant_exception_handler,
    validation_exception_handler,
)
from src.api.middleware.logging_middleware import LoggingMiddleware
from src.api.routes.health_routes import router as health_router
from src.api.routes.search_routes import router as search_router
from src.infrastructure.qdrant.qdrant_vectorstore import AsyncQdrantVectorStore
from src.utils.logger_util import setup_logging

# Load environment variables from .env file
dotenv.load_dotenv()

# -----------------------
# Logger setup
# -----------------------
logger = setup_logging()


# -----------------------
# Lifespan
# -----------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    Initializes the Qdrant vector store on startup and ensures proper cleanup on shutdown.

    Args:
        app (FastAPI): The FastAPI application instance.
    Yields:
        None

    Exceptions:
        Raises exceptions if initialization or cleanup fails.
    """
    ## Ensure the cache directory exists and is writable (HF downloads the models here)
    cache_dir = "/tmp/fastembed_cache"
    os.makedirs(cache_dir, exist_ok=True)  # Ensure directory exists
    # Force /tmp/huggingface in Google Cloud so that it's writable.
    # This is the default cache dir of Huggingface.
    # Otherwise it tries ~/.cache/huggingface (read-only directory) in Google Cloud.
    # That directory is not writable.
    logger.info(f"HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
    logger.info(f"Cache dir: {cache_dir}, Writable: {os.access(cache_dir, os.W_OK)}")
    cache_contents = os.listdir(cache_dir) if os.path.exists(cache_dir) else "Empty"
    logger.info(f"Cache contents before: {cache_contents}")
    try:
        # creates Qdrant client internally
        app.state.vectorstore = AsyncQdrantVectorStore(cache_dir=cache_dir)
    except Exception as e:
        logger.exception("Failed to initialize QdrantVectorStore")
        raise e
    yield
    try:
        await app.state.vectorstore.client.close()
    except Exception:
        logger.exception("Failed to close Qdrant client")


# -----------------------
# FastAPI application
# -----------------------

app = FastAPI(
    title="Substack RAG API",
    version="1.0",
    description="API for Substack Retrieval-Augmented Generation (RAG) system",
    lifespan=lifespan,
    # root_path=root_path,
)


# -----------------------
# Middleware
# -----------------------


# Log the allowed origins
allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
logger.info(f"CORS allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # ["*"],  # allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # only the methods the app uses
    allow_headers=["Authorization", "Content-Type"],  # only headers needed
)

app.add_middleware(LoggingMiddleware)


# -----------------------
# Exception Handlers
# -----------------------
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(UnexpectedResponse, qdrant_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


# -----------------------
# Routers
# -----------------------
app.include_router(search_router, prefix="/search", tags=["search"])
app.include_router(health_router, tags=["health"])

# For Cloud Run, run the app directly
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))  # Cloud Run provides PORT env var

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=True,  # Enable auto-reload for development
    )

    # config = uvicorn.Config(
    #     app,
    #     port=port,
    #     log_level="info",
    #     # loop="uvloop",
    #     # workers=1,
    #     reload=True
    #     )
    # server = uvicorn.Server(config)

    # server.run()
