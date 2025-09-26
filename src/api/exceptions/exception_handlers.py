from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from qdrant_client.http.exceptions import UnexpectedResponse

from src.utils.logger_util import setup_logging

logger = setup_logging()


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle FastAPI request validation errors.

    Args:
        request (Request): The incoming request that caused the validation error.
        exc (Exception): The exception instance.

    Returns:
        JSONResponse: A JSON response with status code 422 and error details.

    """
    if isinstance(exc, RequestValidationError):
        logger.warning(f"Validation error on {request.url}: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={
                "type": "validation_error",
                "message": "Invalid request",
                "details": exc.errors(),
            },
        )

    logger.exception(f"Unexpected exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "type": "internal_error",
            "message": "Internal server error",
            "details": str(exc),
        },
    )


async def qdrant_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected responses from Qdrant.

    Args:
        request (Request): The incoming request that caused the error.
        exc (Exception): The exception instance.

    Returns:
        JSONResponse: A JSON response with status code 500 and error details.

    """
    if isinstance(exc, UnexpectedResponse):
        logger.error(f"Qdrant error on {request.url}: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "type": "qdrant_error",
                "message": "Vector store error",
                "details": str(exc),
            },
        )

    # Fallback to general internal error if exception is not UnexpectedResponse
    logger.exception(f"Unexpected exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "type": "internal_error",
            "message": "Internal server error",
            "details": str(exc),
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all uncaught exceptions in FastAPI.

    Args:
        request (Request): The incoming request that caused the error.
        exc (Exception): The exception instance.

    Returns:
        JSONResponse: A JSON response with status code 500 and error details.

    """
    logger.exception(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "type": "internal_error",
            "message": "Internal server error",
            "details": str(exc),
        },
    )
