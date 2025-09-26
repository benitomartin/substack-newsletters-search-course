import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger_util import setup_logging

logger = setup_logging()


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging incoming HTTP requests and their responses.

    Logs the request method, URL, client IP, and headers.
    Excludes sensitive headers like Authorization and Cookie.
    as well as the response status code and request duration in milliseconds.
    Exceptions raised during request processing are logged with the full traceback.

    Usage:
        Add this middleware to your FastAPI app:
            app.add_middleware(LoggingMiddleware)

    Attributes:
        logger: Configured logger from `setup_logging`.

    """

    async def dispatch(self, request: Request, call_next):
        """Process the incoming request, log its details, and measure execution time.

        Args:
            request (Request): The incoming FastAPI request.
            call_next: Callable to invoke the next middleware or route handler.

        Returns:
            Response: The HTTP response returned by the next middleware or route handler.

        Raises:
            Exception: Propagates any exceptions raised by downstream handlers after logging them.

        """
        start_time = time.time()
        client_host = request.client.host if request.client else "unknown"

        # logger.debug(f"Request headers: {request.headers}")
        # logger.debug(f"Request cookies: {request.cookies}")

        # Exclude sensitive headers from logging
        safe_headers = {
            k: v for k, v in request.headers.items() if k.lower() not in {"authorization", "cookie"}
        }

        logger.info(
            f"Incoming request: {request.method} {request.url} from {client_host} "
            f"headers={safe_headers}"
        )

        try:
            response = await call_next(request)
        except Exception:
            duration = (time.time() - start_time) * 1000
            logger.exception(
                f"Request failed: {request.method} {request.url} from {client_host} "
                f"duration={duration:.2f}ms"
            )
            raise

        duration = (time.time() - start_time) * 1000
        logger.info(
            f"Completed request: {request.method} {request.url} from {client_host} "
            f"status_code={response.status_code} duration={duration:.2f}ms"
        )
        return response
