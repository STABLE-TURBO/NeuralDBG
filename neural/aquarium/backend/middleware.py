"""Middleware for the backend bridge."""

import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        logger.info(f"Request: {request.method} {request.url.path}")

        response = await call_next(request)

        process_time = time.time() - start_time
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"status={response.status_code} duration={process_time:.3f}s"
        )

        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware to validate API key if configured."""

    def __init__(self, app, api_key: str = None):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if self.api_key:
            if request.url.path in ["/", "/health", "/docs", "/openapi.json"]:
                return await call_next(request)

            api_key = request.headers.get("X-API-Key")
            if api_key != self.api_key:
                return Response(
                    content="Invalid or missing API key",
                    status_code=401,
                )

        return await call_next(request)
