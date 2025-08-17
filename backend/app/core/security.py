"""
Security middleware for QuantumOptim by AYNX AI
- Adds common security headers
- Provides a safe placeholder for future auth/rate limiting
"""
from __future__ import annotations

import logging
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Minimal security middleware.

    Currently adds standard security headers to all responses. This is a
    non-blocking middleware so it will not interfere with requests while the
    rest of the application is being built out.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # In the future, plug in API key/JWT validation, rate limits, etc.
        response = await call_next(request)

        # Security headers
        response.headers.setdefault("Strict-Transport-Security", "max-age=63072000; includeSubDomains; preload")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        # Basic CSP that allows app to function; tighten as needed
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self' data: blob: *; script-src 'self' 'unsafe-inline' 'unsafe-eval' *; style-src 'self' 'unsafe-inline' *; img-src 'self' data: *; connect-src *; frame-ancestors 'none'",
        )

        return response
