"""
Security middleware for CORS, rate limiting, and security headers.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional

from flask import Flask, Response, request


class CORSMiddleware:
    """CORS (Cross-Origin Resource Sharing) middleware."""
    
    def __init__(
        self,
        origins: List[str] = None,
        methods: List[str] = None,
        allow_headers: List[str] = None,
        allow_credentials: bool = True
    ):
        """
        Initialize CORS middleware.
        
        Parameters
        ----------
        origins : List[str]
            Allowed origins (default: ['*'])
        methods : List[str]
            Allowed methods
        allow_headers : List[str]
            Allowed headers
        allow_credentials : bool
            Allow credentials
        """
        self.origins = origins or ['*']
        self.methods = methods or ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
        self.allow_headers = allow_headers or ['Content-Type', 'Authorization']
        self.allow_credentials = allow_credentials
    
    def apply(self, app: Flask):
        """Apply CORS middleware to Flask app."""
        @app.after_request
        def add_cors_headers(response: Response) -> Response:
            origin = request.headers.get('Origin')
            
            if '*' in self.origins:
                response.headers['Access-Control-Allow-Origin'] = '*'
            elif origin and origin in self.origins:
                response.headers['Access-Control-Allow-Origin'] = origin
            
            response.headers['Access-Control-Allow-Methods'] = ', '.join(self.methods)
            response.headers['Access-Control-Allow-Headers'] = ', '.join(self.allow_headers)
            
            if self.allow_credentials:
                response.headers['Access-Control-Allow-Credentials'] = 'true'
            
            return response
        
        @app.route('/<path:path>', methods=['OPTIONS'])
        @app.route('/', methods=['OPTIONS'], defaults={'path': ''})
        def handle_options(path: str = ''):
            response = Response()
            response.status_code = 200
            return response


class RateLimitMiddleware:
    """Rate limiting middleware."""
    
    def __init__(self, requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiting middleware.
        
        Parameters
        ----------
        requests : int
            Maximum requests per window
        window_seconds : int
            Time window in seconds
        """
        self.max_requests = requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
    
    def _get_client_id(self) -> str:
        """Get client identifier from request."""
        return request.remote_addr or 'unknown'
    
    def _clean_old_requests(self, client_id: str):
        """Remove requests outside the time window."""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > cutoff_time
        ]
    
    def check_rate_limit(self) -> Optional[Response]:
        """
        Check if request exceeds rate limit.
        
        Returns
        -------
        Optional[Response]
            429 response if rate limit exceeded, None otherwise
        """
        client_id = self._get_client_id()
        current_time = time.time()
        
        self._clean_old_requests(client_id)
        
        if len(self.requests[client_id]) >= self.max_requests:
            retry_after = int(
                self.requests[client_id][0] + self.window_seconds - current_time
            )
            
            response = Response('Rate limit exceeded', 429)
            response.headers['Retry-After'] = str(retry_after)
            response.headers['X-RateLimit-Limit'] = str(self.max_requests)
            response.headers['X-RateLimit-Remaining'] = '0'
            response.headers['X-RateLimit-Reset'] = str(int(current_time + retry_after))
            return response
        
        self.requests[client_id].append(current_time)
        return None
    
    def apply(self, app: Flask):
        """Apply rate limiting middleware to Flask app."""
        @app.before_request
        def check_rate_limit():
            return self.check_rate_limit()
        
        @app.after_request
        def add_rate_limit_headers(response: Response) -> Response:
            client_id = self._get_client_id()
            self._clean_old_requests(client_id)
            
            remaining = max(0, self.max_requests - len(self.requests[client_id]))
            
            response.headers['X-RateLimit-Limit'] = str(self.max_requests)
            response.headers['X-RateLimit-Remaining'] = str(remaining)
            
            if self.requests[client_id]:
                reset_time = int(self.requests[client_id][0] + self.window_seconds)
                response.headers['X-RateLimit-Reset'] = str(reset_time)
            
            return response


class SecurityHeadersMiddleware:
    """Security headers middleware."""
    
    def __init__(self):
        """Initialize security headers middleware."""
        self.headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'SAMEORIGIN',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: https:",
        }
    
    def apply(self, app: Flask):
        """Apply security headers middleware to Flask app."""
        @app.after_request
        def add_security_headers(response: Response) -> Response:
            for header, value in self.headers.items():
                if header not in response.headers:
                    response.headers[header] = value
            return response


def apply_security_middleware(
    app: Flask,
    cors_enabled: bool = True,
    cors_origins: List[str] = None,
    cors_methods: List[str] = None,
    cors_allow_headers: List[str] = None,
    cors_allow_credentials: bool = True,
    rate_limit_enabled: bool = True,
    rate_limit_requests: int = 100,
    rate_limit_window_seconds: int = 60,
    security_headers_enabled: bool = True
):
    """
    Apply all security middleware to Flask app.
    
    Parameters
    ----------
    app : Flask
        Flask application
    cors_enabled : bool
        Enable CORS middleware
    cors_origins : List[str]
        Allowed CORS origins
    cors_methods : List[str]
        Allowed CORS methods
    cors_allow_headers : List[str]
        Allowed CORS headers
    cors_allow_credentials : bool
        Allow CORS credentials
    rate_limit_enabled : bool
        Enable rate limiting
    rate_limit_requests : int
        Max requests per window
    rate_limit_window_seconds : int
        Rate limit window in seconds
    security_headers_enabled : bool
        Enable security headers
    """
    if cors_enabled:
        cors = CORSMiddleware(
            origins=cors_origins,
            methods=cors_methods,
            allow_headers=cors_allow_headers,
            allow_credentials=cors_allow_credentials
        )
        cors.apply(app)
    
    if rate_limit_enabled:
        rate_limiter = RateLimitMiddleware(
            requests=rate_limit_requests,
            window_seconds=rate_limit_window_seconds
        )
        rate_limiter.apply(app)
    
    if security_headers_enabled:
        security_headers = SecurityHeadersMiddleware()
        security_headers.apply(app)
