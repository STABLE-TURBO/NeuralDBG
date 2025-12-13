"""
Security module for Neural DSL.

Provides unified authentication, authorization, and security configuration
for all server components.
"""

from neural.security.auth import (
    Authentication,
    HTTPBasicAuthMiddleware,
    JWTAuthMiddleware,
    create_basic_auth,
    create_jwt_auth,
    require_auth,
)
from neural.security.config import SecurityConfig, load_security_config
from neural.security.middleware import (
    CORSMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    apply_security_middleware,
)

__all__ = [
    'Authentication',
    'HTTPBasicAuthMiddleware',
    'JWTAuthMiddleware',
    'create_basic_auth',
    'create_jwt_auth',
    'require_auth',
    'SecurityConfig',
    'load_security_config',
    'CORSMiddleware',
    'RateLimitMiddleware',
    'SecurityHeadersMiddleware',
    'apply_security_middleware',
]
