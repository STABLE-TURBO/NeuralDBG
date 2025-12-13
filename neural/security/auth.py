"""
Unified authentication middleware for Flask and Dash applications.

Provides HTTP Basic Auth and JWT authentication with Flask integration.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

from flask import Flask, Response, request


class Authentication:
    """Base authentication class."""
    
    def check_auth(self, auth_data: Dict[str, str]) -> bool:
        """
        Check if authentication credentials are valid.
        
        Parameters
        ----------
        auth_data : Dict[str, str]
            Authentication data
            
        Returns
        -------
        bool
            True if authentication is valid
        """
        raise NotImplementedError
    
    def get_auth_data(self, request_obj) -> Optional[Dict[str, str]]:
        """
        Extract authentication data from request.
        
        Parameters
        ----------
        request_obj : Request
            Flask request object
            
        Returns
        -------
        Optional[Dict[str, str]]
            Authentication data or None
        """
        raise NotImplementedError
    
    def authenticate(self) -> Response:
        """Return authentication challenge response."""
        raise NotImplementedError


class HTTPBasicAuthMiddleware(Authentication):
    """HTTP Basic Authentication middleware."""
    
    def __init__(self, username: str, password: str, realm: str = 'Neural DSL'):
        """
        Initialize Basic Auth middleware.
        
        Parameters
        ----------
        username : str
            Valid username
        password : str
            Valid password
        realm : str
            Authentication realm
        """
        self.username = username
        self.password = password
        self.realm = realm
    
    def check_auth(self, auth_data: Dict[str, str]) -> bool:
        """Check if username and password are valid."""
        username = auth_data.get('username', '')
        password = auth_data.get('password', '')
        
        return (
            hmac.compare_digest(username, self.username) and
            hmac.compare_digest(password, self.password)
        )
    
    def get_auth_data(self, request_obj) -> Optional[Dict[str, str]]:
        """Extract Basic Auth credentials from request."""
        auth = request_obj.authorization
        if not auth:
            return None
        
        return {
            'username': auth.username or '',
            'password': auth.password or ''
        }
    
    def authenticate(self) -> Response:
        """Return 401 with WWW-Authenticate header."""
        return Response(
            'Authentication required',
            401,
            {'WWW-Authenticate': f'Basic realm="{self.realm}"'}
        )


class JWTAuthMiddleware(Authentication):
    """JWT (JSON Web Token) Authentication middleware."""
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = 'HS256',
        expiration_hours: int = 24
    ):
        """
        Initialize JWT Auth middleware.
        
        Parameters
        ----------
        secret_key : str
            Secret key for JWT signing
        algorithm : str
            JWT algorithm (default: HS256)
        expiration_hours : int
            Token expiration time in hours
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.expiration_hours = expiration_hours
    
    def create_token(self, payload: Dict[str, Any]) -> str:
        """
        Create a JWT token.
        
        Parameters
        ----------
        payload : Dict[str, Any]
            Token payload data
            
        Returns
        -------
        str
            JWT token
        """
        header = {
            'alg': self.algorithm,
            'typ': 'JWT'
        }
        
        payload = {
            **payload,
            'exp': (datetime.utcnow() + timedelta(hours=self.expiration_hours)).timestamp(),
            'iat': datetime.utcnow().timestamp()
        }
        
        header_b64 = base64.urlsafe_b64encode(
            json.dumps(header).encode()
        ).decode().rstrip('=')
        
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).decode().rstrip('=')
        
        message = f'{header_b64}.{payload_b64}'
        
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        
        signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip('=')
        
        return f'{message}.{signature_b64}'
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token.
        
        Parameters
        ----------
        token : str
            JWT token
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Decoded payload or None if invalid
        """
        try:
            parts = token.split('.')
            if len(parts) != 3:
                return None
            
            header_b64, payload_b64, signature_b64 = parts
            
            message = f'{header_b64}.{payload_b64}'
            
            expected_signature = hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).digest()
            
            expected_signature_b64 = base64.urlsafe_b64encode(
                expected_signature
            ).decode().rstrip('=')
            
            if not hmac.compare_digest(signature_b64, expected_signature_b64):
                return None
            
            padding = '=' * (4 - len(payload_b64) % 4)
            payload_json = base64.urlsafe_b64decode(payload_b64 + padding).decode()
            payload = json.loads(payload_json)
            
            if payload.get('exp', 0) < time.time():
                return None
            
            return payload
        
        except Exception:
            return None
    
    def check_auth(self, auth_data: Dict[str, str]) -> bool:
        """Check if JWT token is valid."""
        token = auth_data.get('token', '')
        payload = self.verify_token(token)
        return payload is not None
    
    def get_auth_data(self, request_obj) -> Optional[Dict[str, str]]:
        """Extract JWT token from request."""
        auth_header = request_obj.headers.get('Authorization', '')
        
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            return {'token': token}
        
        return None
    
    def authenticate(self) -> Response:
        """Return 401 with JSON error."""
        return Response(
            json.dumps({'error': 'Invalid or missing authentication token'}),
            401,
            {'Content-Type': 'application/json'}
        )


def create_basic_auth(username: Optional[str] = None, password: Optional[str] = None) -> Optional[HTTPBasicAuthMiddleware]:
    """
    Create HTTP Basic Auth middleware from credentials.
    
    Parameters
    ----------
    username : Optional[str]
        Username (required if auth is enabled)
    password : Optional[str]
        Password (required if auth is enabled)
        
    Returns
    -------
    Optional[HTTPBasicAuthMiddleware]
        Auth middleware or None if credentials not provided
    """
    if not username or not password:
        return None
    
    return HTTPBasicAuthMiddleware(username, password)


def create_jwt_auth(
    secret_key: Optional[str] = None,
    algorithm: str = 'HS256',
    expiration_hours: int = 24
) -> Optional[JWTAuthMiddleware]:
    """
    Create JWT Auth middleware.
    
    Parameters
    ----------
    secret_key : Optional[str]
        JWT secret key (required if auth is enabled)
    algorithm : str
        JWT algorithm
    expiration_hours : int
        Token expiration hours
        
    Returns
    -------
    Optional[JWTAuthMiddleware]
        Auth middleware or None if secret not provided
    """
    if not secret_key:
        return None
    
    return JWTAuthMiddleware(secret_key, algorithm, expiration_hours)


def require_auth(auth: Authentication) -> Callable:
    """
    Decorator to require authentication for Flask routes.
    
    Parameters
    ----------
    auth : Authentication
        Authentication middleware
        
    Returns
    -------
    Callable
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            auth_data = auth.get_auth_data(request)
            
            if not auth_data or not auth.check_auth(auth_data):
                return auth.authenticate()
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator
