"""
Security configuration management.

Loads and manages security settings from environment variables and config files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class SecurityConfig:
    """Security configuration for server components."""
    
    auth_enabled: bool = True
    auth_type: str = 'basic'
    
    basic_auth_username: Optional[str] = None
    basic_auth_password: Optional[str] = None
    
    jwt_secret_key: Optional[str] = None
    jwt_algorithm: str = 'HS256'
    jwt_expiration_hours: int = 24
    
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ['*'])
    cors_methods: List[str] = field(default_factory=lambda: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
    cors_allow_headers: List[str] = field(default_factory=lambda: ['Content-Type', 'Authorization'])
    cors_allow_credentials: bool = True
    
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    
    security_headers_enabled: bool = True
    
    ssl_enabled: bool = False
    ssl_cert_file: Optional[str] = None
    ssl_key_file: Optional[str] = None
    
    @classmethod
    def from_env(cls, prefix: str = 'NEURAL_') -> SecurityConfig:
        """
        Load security configuration from environment variables.
        
        Parameters
        ----------
        prefix : str
            Environment variable prefix (default: 'NEURAL_')
            
        Returns
        -------
        SecurityConfig
            Security configuration object
        """
        def get_env(key: str, default=None):
            return os.environ.get(f'{prefix}{key}', default)
        
        def get_bool(key: str, default: bool) -> bool:
            val = get_env(key)
            if val is None:
                return default
            return val.lower() in ('true', '1', 'yes', 'on')
        
        def get_int(key: str, default: int) -> int:
            val = get_env(key)
            if val is None:
                return default
            try:
                return int(val)
            except ValueError:
                return default
        
        def get_list(key: str, default: List[str]) -> List[str]:
            val = get_env(key)
            if val is None:
                return default
            return [item.strip() for item in val.split(',') if item.strip()]
        
        return cls(
            auth_enabled=get_bool('AUTH_ENABLED', True),
            auth_type=get_env('AUTH_TYPE', 'basic'),
            
            basic_auth_username=get_env('AUTH_USERNAME'),
            basic_auth_password=get_env('AUTH_PASSWORD'),
            
            jwt_secret_key=get_env('JWT_SECRET_KEY'),
            jwt_algorithm=get_env('JWT_ALGORITHM', 'HS256'),
            jwt_expiration_hours=get_int('JWT_EXPIRATION_HOURS', 24),
            
            cors_enabled=get_bool('CORS_ENABLED', True),
            cors_origins=get_list('CORS_ORIGINS', ['*']),
            cors_methods=get_list('CORS_METHODS', ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']),
            cors_allow_headers=get_list('CORS_ALLOW_HEADERS', ['Content-Type', 'Authorization']),
            cors_allow_credentials=get_bool('CORS_ALLOW_CREDENTIALS', True),
            
            rate_limit_enabled=get_bool('RATE_LIMIT_ENABLED', True),
            rate_limit_requests=get_int('RATE_LIMIT_REQUESTS', 100),
            rate_limit_window_seconds=get_int('RATE_LIMIT_WINDOW_SECONDS', 60),
            
            security_headers_enabled=get_bool('SECURITY_HEADERS_ENABLED', True),
            
            ssl_enabled=get_bool('SSL_ENABLED', False),
            ssl_cert_file=get_env('SSL_CERT_FILE'),
            ssl_key_file=get_env('SSL_KEY_FILE'),
        )
    
    @classmethod
    def from_yaml(cls, config_path: str) -> SecurityConfig:
        """
        Load security configuration from YAML file.
        
        Parameters
        ----------
        config_path : str
            Path to YAML configuration file
            
        Returns
        -------
        SecurityConfig
            Security configuration object
        """
        if not os.path.exists(config_path):
            return cls()
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        
        security_config = config_dict.get('security', {})
        
        return cls(
            auth_enabled=security_config.get('auth_enabled', True),
            auth_type=security_config.get('auth_type', 'basic'),
            
            basic_auth_username=security_config.get('basic_auth_username'),
            basic_auth_password=security_config.get('basic_auth_password'),
            
            jwt_secret_key=security_config.get('jwt_secret_key'),
            jwt_algorithm=security_config.get('jwt_algorithm', 'HS256'),
            jwt_expiration_hours=security_config.get('jwt_expiration_hours', 24),
            
            cors_enabled=security_config.get('cors_enabled', True),
            cors_origins=security_config.get('cors_origins', ['*']),
            cors_methods=security_config.get('cors_methods', ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']),
            cors_allow_headers=security_config.get('cors_allow_headers', ['Content-Type', 'Authorization']),
            cors_allow_credentials=security_config.get('cors_allow_credentials', True),
            
            rate_limit_enabled=security_config.get('rate_limit_enabled', True),
            rate_limit_requests=security_config.get('rate_limit_requests', 100),
            rate_limit_window_seconds=security_config.get('rate_limit_window_seconds', 60),
            
            security_headers_enabled=security_config.get('security_headers_enabled', True),
            
            ssl_enabled=security_config.get('ssl_enabled', False),
            ssl_cert_file=security_config.get('ssl_cert_file'),
            ssl_key_file=security_config.get('ssl_key_file'),
        )
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'auth_enabled': self.auth_enabled,
            'auth_type': self.auth_type,
            'jwt_algorithm': self.jwt_algorithm,
            'jwt_expiration_hours': self.jwt_expiration_hours,
            'cors_enabled': self.cors_enabled,
            'cors_origins': self.cors_origins,
            'cors_methods': self.cors_methods,
            'cors_allow_headers': self.cors_allow_headers,
            'cors_allow_credentials': self.cors_allow_credentials,
            'rate_limit_enabled': self.rate_limit_enabled,
            'rate_limit_requests': self.rate_limit_requests,
            'rate_limit_window_seconds': self.rate_limit_window_seconds,
            'security_headers_enabled': self.security_headers_enabled,
            'ssl_enabled': self.ssl_enabled,
        }


def load_security_config(
    config_path: Optional[str] = None,
    use_env: bool = True,
    env_prefix: str = 'NEURAL_'
) -> SecurityConfig:
    """
    Load security configuration with fallback priority:
    1. Environment variables (if use_env=True)
    2. YAML config file (if config_path provided)
    3. Default values
    
    Parameters
    ----------
    config_path : Optional[str]
        Path to YAML configuration file
    use_env : bool
        Whether to load from environment variables (default: True)
    env_prefix : str
        Environment variable prefix (default: 'NEURAL_')
        
    Returns
    -------
    SecurityConfig
        Loaded security configuration
    """
    if use_env:
        config = SecurityConfig.from_env(env_prefix)
        
        if config_path and os.path.exists(config_path):
            file_config = SecurityConfig.from_yaml(config_path)
            
            if config.basic_auth_username is None:
                config.basic_auth_username = file_config.basic_auth_username
            if config.basic_auth_password is None:
                config.basic_auth_password = file_config.basic_auth_password
            if config.jwt_secret_key is None:
                config.jwt_secret_key = file_config.jwt_secret_key
            if config.ssl_cert_file is None:
                config.ssl_cert_file = file_config.ssl_cert_file
            if config.ssl_key_file is None:
                config.ssl_key_file = file_config.ssl_key_file
        
        return config
    
    elif config_path:
        return SecurityConfig.from_yaml(config_path)
    
    return SecurityConfig()
