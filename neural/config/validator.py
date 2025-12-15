"""Configuration validation system for Neural DSL."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    
    level: str  # 'error', 'warning', 'info'
    variable: str
    message: str
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)
    invalid_values: Dict[str, str] = field(default_factory=dict)
    
    def add_error(self, variable: str, message: str, suggestion: Optional[str] = None):
        """Add an error issue."""
        self.issues.append(ValidationIssue('error', variable, message, suggestion))
        self.valid = False
    
    def add_warning(self, variable: str, message: str, suggestion: Optional[str] = None):
        """Add a warning issue."""
        self.issues.append(ValidationIssue('warning', variable, message, suggestion))
    
    def add_info(self, variable: str, message: str, suggestion: Optional[str] = None):
        """Add an info issue."""
        self.issues.append(ValidationIssue('info', variable, message, suggestion))
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(issue.level == 'error' for issue in self.issues)
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(issue.level == 'warning' for issue in self.issues)
    
    def get_summary(self) -> str:
        """Get a summary of validation results."""
        errors = sum(1 for i in self.issues if i.level == 'error')
        warnings = sum(1 for i in self.issues if i.level == 'warning')
        infos = sum(1 for i in self.issues if i.level == 'info')
        
        status = "PASS" if self.valid else "FAIL"
        return f"Validation {status}: {errors} errors, {warnings} warnings, {infos} info"


class ConfigValidator:
    """Validates configuration for Neural DSL services."""
    
    # Required environment variables for each service
    REQUIRED_VARS = {
        'api': [
            'SECRET_KEY',
            'DATABASE_URL',
        ],
        'dashboard': [
            'SECRET_KEY',
        ],
        'aquarium': [
            'SECRET_KEY',
        ],
        'celery': [
            'REDIS_HOST',
            'CELERY_BROKER_URL',
            'CELERY_RESULT_BACKEND',
        ],
    }
    
    # Optional environment variables with defaults
    OPTIONAL_VARS = {
        'api': {
            'API_HOST': '0.0.0.0',
            'API_PORT': '8000',
            'API_WORKERS': '4',
            'DEBUG': 'false',
            'RATE_LIMIT_ENABLED': 'true',
            'RATE_LIMIT_REQUESTS': '100',
            'RATE_LIMIT_PERIOD': '60',
            'STORAGE_PATH': './neural_storage',
            'EXPERIMENTS_PATH': './neural_experiments',
            'MODELS_PATH': './neural_models',
            'WEBHOOK_TIMEOUT': '30',
            'WEBHOOK_RETRY_LIMIT': '3',
            'CORS_ORIGINS': '["http://localhost:3000","http://localhost:8000"]',
        },
        'dashboard': {
            'DASHBOARD_HOST': '0.0.0.0',
            'DASHBOARD_PORT': '8050',
            'DEBUG': 'false',
        },
        'aquarium': {
            'AQUARIUM_HOST': '0.0.0.0',
            'AQUARIUM_PORT': '8051',
            'DEBUG': 'false',
        },
        'redis': {
            'REDIS_PORT': '6379',
            'REDIS_DB': '0',
        },
    }
    
    # Validation patterns for specific variables
    VALIDATION_PATTERNS = {
        'SECRET_KEY': {
            'pattern': r'^.{32,}$',
            'message': 'SECRET_KEY must be at least 32 characters long',
            'suggestion': 'Generate a secure key with: python -c "import secrets; print(secrets.token_hex(32))"'
        },
        'DATABASE_URL': {
            'pattern': r'^(sqlite|postgresql|mysql):\/\/.+',
            'message': 'DATABASE_URL must be a valid database connection string',
            'suggestion': 'Example: postgresql://user:pass@localhost:5432/dbname'
        },
        'API_PORT': {
            'pattern': r'^\d{2,5}$',
            'validator': lambda v: 1024 <= int(v) <= 65535,
            'message': 'Port must be between 1024 and 65535',
        },
        'DASHBOARD_PORT': {
            'pattern': r'^\d{2,5}$',
            'validator': lambda v: 1024 <= int(v) <= 65535,
            'message': 'Port must be between 1024 and 65535',
        },
        'AQUARIUM_PORT': {
            'pattern': r'^\d{2,5}$',
            'validator': lambda v: 1024 <= int(v) <= 65535,
            'message': 'Port must be between 1024 and 65535',
        },
        'MARKETPLACE_PORT': {
            'pattern': r'^\d{2,5}$',
            'validator': lambda v: 1024 <= int(v) <= 65535,
            'message': 'Port must be between 1024 and 65535',
        },
        'REDIS_HOST': {
            'pattern': r'^[a-zA-Z0-9.-]+$',
            'message': 'REDIS_HOST must be a valid hostname or IP address',
        },
        'REDIS_PORT': {
            'pattern': r'^\d{2,5}$',
            'validator': lambda v: 1 <= int(v) <= 65535,
            'message': 'Port must be between 1 and 65535',
        },
    }
    
    # Dangerous default values that should trigger warnings
    DANGEROUS_DEFAULTS = {
        'SECRET_KEY': [
            'change-me-in-production',
            'change-me-in-production-use-strong-random-key',
            'insecure-secret-key',
            'development',
            'test',
        ],
    }
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize the configuration validator.
        
        Parameters
        ----------
        env_file : str, optional
            Path to .env file to load (defaults to .env in current directory)
        """
        self.env_file = env_file or '.env'
        self.env_vars: Dict[str, str] = {}
        self._load_env()
    
    def _load_env(self):
        """Load environment variables from .env file and system."""
        # Load from system environment
        self.env_vars = dict(os.environ)
        
        # Load from .env file if it exists
        if os.path.exists(self.env_file):
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        # System env takes precedence
                        if key not in self.env_vars:
                            self.env_vars[key] = value
    
    def validate(self, services: Optional[List[str]] = None) -> ValidationResult:
        """Validate configuration for specified services.
        
        Parameters
        ----------
        services : list of str, optional
            List of services to validate. If None, validates all services.
            Valid services: 'api', 'dashboard', 'aquarium', 'celery'
        
        Returns
        -------
        ValidationResult
            Validation result with any issues found
        """
        result = ValidationResult(valid=True)
        
        # Default to all services if none specified
        if services is None:
            services = list(self.REQUIRED_VARS.keys())
        
        # Check for .env file
        if not os.path.exists(self.env_file):
            result.add_warning(
                '.env',
                f'Configuration file {self.env_file} not found',
                'Create a .env file based on .env.example'
            )
        
        # Validate each service
        for service in services:
            self._validate_service(service, result)
        
        # Check for port conflicts
        self._check_port_conflicts(result)
        
        # Check for dangerous defaults
        self._check_dangerous_defaults(result)
        
        return result
    
    def _validate_service(self, service: str, result: ValidationResult):
        """Validate configuration for a specific service."""
        if service not in self.REQUIRED_VARS:
            result.add_warning(service, f'Unknown service: {service}')
            return
        
        # Check required variables
        for var in self.REQUIRED_VARS.get(service, []):
            if var not in self.env_vars or not self.env_vars[var]:
                result.add_error(
                    var,
                    f'Required variable for {service} is missing or empty'
                )
                result.missing_required.append(var)
        
        # Check optional variables and apply defaults
        for var, default in self.OPTIONAL_VARS.get(service, {}).items():
            if var not in self.env_vars or not self.env_vars[var]:
                result.add_info(
                    var,
                    f'Optional variable not set, will use default: {default}'
                )
                result.missing_optional.append(var)
        
        # Validate variable values
        all_vars = list(self.REQUIRED_VARS.get(service, [])) + list(self.OPTIONAL_VARS.get(service, {}).keys())
        for var in all_vars:
            if var in self.env_vars and self.env_vars[var]:
                self._validate_variable(var, self.env_vars[var], result)
    
    def _validate_variable(self, var: str, value: str, result: ValidationResult):
        """Validate a specific variable value."""
        if var not in self.VALIDATION_PATTERNS:
            return
        
        pattern_config = self.VALIDATION_PATTERNS[var]
        
        # Check regex pattern
        if 'pattern' in pattern_config:
            if not re.match(pattern_config['pattern'], value):
                result.add_error(
                    var,
                    pattern_config['message'],
                    pattern_config.get('suggestion')
                )
                result.invalid_values[var] = value
        
        # Check custom validator
        if 'validator' in pattern_config:
            try:
                if not pattern_config['validator'](value):
                    result.add_error(
                        var,
                        pattern_config['message'],
                        pattern_config.get('suggestion')
                    )
                    result.invalid_values[var] = value
            except Exception as e:
                result.add_error(
                    var,
                    f'Validation failed: {str(e)}',
                    pattern_config.get('suggestion')
                )
    
    def _check_port_conflicts(self, result: ValidationResult):
        """Check for port conflicts between services."""
        port_vars = ['API_PORT', 'DASHBOARD_PORT', 'AQUARIUM_PORT', 'MARKETPLACE_PORT', 'REDIS_PORT']
        ports: Dict[str, List[str]] = {}
        
        for var in port_vars:
            if var in self.env_vars:
                port = self.env_vars[var]
                if port not in ports:
                    ports[port] = []
                ports[port].append(var)
        
        for port, vars_list in ports.items():
            if len(vars_list) > 1:
                result.add_error(
                    ', '.join(vars_list),
                    f'Port conflict detected: {", ".join(vars_list)} are all using port {port}',
                    'Assign different ports to each service'
                )
    
    def _check_dangerous_defaults(self, result: ValidationResult):
        """Check for dangerous default values."""
        for var, dangerous_values in self.DANGEROUS_DEFAULTS.items():
            if var in self.env_vars:
                value = self.env_vars[var]
                if value.lower() in [v.lower() for v in dangerous_values]:
                    result.add_error(
                        var,
                        f'Using insecure default value: {value}',
                        'Generate a secure random value for production use'
                    )
    
    def validate_startup(self, services: Optional[List[str]] = None) -> bool:
        """Validate configuration at startup and raise exception if invalid.
        
        Parameters
        ----------
        services : list of str, optional
            List of services to validate
        
        Returns
        -------
        bool
            True if validation passed
        
        Raises
        ------
        RuntimeError
            If validation fails with errors
        """
        result = self.validate(services)
        
        if result.has_errors():
            error_msg = [f"Configuration validation failed:\n"]
            for issue in result.issues:
                if issue.level == 'error':
                    error_msg.append(f"  - {issue.variable}: {issue.message}")
                    if issue.suggestion:
                        error_msg.append(f"    Suggestion: {issue.suggestion}")
            raise RuntimeError('\n'.join(error_msg))
        
        return True
    
    def get_value(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a configuration value.
        
        Parameters
        ----------
        key : str
            Environment variable name
        default : str, optional
            Default value if not found
        
        Returns
        -------
        str or None
            Configuration value
        """
        return self.env_vars.get(key, default)
    
    def export_validation_report(self, result: ValidationResult, output_file: str):
        """Export validation report to a file.
        
        Parameters
        ----------
        result : ValidationResult
            Validation result to export
        output_file : str
            Path to output file
        """
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Neural DSL Configuration Validation Report\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Summary: {result.get_summary()}\n\n")
            
            if result.missing_required:
                f.write("Missing Required Variables:\n")
                for var in result.missing_required:
                    f.write(f"  - {var}\n")
                f.write("\n")
            
            if result.missing_optional:
                f.write("Missing Optional Variables (defaults will be used):\n")
                for var in result.missing_optional:
                    f.write(f"  - {var}\n")
                f.write("\n")
            
            if result.invalid_values:
                f.write("Invalid Values:\n")
                for var, value in result.invalid_values.items():
                    f.write(f"  - {var}: {value}\n")
                f.write("\n")
            
            if result.issues:
                f.write("Issues:\n")
                for issue in result.issues:
                    f.write(f"  [{issue.level.upper()}] {issue.variable}\n")
                    f.write(f"    {issue.message}\n")
                    if issue.suggestion:
                        f.write(f"    Suggestion: {issue.suggestion}\n")
                    f.write("\n")
