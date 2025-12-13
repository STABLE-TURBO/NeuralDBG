# Standardized Security Implementation

This document provides a comprehensive overview of the standardized authentication and security configuration across all Neural DSL server components.

## Overview

All Neural DSL servers now use a unified security module that provides:

- **Consistent Authentication**: HTTP Basic Auth and JWT support
- **Environment-Based Configuration**: Secure credential management via environment variables
- **CORS Management**: Configurable cross-origin resource sharing
- **Rate Limiting**: Protection against abuse and DoS attacks
- **Security Headers**: Standard HTTP security headers
- **SSL/TLS Support**: HTTPS configuration for production deployments

## Implemented Servers

| Server Component | Port | Auth Type | Status |
|-----------------|------|-----------|--------|
| Dashboard (NeuralDbg) | 8050 | Basic/JWT | ✅ Implemented |
| No-Code Interface | 8051 | Basic/JWT | ✅ Implemented |
| Aquarium Visualization | 8052 | Basic/JWT | ✅ Implemented |
| Monitoring Dashboard | 8053 | Basic/JWT | ✅ Implemented |
| API Server | 8000 | Basic/JWT | ✅ Implemented |
| Marketplace API | 5000 | Basic/JWT | ✅ Implemented |
| Collaboration Server | 8080 | Token | ✅ Implemented |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Security Configuration                   │
│  ┌──────────────────┐      ┌─────────────────────────┐ │
│  │  Environment     │      │    YAML Config          │ │
│  │  Variables       │  or  │    (security_config.yaml)│ │
│  │  (.env)          │      │                         │ │
│  └──────────────────┘      └─────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Neural Security Module                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ config.py    │  │   auth.py    │  │middleware.py │ │
│  │              │  │              │  │              │ │
│  │ - Load env   │  │ - BasicAuth  │  │ - CORS       │ │
│  │ - Parse YAML │  │ - JWT        │  │ - Rate Limit │ │
│  │ - Defaults   │  │ - Decorators │  │ - Headers    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Server Components                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────┐ │
│  │Dashboard│  │No-Code  │  │Aquarium │  │   API    │ │
│  │ :8050   │  │ :8051   │  │ :8052   │  │  :8000   │ │
│  └─────────┘  └─────────┘  └─────────┘  └──────────┘ │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐               │
│  │Monitor  │  │Marketplc│  │Collab   │               │
│  │ :8053   │  │ :5000   │  │ :8080   │               │
│  └─────────┘  └─────────┘  └─────────┘               │
└─────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables (.env)

```bash
# Authentication
NEURAL_AUTH_ENABLED=true
NEURAL_AUTH_TYPE=basic  # Options: basic, jwt

# HTTP Basic Authentication
NEURAL_AUTH_USERNAME=admin
NEURAL_AUTH_PASSWORD=secure-password

# JWT Authentication
NEURAL_JWT_SECRET_KEY=your-secret-key
NEURAL_JWT_ALGORITHM=HS256
NEURAL_JWT_EXPIRATION_HOURS=24

# CORS
NEURAL_CORS_ENABLED=true
NEURAL_CORS_ORIGINS=http://localhost:3000,http://localhost:8000
NEURAL_CORS_METHODS=GET,POST,PUT,DELETE,OPTIONS
NEURAL_CORS_ALLOW_HEADERS=Content-Type,Authorization
NEURAL_CORS_ALLOW_CREDENTIALS=true

# Rate Limiting
NEURAL_RATE_LIMIT_ENABLED=true
NEURAL_RATE_LIMIT_REQUESTS=100
NEURAL_RATE_LIMIT_WINDOW_SECONDS=60

# Security Headers
NEURAL_SECURITY_HEADERS_ENABLED=true

# SSL/TLS
NEURAL_SSL_ENABLED=false
NEURAL_SSL_CERT_FILE=/path/to/cert.pem
NEURAL_SSL_KEY_FILE=/path/to/key.pem
```

### YAML Configuration (Optional)

```yaml
security:
  auth_enabled: true
  auth_type: basic
  basic_auth_username: admin
  basic_auth_password: secure-password
  jwt_secret_key: your-secret-key
  cors_enabled: true
  cors_origins:
    - http://localhost:3000
    - http://localhost:8000
  rate_limit_enabled: true
  rate_limit_requests: 100
  ssl_enabled: false
```

## Implementation Details

### 1. Security Module (`neural/security/`)

#### `config.py`
- Loads configuration from environment variables
- Supports YAML configuration files
- Provides sensible defaults
- Priority: Environment > YAML > Defaults

#### `auth.py`
- HTTP Basic Authentication implementation
- JWT token creation and verification
- Authentication decorators for Flask routes
- Token expiration and validation

#### `middleware.py`
- CORS middleware with configurable origins
- Rate limiting per IP address
- Security headers (CSP, X-Frame-Options, etc.)
- Easy Flask integration

### 2. Server Integration

Each server follows this pattern:

```python
from neural.security import (
    load_security_config,
    create_basic_auth,
    create_jwt_auth,
    apply_security_middleware,
)

# Load configuration
security_config = load_security_config()

# Create Flask/Dash app
app = Flask(__name__)

# Apply security middleware
apply_security_middleware(
    app,
    cors_enabled=security_config.cors_enabled,
    cors_origins=security_config.cors_origins,
    rate_limit_enabled=security_config.rate_limit_enabled,
    rate_limit_requests=security_config.rate_limit_requests,
    rate_limit_window_seconds=security_config.rate_limit_window_seconds,
    security_headers_enabled=security_config.security_headers_enabled,
)

# Setup authentication
if security_config.auth_enabled:
    if security_config.auth_type == 'jwt':
        auth = create_jwt_auth(
            security_config.jwt_secret_key,
            security_config.jwt_algorithm,
            security_config.jwt_expiration_hours
        )
    else:
        auth = create_basic_auth(
            security_config.basic_auth_username,
            security_config.basic_auth_password
        )

# Protect routes
@app.route('/protected')
@require_auth(auth)
def protected_route():
    return {'message': 'Authenticated!'}
```

## Security Features

### Authentication

#### HTTP Basic Auth
- Simple username/password authentication
- Suitable for dashboards and internal tools
- Credentials hashed with HMAC comparison
- Automatic WWW-Authenticate headers

#### JWT (JSON Web Tokens)
- Stateless token-based authentication
- Configurable expiration time
- HMAC-SHA256 signing
- Suitable for REST APIs

### CORS (Cross-Origin Resource Sharing)

- Configurable allowed origins
- Support for credentials
- Automatic OPTIONS handling
- Per-method configuration

### Rate Limiting

- Per-IP address tracking
- Sliding window algorithm
- Configurable limits and windows
- Automatic retry-after headers
- Rate limit headers in responses

### Security Headers

Automatically applied headers:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: SAMEORIGIN`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security` (when SSL enabled)
- `Content-Security-Policy`

### SSL/TLS Support

- Certificate and key file configuration
- Automatic HTTPS binding
- Supports Let's Encrypt certificates
- Production-ready SSL context

## Usage Examples

### Starting Servers

```bash
# Development (no auth)
NEURAL_AUTH_ENABLED=false python neural/dashboard/dashboard.py

# Production (with auth)
NEURAL_AUTH_ENABLED=true \
NEURAL_AUTH_USERNAME=admin \
NEURAL_AUTH_PASSWORD=secret \
NEURAL_SSL_ENABLED=true \
NEURAL_SSL_CERT_FILE=/etc/ssl/cert.pem \
NEURAL_SSL_KEY_FILE=/etc/ssl/key.pem \
python neural/dashboard/dashboard.py
```

### Client Authentication

#### HTTP Basic Auth
```bash
curl -u admin:secret http://localhost:8050
```

```python
import requests
response = requests.get(
    'http://localhost:8050',
    auth=('admin', 'secret')
)
```

#### JWT
```python
import requests

# Get JWT token
token = 'your-jwt-token'

headers = {'Authorization': f'Bearer {token}'}
response = requests.post(
    'http://localhost:8000/api/ai/chat',
    headers=headers,
    json={'user_input': 'Create a CNN'}
)
```

## Testing

Run the test suite:

```bash
python neural/security/test_security.py
```

Tests cover:
- Configuration loading
- Basic authentication
- JWT authentication
- Flask integration
- Rate limiting
- CORS headers
- Security headers

## Security Best Practices

1. **Credentials Management**
   - Never commit `.env` files
   - Use strong passwords (12+ characters)
   - Rotate JWT secrets regularly
   - Use environment variables in production

2. **Network Security**
   - Enable SSL/TLS in production
   - Use reverse proxy (nginx/Apache) for SSL termination
   - Restrict CORS origins (no `*` in production)
   - Enable rate limiting

3. **Authentication**
   - Use JWT for APIs
   - Use Basic Auth for dashboards
   - Implement token refresh for long-lived sessions
   - Log authentication failures

4. **Monitoring**
   - Monitor rate limit violations
   - Track authentication failures
   - Alert on security header misconfigurations
   - Review access logs regularly

## Documentation

- **README**: [neural/security/README.md](neural/security/README.md)
- **Migration Guide**: [neural/security/MIGRATION_GUIDE.md](neural/security/MIGRATION_GUIDE.md)
- **Example Config**: [.env.example](.env.example)
- **YAML Example**: [neural/security/security_config.yaml.example](neural/security/security_config.yaml.example)

## Files Created/Modified

### New Files
- `neural/security/__init__.py` - Module exports
- `neural/security/config.py` - Configuration management
- `neural/security/auth.py` - Authentication implementations
- `neural/security/middleware.py` - Security middleware
- `neural/security/README.md` - Detailed documentation
- `neural/security/MIGRATION_GUIDE.md` - Migration instructions
- `neural/security/test_security.py` - Test suite
- `neural/security/security_config.yaml.example` - YAML example
- `.env.example` - Environment variable example
- `SECURITY_IMPLEMENTATION.md` - This document

### Modified Files
- `neural/dashboard/dashboard.py` - Added security integration
- `neural/no_code/no_code.py` - Added security integration
- `neural/visualization/aquarium_server.py` - Added security integration
- `neural/monitoring/dashboard.py` - Added security integration
- `neural/aquarium/backend/api.py` - Added security integration
- `neural/marketplace/api.py` - Added security integration
- `neural/collaboration/server.py` - Added security integration
- `.gitignore` - Added security file exclusions

## Future Enhancements

Potential improvements for future versions:

1. **OAuth2/OIDC Support** - Integration with OAuth providers
2. **API Key Authentication** - Simple key-based auth for APIs
3. **Session Management** - Stateful sessions with Redis
4. **2FA/MFA** - Two-factor authentication support
5. **Role-Based Access Control (RBAC)** - Granular permissions
6. **Audit Logging** - Security event logging
7. **IP Whitelisting** - Network-level access control
8. **Certificate Pinning** - Enhanced SSL security

## Compliance

This implementation helps meet security requirements for:

- **OWASP Top 10** - Protection against common vulnerabilities
- **SOC 2** - Access control and authentication
- **HIPAA** - Secure data transmission
- **PCI DSS** - Network security requirements
- **GDPR** - Data protection by design

## Support

For issues or questions:

1. Check the [README](neural/security/README.md)
2. Review the [Migration Guide](neural/security/MIGRATION_GUIDE.md)
3. Run tests: `python neural/security/test_security.py`
4. Open an issue on GitHub

## License

This security module is part of Neural DSL and follows the same license as the main project.
