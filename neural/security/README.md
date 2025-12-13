# Neural DSL Security Module

Standardized authentication and security configuration for all Neural DSL server components.

## Overview

This module provides unified security features across all Neural DSL servers:

- **Dashboard** (port 8050)
- **No-Code Interface** (port 8051)
- **Aquarium Visualization** (port 8052)
- **Monitoring Dashboard** (port 8053)
- **API Server** (port 8000)
- **Marketplace API** (port 5000)
- **Collaboration Server** (port 8080)

## Features

### Authentication
- **HTTP Basic Auth**: Username/password authentication
- **JWT (JSON Web Tokens)**: Token-based authentication with configurable expiration

### Security Middleware
- **CORS**: Cross-Origin Resource Sharing with configurable origins
- **Rate Limiting**: Configurable request limits per time window
- **Security Headers**: Standard security headers (CSP, X-Frame-Options, etc.)
- **SSL/TLS Support**: HTTPS support with certificate configuration

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Authentication
NEURAL_AUTH_ENABLED=true
NEURAL_AUTH_TYPE=basic  # Options: basic, jwt

# HTTP Basic Authentication
NEURAL_AUTH_USERNAME=admin
NEURAL_AUTH_PASSWORD=changeme

# JWT Authentication
NEURAL_JWT_SECRET_KEY=your-secret-key-here
NEURAL_JWT_ALGORITHM=HS256
NEURAL_JWT_EXPIRATION_HOURS=24

# CORS Configuration
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

# SSL/TLS Configuration
NEURAL_SSL_ENABLED=false
NEURAL_SSL_CERT_FILE=/path/to/cert.pem
NEURAL_SSL_KEY_FILE=/path/to/key.pem
```

### YAML Configuration

Alternatively, use `security_config.yaml`:

```yaml
security:
  auth_enabled: true
  auth_type: basic
  
  basic_auth_username: admin
  basic_auth_password: changeme
  
  jwt_secret_key: your-secret-key
  jwt_algorithm: HS256
  jwt_expiration_hours: 24
  
  cors_enabled: true
  cors_origins:
    - http://localhost:3000
    - http://localhost:8000
  
  rate_limit_enabled: true
  rate_limit_requests: 100
  rate_limit_window_seconds: 60
  
  ssl_enabled: false
  ssl_cert_file: /path/to/cert.pem
  ssl_key_file: /path/to/key.pem
```

## Usage

### Basic Setup

All servers automatically load security configuration:

```python
from neural.security import load_security_config, apply_security_middleware

# Load configuration
security_config = load_security_config()

# Apply to Flask app
apply_security_middleware(
    app,
    cors_enabled=security_config.cors_enabled,
    cors_origins=security_config.cors_origins,
    rate_limit_enabled=security_config.rate_limit_enabled,
)
```

### Authentication

#### HTTP Basic Auth

```python
from neural.security import create_basic_auth

auth = create_basic_auth(username='admin', password='secret')
```

#### JWT Authentication

```python
from neural.security import create_jwt_auth

auth = create_jwt_auth(
    secret_key='your-secret-key',
    algorithm='HS256',
    expiration_hours=24
)

# Create a token
token = auth.create_token({'user_id': '123', 'username': 'alice'})

# Verify a token
payload = auth.verify_token(token)
```

### Protecting Routes

```python
from flask import Flask
from neural.security import require_auth, create_basic_auth

app = Flask(__name__)
auth = create_basic_auth('admin', 'secret')

@app.route('/protected')
@require_auth(auth)
def protected_route():
    return {'message': 'Authenticated!'}
```

## Server-Specific Configuration

### Dashboard (8050)

```bash
# Start with authentication
NEURAL_AUTH_ENABLED=true \
NEURAL_AUTH_USERNAME=admin \
NEURAL_AUTH_PASSWORD=secret \
python neural/dashboard/dashboard.py
```

### No-Code Interface (8051)

```bash
NEURAL_AUTH_ENABLED=true \
python neural/no_code/no_code.py
```

### API Server (8000)

```bash
# Use JWT for API authentication
NEURAL_AUTH_ENABLED=true \
NEURAL_AUTH_TYPE=jwt \
NEURAL_JWT_SECRET_KEY=my-secret-key \
python neural/aquarium/backend/api.py
```

### Marketplace API (5000)

```bash
# Protect write operations only
NEURAL_AUTH_ENABLED=true \
python -m neural.marketplace.api
```

### Collaboration Server (8080)

```bash
# Use token-based authentication
NEURAL_AUTH_ENABLED=true \
NEURAL_COLLAB_AUTH_TOKEN=collaboration-secret \
python -m neural.collaboration.server
```

## Security Best Practices

1. **Never commit credentials**: Use environment variables or secure vaults
2. **Use strong passwords**: Minimum 12 characters with mixed case, numbers, symbols
3. **Rotate JWT secrets**: Change JWT secret keys periodically
4. **Enable HTTPS**: Use SSL/TLS certificates in production
5. **Restrict CORS origins**: Don't use `*` in production
6. **Monitor rate limits**: Adjust based on legitimate usage patterns
7. **Use secure tokens**: Generate cryptographically secure random tokens

## Generating Secure Secrets

### JWT Secret Key

```python
import secrets
print(secrets.token_urlsafe(32))
```

### Collaboration Token

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## SSL/TLS Setup

### Generate Self-Signed Certificate (Development)

```bash
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout key.pem -out cert.pem -days 365
```

### Production Certificates

Use Let's Encrypt or your certificate provider:

```bash
NEURAL_SSL_ENABLED=true
NEURAL_SSL_CERT_FILE=/etc/letsencrypt/live/yourdomain.com/fullchain.pem
NEURAL_SSL_KEY_FILE=/etc/letsencrypt/live/yourdomain.com/privkey.pem
```

## Troubleshooting

### Authentication Not Working

1. Check environment variables are loaded
2. Verify credentials match exactly
3. Check request headers include `Authorization`

### CORS Errors

1. Add your frontend origin to `NEURAL_CORS_ORIGINS`
2. Verify `Access-Control-Allow-Credentials` is set correctly
3. Check preflight OPTIONS requests are handled

### Rate Limiting Issues

1. Increase `NEURAL_RATE_LIMIT_REQUESTS` if needed
2. Adjust `NEURAL_RATE_LIMIT_WINDOW_SECONDS`
3. Check client IP is correct for rate limit key

## API Examples

### Dashboard (Basic Auth)

```bash
curl -u admin:secret http://localhost:8050
```

### API Server (JWT)

```python
import requests

# Get token (implement your own auth endpoint)
token = "your-jwt-token"

headers = {'Authorization': f'Bearer {token}'}
response = requests.post(
    'http://localhost:8000/api/ai/chat',
    headers=headers,
    json={'user_input': 'Create a CNN'}
)
```

### Marketplace API

```bash
# List models (no auth)
curl http://localhost:5000/api/models

# Upload model (requires auth)
curl -u admin:secret -X POST \
  http://localhost:5000/api/models/upload \
  -H "Content-Type: application/json" \
  -d '{"name": "my-model", "author": "user"}'
```

## Contributing

When adding new server components:

1. Import security modules
2. Load security configuration
3. Apply security middleware
4. Add authentication decorators to protected routes
5. Support SSL/TLS configuration
6. Document any component-specific security settings
