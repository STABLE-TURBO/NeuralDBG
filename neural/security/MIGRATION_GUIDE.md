# Security Module Migration Guide

This guide helps you migrate existing Neural DSL server deployments to use the new standardized security module.

## Overview of Changes

The new security module provides:
- Unified authentication across all servers
- Environment-based configuration
- Built-in CORS, rate limiting, and security headers
- Consistent SSL/TLS support
- Backward compatibility with existing deployments

## Migration Steps

### Step 1: Install Dependencies

Ensure you have the required packages:

```bash
pip install pyyaml  # For YAML configuration support
```

### Step 2: Create Environment Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your security settings:

```bash
# Minimal configuration for development
NEURAL_AUTH_ENABLED=false  # Start with auth disabled
NEURAL_CORS_ENABLED=true
NEURAL_CORS_ORIGINS=http://localhost:3000,http://localhost:8000
```

### Step 3: Enable Authentication (Optional)

For production deployments, enable authentication:

```bash
# HTTP Basic Auth (simple setup)
NEURAL_AUTH_ENABLED=true
NEURAL_AUTH_TYPE=basic
NEURAL_AUTH_USERNAME=admin
NEURAL_AUTH_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(16))")
```

Or use JWT for API servers:

```bash
# JWT Auth (for REST APIs)
NEURAL_AUTH_ENABLED=true
NEURAL_AUTH_TYPE=jwt
NEURAL_JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

### Step 4: Update Server Startup

All servers now automatically load security configuration. No code changes needed!

```bash
# Dashboard (port 8050) - already integrated
python neural/dashboard/dashboard.py

# No-Code Interface (port 8051) - already integrated
python neural/no_code/no_code.py

# API Server (port 8000) - already integrated
python neural/aquarium/backend/api.py

# All other servers are also updated
```

### Step 5: Configure CORS Origins

Update CORS origins to match your frontend URLs:

```bash
# Development
NEURAL_CORS_ORIGINS=http://localhost:3000,http://localhost:8051

# Production
NEURAL_CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

### Step 6: Enable SSL/TLS (Production)

For production deployments:

```bash
# Generate or obtain SSL certificates
# Then configure:
NEURAL_SSL_ENABLED=true
NEURAL_SSL_CERT_FILE=/path/to/cert.pem
NEURAL_SSL_KEY_FILE=/path/to/key.pem
```

## Server-Specific Migration

### Dashboard (Port 8050)

**Before:**
```python
# Custom CORS configuration
from flask_cors import CORS
CORS(server, origins=['http://localhost:8050'])
```

**After:**
```bash
# Configuration via environment
NEURAL_CORS_ORIGINS=http://localhost:8050
```

**Access:**
```bash
# Without auth
curl http://localhost:8050

# With basic auth
curl -u admin:password http://localhost:8050

# With JWT
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8050
```

### No-Code Interface (Port 8051)

**Before:**
```python
# No authentication
app.run_server(debug=True, port=8051)
```

**After:**
```bash
# Enable auth in environment
NEURAL_AUTH_ENABLED=true
NEURAL_AUTH_USERNAME=designer
NEURAL_AUTH_PASSWORD=secret123
```

### API Server (Port 8000)

**Before:**
```python
# Custom CORS
CORS(app)
```

**After:**
```bash
# JWT recommended for APIs
NEURAL_AUTH_TYPE=jwt
NEURAL_JWT_SECRET_KEY=your-secret-key
```

**Client Code:**
```python
import requests

# Get JWT token (implement your auth endpoint)
token = get_jwt_token()

headers = {'Authorization': f'Bearer {token}'}
response = requests.post(
    'http://localhost:8000/api/ai/chat',
    headers=headers,
    json={'user_input': 'Create a model'}
)
```

### Marketplace API (Port 5000)

**Before:**
```python
# CORS only
from flask_cors import CORS
CORS(self.app)
```

**After:**
```bash
# Public reads, authenticated writes
NEURAL_AUTH_ENABLED=true
```

Operations requiring authentication:
- Upload model
- Update model
- Delete model

### Collaboration Server (Port 8080)

**Before:**
```python
# No authentication
server = CollaborationServer(host='localhost', port=8080)
```

**After:**
```bash
# Token-based authentication
NEURAL_AUTH_ENABLED=true
NEURAL_COLLAB_AUTH_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

**Client Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8080');

// Send auth message
ws.send(JSON.stringify({
    type: 'auth',
    workspace_id: 'workspace-123',
    user_id: 'user-456',
    username: 'Alice',
    auth_token: 'YOUR_AUTH_TOKEN'
}));
```

## Common Migration Scenarios

### Scenario 1: Development Environment

Keep it simple with minimal security:

```bash
NEURAL_AUTH_ENABLED=false
NEURAL_CORS_ENABLED=true
NEURAL_CORS_ORIGINS=*
NEURAL_RATE_LIMIT_ENABLED=false
NEURAL_SSL_ENABLED=false
```

### Scenario 2: Production Single-Server

Full security for a single server:

```bash
# Enable all security features
NEURAL_AUTH_ENABLED=true
NEURAL_AUTH_TYPE=basic
NEURAL_AUTH_USERNAME=admin
NEURAL_AUTH_PASSWORD=strong-password-here

NEURAL_CORS_ENABLED=true
NEURAL_CORS_ORIGINS=https://yourdomain.com
NEURAL_CORS_ALLOW_CREDENTIALS=true

NEURAL_RATE_LIMIT_ENABLED=true
NEURAL_RATE_LIMIT_REQUESTS=100
NEURAL_RATE_LIMIT_WINDOW_SECONDS=60

NEURAL_SECURITY_HEADERS_ENABLED=true

NEURAL_SSL_ENABLED=true
NEURAL_SSL_CERT_FILE=/etc/ssl/certs/neural.crt
NEURAL_SSL_KEY_FILE=/etc/ssl/private/neural.key
```

### Scenario 3: Production Multi-Server with API Gateway

Use JWT for inter-service authentication:

```bash
# Shared JWT secret across all services
NEURAL_AUTH_ENABLED=true
NEURAL_AUTH_TYPE=jwt
NEURAL_JWT_SECRET_KEY=shared-secret-key-across-all-services

# Per-service CORS
NEURAL_CORS_ORIGINS=https://api-gateway.yourdomain.com
```

### Scenario 4: Kubernetes Deployment

Use secrets for sensitive values:

```yaml
# k8s-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: neural-security
type: Opaque
stringData:
  auth-username: admin
  auth-password: your-password
  jwt-secret: your-jwt-secret
---
# deployment.yaml
env:
  - name: NEURAL_AUTH_ENABLED
    value: "true"
  - name: NEURAL_AUTH_USERNAME
    valueFrom:
      secretKeyRef:
        name: neural-security
        key: auth-username
  - name: NEURAL_AUTH_PASSWORD
    valueFrom:
      secretKeyRef:
        name: neural-security
        key: auth-password
```

## Testing Your Migration

### 1. Test Without Authentication

```bash
# Should work without auth
curl http://localhost:8050
curl http://localhost:8051
curl http://localhost:8000/health
```

### 2. Enable Authentication

```bash
export NEURAL_AUTH_ENABLED=true
export NEURAL_AUTH_USERNAME=test
export NEURAL_AUTH_PASSWORD=test123
```

### 3. Test With Authentication

```bash
# Should fail without auth
curl http://localhost:8050
# Response: 401 Unauthorized

# Should succeed with auth
curl -u test:test123 http://localhost:8050
# Response: 200 OK
```

### 4. Test CORS

```bash
# Test CORS preflight
curl -X OPTIONS \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  http://localhost:8000/api/ai/chat

# Check for CORS headers in response
```

### 5. Test Rate Limiting

```bash
# Send multiple requests quickly
for i in {1..110}; do
  curl http://localhost:8000/health
done

# Should see 429 Too Many Requests after 100 requests
```

## Troubleshooting

### Issue: CORS Errors After Migration

**Solution:**
```bash
# Add your frontend URL to CORS origins
NEURAL_CORS_ORIGINS=http://localhost:3000,https://app.yourdomain.com
```

### Issue: Authentication Not Working

**Check:**
1. Environment variables are loaded: `echo $NEURAL_AUTH_ENABLED`
2. Credentials match exactly (no extra spaces)
3. Authorization header format is correct

### Issue: SSL Certificate Errors

**Solution:**
```bash
# Development: Use self-signed certs
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout key.pem -out cert.pem -days 365

# Production: Use Let's Encrypt
certbot certonly --standalone -d yourdomain.com
```

### Issue: Rate Limiting Too Strict

**Solution:**
```bash
# Increase limits
NEURAL_RATE_LIMIT_REQUESTS=500
NEURAL_RATE_LIMIT_WINDOW_SECONDS=60
```

### Issue: WebSocket Authentication Failing

**Solution:**
```bash
# Generate and set collaboration token
export NEURAL_COLLAB_AUTH_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# Use same token in client connection
```

## Rollback Plan

If you need to rollback:

1. Disable authentication:
```bash
NEURAL_AUTH_ENABLED=false
```

2. Remove security middleware (temporary):
```python
# Comment out in your server file
# apply_security_middleware(app, ...)
```

3. Restart servers

4. Fix issues and re-enable security

## Best Practices

1. **Use environment variables** for all sensitive configuration
2. **Never commit** `.env` or security config files
3. **Rotate secrets** regularly (JWT keys, passwords)
4. **Enable SSL/TLS** in production
5. **Monitor rate limits** and adjust as needed
6. **Use JWT** for API servers, Basic Auth for dashboards
7. **Restrict CORS origins** in production
8. **Test thoroughly** in staging before production deployment

## Need Help?

- Check the [Security README](README.md) for detailed documentation
- Review example configurations in `.env.example`
- Open an issue on GitHub for support
