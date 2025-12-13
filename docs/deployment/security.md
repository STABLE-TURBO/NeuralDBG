# Security Best Practices

Production security guidelines for Neural DSL deployment.

## Table of Contents

- [Security Checklist](#security-checklist)
- [Authentication & Authorization](#authentication--authorization)
- [Network Security](#network-security)
- [Data Security](#data-security)
- [Container Security](#container-security)
- [Secrets Management](#secrets-management)
- [Monitoring & Auditing](#monitoring--auditing)
- [Compliance](#compliance)

## Security Checklist

### Pre-Deployment

- [ ] Change all default passwords and API keys
- [ ] Generate strong random secrets (minimum 32 characters)
- [ ] Configure SSL/TLS with valid certificates
- [ ] Set up secrets management (Vault, AWS Secrets Manager, etc.)
- [ ] Review and configure firewall rules
- [ ] Enable authentication for all services
- [ ] Configure rate limiting
- [ ] Set up CORS policies
- [ ] Enable audit logging
- [ ] Configure backup encryption

### Post-Deployment

- [ ] Conduct security audit
- [ ] Perform penetration testing
- [ ] Set up intrusion detection
- [ ] Configure log aggregation
- [ ] Enable monitoring and alerting
- [ ] Document incident response procedures
- [ ] Set up automated security scanning
- [ ] Configure vulnerability management
- [ ] Establish credential rotation schedule
- [ ] Review access controls quarterly

## Authentication & Authorization

### API Key Authentication

```python
# neural/api/auth.py
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import secrets

API_KEY_HEADER = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key from request header."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key"
        )
    
    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, os.getenv("ADMIN_API_KEY")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return api_key
```

### JWT Authentication

```python
# neural/api/jwt_auth.py
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: timedelta = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str = Depends(oauth2_scheme)):
    """Verify JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    return username
```

### Role-Based Access Control (RBAC)

```python
# neural/api/rbac.py
from enum import Enum
from typing import List
from fastapi import Depends, HTTPException, status

class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"

class Permission(str, Enum):
    READ_MODELS = "read:models"
    WRITE_MODELS = "write:models"
    DELETE_MODELS = "delete:models"
    TRAIN_MODELS = "train:models"
    DEPLOY_MODELS = "deploy:models"

ROLE_PERMISSIONS = {
    Role.ADMIN: [
        Permission.READ_MODELS,
        Permission.WRITE_MODELS,
        Permission.DELETE_MODELS,
        Permission.TRAIN_MODELS,
        Permission.DEPLOY_MODELS,
    ],
    Role.USER: [
        Permission.READ_MODELS,
        Permission.WRITE_MODELS,
        Permission.TRAIN_MODELS,
    ],
    Role.VIEWER: [
        Permission.READ_MODELS,
    ],
}

def require_permission(required_permission: Permission):
    """Decorator to require specific permission."""
    def permission_checker(user: dict = Depends(get_current_user)):
        user_role = Role(user.get("role"))
        permissions = ROLE_PERMISSIONS.get(user_role, [])
        
        if required_permission not in permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        return user
    
    return permission_checker
```

### Multi-Factor Authentication (MFA)

```python
# neural/api/mfa.py
import pyotp
from fastapi import HTTPException, status

class MFAService:
    def __init__(self):
        self.issuer_name = "Neural DSL"
    
    def generate_secret(self, username: str) -> dict:
        """Generate TOTP secret for user."""
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)
        
        provisioning_uri = totp.provisioning_uri(
            name=username,
            issuer_name=self.issuer_name
        )
        
        return {
            "secret": secret,
            "provisioning_uri": provisioning_uri,
        }
    
    def verify_token(self, secret: str, token: str) -> bool:
        """Verify TOTP token."""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)
    
    def require_mfa(self, user: dict, token: str):
        """Require MFA verification."""
        if not user.get("mfa_enabled"):
            return True
        
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="MFA token required"
            )
        
        if not self.verify_token(user["mfa_secret"], token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid MFA token"
            )
        
        return True
```

## Network Security

### SSL/TLS Configuration

**Nginx SSL Best Practices:**

```nginx
# nginx-ssl.conf
server {
    listen 443 ssl http2;
    server_name neural.example.com;

    # SSL certificates
    ssl_certificate /etc/ssl/certs/neural.crt;
    ssl_certificate_key /etc/ssl/private/neural.key;
    ssl_trusted_certificate /etc/ssl/certs/ca-chain.crt;

    # SSL protocols and ciphers
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384';
    ssl_prefer_server_ciphers off;

    # SSL optimization
    ssl_session_cache shared:SSL:50m;
    ssl_session_timeout 1d;
    ssl_session_tickets off;

    # OCSP stapling
    ssl_stapling on;
    ssl_stapling_verify on;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';" always;

    # HSTS preload
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
}
```

### Firewall Configuration

**UFW (Ubuntu):**

```bash
# Enable firewall
sudo ufw enable

# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Deny all other incoming
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Rate limiting
sudo ufw limit 22/tcp

# Check status
sudo ufw status verbose
```

**iptables:**

```bash
#!/bin/bash
# iptables-rules.sh

# Flush existing rules
iptables -F
iptables -X
iptables -Z

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (with rate limiting)
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --set
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --update --seconds 60 --hitcount 4 -j DROP
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP/HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Drop invalid packets
iptables -A INPUT -m conntrack --ctstate INVALID -j DROP

# Save rules
iptables-save > /etc/iptables/rules.v4
```

### Network Isolation

**Docker network isolation:**

```yaml
# docker-compose.yml
version: '3.8'

networks:
  frontend:
    driver: bridge
    internal: false
  backend:
    driver: bridge
    internal: true

services:
  nginx:
    networks:
      - frontend
      - backend
  
  api:
    networks:
      - backend
  
  postgres:
    networks:
      - backend
  
  redis:
    networks:
      - backend
```

**Kubernetes NetworkPolicy:**

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
  namespace: neural-prod
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-api-to-db
  namespace: neural-prod
spec:
  podSelector:
    matchLabels:
      app: postgres
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: neural-api
    ports:
    - protocol: TCP
      port: 5432
```

### Rate Limiting

```python
# neural/api/rate_limiter.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import FastAPI, Request

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/minute"],
    storage_uri=os.getenv("REDIS_URL"),
)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/compile")
@limiter.limit("10/minute")
async def compile_model(request: Request):
    """Compile model with rate limiting."""
    pass

@app.post("/api/train")
@limiter.limit("5/hour")
async def train_model(request: Request):
    """Train model with stricter rate limiting."""
    pass
```

## Data Security

### Encryption at Rest

**Database encryption:**

```bash
# PostgreSQL encryption
# Enable transparent data encryption (TDE)
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET ssl_cert_file = '/etc/ssl/certs/server.crt';
ALTER SYSTEM SET ssl_key_file = '/etc/ssl/private/server.key';

# Encrypt specific columns
CREATE EXTENSION pgcrypto;

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100),
    email VARCHAR(255),
    password_hash BYTEA,
    api_key BYTEA
);

-- Encrypt API key on insert
INSERT INTO users (username, email, api_key)
VALUES ('user', 'user@example.com', pgp_sym_encrypt('api-key', 'encryption-key'));

-- Decrypt API key on select
SELECT username, pgp_sym_decrypt(api_key, 'encryption-key') as api_key
FROM users;
```

**File encryption:**

```python
# neural/storage/encryption.py
from cryptography.fernet import Fernet
import os

class FileEncryption:
    def __init__(self):
        key = os.getenv("ENCRYPTION_KEY")
        if not key:
            key = Fernet.generate_key()
        self.cipher = Fernet(key)
    
    def encrypt_file(self, file_path: str, output_path: str):
        """Encrypt file."""
        with open(file_path, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.cipher.encrypt(data)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
    
    def decrypt_file(self, file_path: str, output_path: str):
        """Decrypt file."""
        with open(file_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.cipher.decrypt(encrypted_data)
        
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
```

### Encryption in Transit

**Database connections:**

```python
# Use SSL for PostgreSQL
DATABASE_URL = "postgresql://user:pass@host:5432/db?sslmode=require"

# Use TLS for Redis
REDIS_URL = "rediss://:password@host:6379/0"

# Verify certificates
DATABASE_URL = "postgresql://user:pass@host:5432/db?sslmode=verify-full&sslrootcert=/path/to/ca.crt"
```

### Data Sanitization

```python
# neural/api/sanitize.py
import bleach
import re
from typing import Any

def sanitize_string(value: str, max_length: int = 1000) -> str:
    """Sanitize string input."""
    # Remove HTML tags
    value = bleach.clean(value, tags=[], strip=True)
    
    # Limit length
    value = value[:max_length]
    
    # Remove null bytes
    value = value.replace('\x00', '')
    
    return value.strip()

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal."""
    # Remove directory separators
    filename = filename.replace('/', '').replace('\\', '')
    
    # Remove null bytes
    filename = filename.replace('\x00', '')
    
    # Allow only alphanumeric, dash, underscore, and dot
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
    
    # Prevent hidden files
    if filename.startswith('.'):
        filename = filename[1:]
    
    return filename

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
```

## Container Security

### Image Scanning

```bash
# Install Trivy
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Scan image for vulnerabilities
trivy image neural-dsl:latest

# Scan with severity filter
trivy image --severity HIGH,CRITICAL neural-dsl:latest

# Generate report
trivy image --format json --output report.json neural-dsl:latest
```

### Secure Dockerfile

```dockerfile
# Use specific version, not latest
FROM python:3.10.12-slim-bullseye

# Run as non-root user
RUN groupadd -r neural && useradd -r -g neural neural

# Install security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY --chown=neural:neural requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=neural:neural neural/ neural/

# Remove unnecessary packages
RUN pip uninstall -y pip setuptools

# Switch to non-root user
USER neural

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["python", "-m", "neural.api.main"]
```

### Container Runtime Security

```yaml
# kubernetes-pod-security.yaml
apiVersion: v1
kind: Pod
metadata:
  name: neural-api
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: api
    image: neural-dsl:latest
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
    volumeMounts:
    - name: tmp
      mountPath: /tmp
    resources:
      limits:
        cpu: "2"
        memory: "2Gi"
      requests:
        cpu: "1"
        memory: "1Gi"
  volumes:
  - name: tmp
    emptyDir: {}
```

## Secrets Management

See [Secrets Management](secrets-management.md) for detailed guides on:
- HashiCorp Vault
- AWS Secrets Manager
- Google Secret Manager
- Azure Key Vault
- Kubernetes Secrets

## Monitoring & Auditing

### Audit Logging

```python
# neural/api/audit.py
import logging
from datetime import datetime
from typing import Any, Dict
from fastapi import Request

logger = logging.getLogger("audit")

class AuditLogger:
    def __init__(self):
        handler = logging.FileHandler("/var/log/neural/audit.log")
        handler.setFormatter(logging.Formatter(
            '{"timestamp":"%(asctime)s","level":"%(levelname)s","message":%(message)s}'
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    def log_event(
        self,
        request: Request,
        action: str,
        resource: str,
        result: str,
        details: Dict[str, Any] = None
    ):
        """Log audit event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "resource": resource,
            "result": result,
            "user": getattr(request.state, "user", "anonymous"),
            "ip": request.client.host,
            "method": request.method,
            "path": request.url.path,
            "details": details or {},
        }
        logger.info(event)

# Usage
@app.post("/api/models")
async def create_model(request: Request, model: Model):
    audit = AuditLogger()
    try:
        # Create model
        result = create_model_service(model)
        audit.log_event(request, "CREATE", "model", "SUCCESS", {"model_id": result.id})
        return result
    except Exception as e:
        audit.log_event(request, "CREATE", "model", "FAILURE", {"error": str(e)})
        raise
```

### Security Monitoring

```python
# neural/api/security_monitor.py
from prometheus_client import Counter, Histogram

# Metrics
auth_failures = Counter('auth_failures_total', 'Total authentication failures', ['method'])
api_requests = Histogram('api_request_duration_seconds', 'API request duration', ['endpoint', 'method'])
rate_limit_hits = Counter('rate_limit_hits_total', 'Rate limit violations', ['endpoint'])

def monitor_auth_failure(method: str):
    """Track authentication failures."""
    auth_failures.labels(method=method).inc()

def monitor_rate_limit(endpoint: str):
    """Track rate limit violations."""
    rate_limit_hits.labels(endpoint=endpoint).inc()
```

## Compliance

### GDPR Compliance

```python
# neural/api/gdpr.py
from datetime import datetime, timedelta
from typing import List

class GDPRService:
    def export_user_data(self, user_id: str) -> dict:
        """Export all user data (GDPR Article 20)."""
        return {
            "user_id": user_id,
            "personal_data": self.get_personal_data(user_id),
            "models": self.get_user_models(user_id),
            "experiments": self.get_user_experiments(user_id),
            "activity_log": self.get_user_activity(user_id),
            "exported_at": datetime.utcnow().isoformat(),
        }
    
    def delete_user_data(self, user_id: str):
        """Delete all user data (GDPR Article 17)."""
        # Anonymize or delete
        self.anonymize_user(user_id)
        self.delete_user_models(user_id)
        self.delete_user_experiments(user_id)
        self.anonymize_activity_log(user_id)
    
    def anonymize_logs(self, days_old: int = 90):
        """Anonymize logs older than specified days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        # Implement log anonymization
```

### SOC 2 Compliance

```python
# neural/api/soc2.py
class AccessControl:
    def log_access(self, user: str, resource: str, action: str):
        """Log all access for SOC 2 compliance."""
        pass
    
    def review_access_logs(self, start_date: datetime, end_date: datetime):
        """Generate access review report."""
        pass
    
    def enforce_password_policy(self, password: str) -> bool:
        """Enforce strong password policy."""
        if len(password) < 12:
            return False
        if not any(c.isupper() for c in password):
            return False
        if not any(c.islower() for c in password):
            return False
        if not any(c.isdigit() for c in password):
            return False
        if not any(c in "!@#$%^&*" for c in password):
            return False
        return True
```

## Security Incident Response

### Incident Response Plan

1. **Detection**: Monitor logs and alerts
2. **Containment**: Isolate affected systems
3. **Eradication**: Remove threat and vulnerabilities
4. **Recovery**: Restore systems and verify
5. **Lessons Learned**: Document and improve

### Emergency Procedures

```bash
# Emergency shutdown script
#!/bin/bash
# emergency-shutdown.sh

echo "Emergency shutdown initiated..."

# Stop accepting new requests
kubectl scale deployment neural-api --replicas=0 -n neural-prod

# Block all incoming traffic
iptables -I INPUT -j DROP

# Dump logs for forensics
kubectl logs -n neural-prod --all-containers --timestamps > /var/log/incident-$(date +%Y%m%d).log

# Notify security team
curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK \
  -d '{"text":"🚨 Emergency shutdown initiated - check logs immediately"}'
```

See [Troubleshooting Guide](troubleshooting.md) for security incident procedures.
