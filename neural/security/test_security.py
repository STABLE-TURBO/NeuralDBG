"""
Quick test script to verify security module functionality.

Run this script to test authentication, rate limiting, and CORS.
"""

import os
import time
from flask import Flask

# Set test environment variables
os.environ['NEURAL_AUTH_ENABLED'] = 'true'
os.environ['NEURAL_AUTH_TYPE'] = 'basic'
os.environ['NEURAL_AUTH_USERNAME'] = 'testuser'
os.environ['NEURAL_AUTH_PASSWORD'] = 'testpass'
os.environ['NEURAL_CORS_ENABLED'] = 'true'
os.environ['NEURAL_CORS_ORIGINS'] = 'http://localhost:3000'
os.environ['NEURAL_RATE_LIMIT_ENABLED'] = 'true'
os.environ['NEURAL_RATE_LIMIT_REQUESTS'] = '5'
os.environ['NEURAL_RATE_LIMIT_WINDOW_SECONDS'] = '10'

from neural.security import (
    load_security_config,
    create_basic_auth,
    create_jwt_auth,
    require_auth,
    apply_security_middleware,
)


def test_config_loading():
    """Test loading security configuration."""
    print("Testing configuration loading...")
    
    config = load_security_config()
    
    assert config.auth_enabled == True
    assert config.auth_type == 'basic'
    assert config.basic_auth_username == 'testuser'
    assert config.basic_auth_password == 'testpass'
    assert config.cors_enabled == True
    assert 'http://localhost:3000' in config.cors_origins
    assert config.rate_limit_enabled == True
    assert config.rate_limit_requests == 5
    
    print("✓ Configuration loading works!")


def test_basic_auth():
    """Test HTTP Basic Authentication."""
    print("\nTesting HTTP Basic Authentication...")
    
    auth = create_basic_auth('testuser', 'testpass')
    
    # Test valid credentials
    valid_creds = {'username': 'testuser', 'password': 'testpass'}
    assert auth.check_auth(valid_creds) == True
    
    # Test invalid credentials
    invalid_creds = {'username': 'testuser', 'password': 'wrong'}
    assert auth.check_auth(invalid_creds) == False
    
    print("✓ Basic authentication works!")


def test_jwt_auth():
    """Test JWT authentication."""
    print("\nTesting JWT Authentication...")
    
    auth = create_jwt_auth('secret-key', 'HS256', 24)
    
    # Create token
    payload = {'user_id': '123', 'username': 'alice'}
    token = auth.create_token(payload)
    
    assert token is not None
    assert len(token.split('.')) == 3  # JWT has 3 parts
    
    # Verify token
    decoded = auth.verify_token(token)
    assert decoded is not None
    assert decoded['user_id'] == '123'
    assert decoded['username'] == 'alice'
    
    # Test invalid token
    invalid_decoded = auth.verify_token('invalid.token.here')
    assert invalid_decoded is None
    
    print("✓ JWT authentication works!")


def test_flask_integration():
    """Test Flask integration."""
    print("\nTesting Flask integration...")
    
    app = Flask(__name__)
    config = load_security_config()
    
    # Apply security middleware
    apply_security_middleware(
        app,
        cors_enabled=config.cors_enabled,
        cors_origins=config.cors_origins,
        rate_limit_enabled=config.rate_limit_enabled,
        rate_limit_requests=config.rate_limit_requests,
        rate_limit_window_seconds=config.rate_limit_window_seconds,
    )
    
    # Create test route
    auth = create_basic_auth('testuser', 'testpass')
    
    @app.route('/protected')
    @require_auth(auth)
    def protected():
        return {'message': 'Success'}
    
    @app.route('/public')
    def public():
        return {'message': 'Public'}
    
    # Test with Flask test client
    client = app.test_client()
    
    # Test public endpoint
    response = client.get('/public')
    assert response.status_code == 200
    
    # Test protected endpoint without auth
    response = client.get('/protected')
    assert response.status_code == 401
    
    # Test protected endpoint with auth
    from base64 import b64encode
    credentials = b64encode(b'testuser:testpass').decode('utf-8')
    response = client.get(
        '/protected',
        headers={'Authorization': f'Basic {credentials}'}
    )
    assert response.status_code == 200
    
    print("✓ Flask integration works!")


def test_rate_limiting():
    """Test rate limiting."""
    print("\nTesting rate limiting...")
    
    app = Flask(__name__)
    
    apply_security_middleware(
        app,
        rate_limit_enabled=True,
        rate_limit_requests=3,
        rate_limit_window_seconds=5,
    )
    
    @app.route('/test')
    def test_route():
        return {'message': 'OK'}
    
    client = app.test_client()
    
    # First 3 requests should succeed
    for i in range(3):
        response = client.get('/test')
        assert response.status_code == 200
    
    # 4th request should be rate limited
    response = client.get('/test')
    assert response.status_code == 429
    
    print("✓ Rate limiting works!")


def test_cors():
    """Test CORS headers."""
    print("\nTesting CORS...")
    
    app = Flask(__name__)
    
    apply_security_middleware(
        app,
        cors_enabled=True,
        cors_origins=['http://localhost:3000'],
        cors_methods=['GET', 'POST'],
        cors_allow_headers=['Content-Type', 'Authorization'],
    )
    
    @app.route('/test')
    def test_route():
        return {'message': 'OK'}
    
    client = app.test_client()
    
    # Test CORS headers
    response = client.get(
        '/test',
        headers={'Origin': 'http://localhost:3000'}
    )
    
    assert 'Access-Control-Allow-Origin' in response.headers
    assert response.headers['Access-Control-Allow-Origin'] == 'http://localhost:3000'
    
    print("✓ CORS works!")


def test_security_headers():
    """Test security headers."""
    print("\nTesting security headers...")
    
    app = Flask(__name__)
    
    apply_security_middleware(
        app,
        security_headers_enabled=True,
    )
    
    @app.route('/test')
    def test_route():
        return {'message': 'OK'}
    
    client = app.test_client()
    response = client.get('/test')
    
    assert 'X-Content-Type-Options' in response.headers
    assert response.headers['X-Content-Type-Options'] == 'nosniff'
    assert 'X-Frame-Options' in response.headers
    assert 'X-XSS-Protection' in response.headers
    
    print("✓ Security headers work!")


if __name__ == '__main__':
    print("=" * 60)
    print("Neural DSL Security Module Test Suite")
    print("=" * 60)
    
    try:
        test_config_loading()
        test_basic_auth()
        test_jwt_auth()
        test_flask_integration()
        test_rate_limiting()
        test_cors()
        test_security_headers()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise
