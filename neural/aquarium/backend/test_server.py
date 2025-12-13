"""Tests for the backend bridge server."""

import pytest
from fastapi.testclient import TestClient

from .server import create_app


@pytest.fixture
def client():
    """Create a test client."""
    app = create_app()
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Neural DSL Backend Bridge"
    assert data["status"] == "running"


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_parse_valid_dsl(client):
    """Test parsing valid DSL code."""
    dsl_code = """
    network TestModel {
        input: (28, 28, 1)
        layers:
            Dense(64)
            Output(10)
        optimizer: Adam(learning_rate=0.001)
        loss: categorical_crossentropy
    }
    """

    response = client.post(
        "/api/parse",
        json={"dsl_code": dsl_code, "parser_type": "network"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["model_data"] is not None
    assert "layers" in data["model_data"]


def test_parse_invalid_dsl(client):
    """Test parsing invalid DSL code."""
    dsl_code = "invalid dsl code"

    response = client.post(
        "/api/parse",
        json={"dsl_code": dsl_code}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["error"] is not None


def test_shape_propagation(client):
    """Test shape propagation."""
    model_data = {
        "input": {"shape": [28, 28, 1]},
        "layers": [
            {"type": "Flatten", "params": {}},
            {"type": "Dense", "params": {"units": 64}},
            {"type": "Output", "params": {"units": 10}},
        ],
    }

    response = client.post(
        "/api/shape-propagation",
        json={"model_data": model_data, "framework": "tensorflow"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["shape_history"] is not None
    assert len(data["shape_history"]) > 0


def test_code_generation_tensorflow(client):
    """Test TensorFlow code generation."""
    model_data = {
        "input": {"shape": [28, 28, 1]},
        "layers": [
            {"type": "Flatten", "params": {}},
            {"type": "Dense", "params": {"units": 64}},
            {"type": "Output", "params": {"units": 10}},
        ],
        "optimizer": {"type": "Adam"},
        "loss": {"value": "categorical_crossentropy"},
    }

    response = client.post(
        "/api/generate-code",
        json={"model_data": model_data, "backend": "tensorflow"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["code"] is not None
    assert "tensorflow" in data["code"]


def test_code_generation_pytorch(client):
    """Test PyTorch code generation."""
    model_data = {
        "input": {"shape": [28, 28, 1]},
        "layers": [
            {"type": "Flatten", "params": {}},
            {"type": "Dense", "params": {"units": 64}},
            {"type": "Output", "params": {"units": 10}},
        ],
        "optimizer": {"type": "Adam"},
        "loss": {"value": "crossentropy"},
    }

    response = client.post(
        "/api/generate-code",
        json={"model_data": model_data, "backend": "pytorch"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["code"] is not None
    assert "torch" in data["code"]


def test_compile_dsl(client):
    """Test complete compilation pipeline."""
    dsl_code = """
    network CompileTest {
        input: (28, 28, 1)
        layers:
            Flatten()
            Dense(64)
            Output(10)
        optimizer: Adam(learning_rate=0.001)
        loss: categorical_crossentropy
    }
    """

    response = client.post(
        "/api/compile",
        json={"dsl_code": dsl_code, "backend": "tensorflow"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["code"] is not None
    assert data["model_data"] is not None


def test_start_training_job(client):
    """Test starting a training job."""
    code = """
import time
print("Test job")
time.sleep(0.1)
"""

    response = client.post(
        "/api/jobs/start",
        json={"code": code, "job_name": "test_job"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "job_id" in data


def test_list_jobs(client):
    """Test listing jobs."""
    response = client.get("/api/jobs")

    assert response.status_code == 200
    data = response.json()
    assert "jobs" in data
    assert isinstance(data["jobs"], list)
