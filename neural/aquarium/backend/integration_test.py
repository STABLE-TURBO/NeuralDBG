"""Integration tests for the complete backend bridge workflow."""

import time

import pytest

from .client import create_client


@pytest.fixture
def client():
    """Create test client."""
    return create_client("http://localhost:8000")


@pytest.fixture
def sample_dsl():
    """Sample DSL code for testing."""
    return """
    network TestModel {
        input: (28, 28, 1)
        layers:
            Flatten()
            Dense(64, activation="relu")
            Output(10, activation="softmax")
        optimizer: Adam(learning_rate=0.001)
        loss: categorical_crossentropy
    }
    """


def test_health_check(client):
    """Test backend health check."""
    result = client.health_check()
    assert result["status"] == "healthy"


def test_parse_dsl(client, sample_dsl):
    """Test DSL parsing."""
    result = client.parse(sample_dsl)
    assert result["success"] is True
    assert result["model_data"] is not None
    assert "layers" in result["model_data"]
    assert len(result["model_data"]["layers"]) > 0


def test_shape_propagation(client, sample_dsl):
    """Test shape propagation workflow."""
    parse_result = client.parse(sample_dsl)
    assert parse_result["success"] is True

    model_data = parse_result["model_data"]
    shape_result = client.propagate_shapes(model_data, framework="tensorflow")

    assert shape_result["success"] is True
    assert shape_result["shape_history"] is not None
    assert len(shape_result["shape_history"]) > 0


def test_code_generation_tensorflow(client, sample_dsl):
    """Test TensorFlow code generation."""
    parse_result = client.parse(sample_dsl)
    model_data = parse_result["model_data"]

    code_result = client.generate_code(model_data, backend="tensorflow")

    assert code_result["success"] is True
    assert code_result["code"] is not None
    assert "tensorflow" in code_result["code"].lower()
    assert "keras" in code_result["code"].lower()


def test_code_generation_pytorch(client, sample_dsl):
    """Test PyTorch code generation."""
    parse_result = client.parse(sample_dsl)
    model_data = parse_result["model_data"]

    code_result = client.generate_code(model_data, backend="pytorch")

    assert code_result["success"] is True
    assert code_result["code"] is not None
    assert "torch" in code_result["code"].lower()


def test_compile_pipeline(client, sample_dsl):
    """Test complete compilation pipeline."""
    result = client.compile(sample_dsl, backend="tensorflow")

    assert result["success"] is True
    assert result["code"] is not None
    assert result["model_data"] is not None
    assert result["shape_history"] is not None


def test_training_job_lifecycle(client):
    """Test complete training job lifecycle."""
    code = """
import time
for i in range(3):
    print(f"Step {i+1}")
    time.sleep(0.5)
print("Done!")
"""

    start_result = client.start_job(code, job_name="test_lifecycle")
    assert start_result["success"] is True
    job_id = start_result["job_id"]

    time.sleep(0.5)
    status = client.get_job_status(job_id)
    assert status["job_id"] == job_id
    assert status["status"] in ["running", "completed"]

    final_status = client.wait_for_job(job_id)
    assert final_status["status"] in ["completed", "failed"]

    if final_status["status"] == "completed":
        assert "Done!" in final_status["output"]


def test_list_jobs(client):
    """Test listing jobs."""
    jobs = client.list_jobs()
    assert isinstance(jobs, list)


def test_compile_and_run(client, sample_dsl):
    """Test compile and run workflow."""
    result = client.compile(sample_dsl, backend="tensorflow")
    assert result["success"] is True

    code_with_quick_exit = result["code"] + "\nprint('Quick test complete')"

    job_result = client.start_job(code_with_quick_exit, job_name="quick_test")
    assert job_result["success"] is True

    job_id = job_result["job_id"]
    time.sleep(2)

    status = client.get_job_status(job_id)
    assert status["status"] in ["running", "completed", "failed"]


def test_stop_job(client):
    """Test stopping a running job."""
    code = """
import time
for i in range(100):
    print(f"Step {i}")
    time.sleep(0.5)
"""

    start_result = client.start_job(code, job_name="test_stop")
    job_id = start_result["job_id"]

    time.sleep(1)

    stop_result = client.stop_job(job_id)
    assert stop_result["success"] is True

    time.sleep(0.5)

    status = client.get_job_status(job_id)
    assert status["status"] in ["stopped", "failed"]


def test_error_handling(client):
    """Test error handling for invalid inputs."""
    result = client.parse("invalid dsl code")
    assert result["success"] is False
    assert result["error"] is not None

    invalid_model_data = {"invalid": "data"}
    shape_result = client.propagate_shapes(invalid_model_data)
    assert shape_result["success"] is False


def test_multi_backend_generation(client, sample_dsl):
    """Test code generation for multiple backends."""
    parse_result = client.parse(sample_dsl)
    model_data = parse_result["model_data"]

    backends = ["tensorflow", "pytorch"]

    for backend in backends:
        result = client.generate_code(model_data, backend=backend)
        assert result["success"] is True
        assert result["code"] is not None
        print(f"✓ {backend} code generation successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
