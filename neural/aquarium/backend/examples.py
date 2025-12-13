"""Example usage of the Neural DSL Backend Bridge."""

import asyncio
import json

import requests


def example_parse():
    """Example: Parse DSL code."""
    print("=" * 60)
    print("Example: Parse DSL Code")
    print("=" * 60)

    dsl_code = """
    network SimpleModel {
        input: (28, 28, 1)
        layers:
            Conv2D(filters=32, kernel_size=3, activation="relu")
            MaxPooling2D(pool_size=2)
            Flatten()
            Dense(128, activation="relu")
            Output(10, activation="softmax")
        optimizer: Adam(learning_rate=0.001)
        loss: categorical_crossentropy
    }
    """

    response = requests.post(
        "http://localhost:8000/api/parse",
        json={"dsl_code": dsl_code, "parser_type": "network"},
    )

    result = response.json()
    if result["success"]:
        print("✓ Parsing successful!")
        print(f"Model: {result['model_data'].get('network_name', 'Unknown')}")
        print(f"Layers: {len(result['model_data'].get('layers', []))}")
    else:
        print(f"✗ Parsing failed: {result['error']}")

    return result


def example_shape_propagation(model_data):
    """Example: Propagate shapes through model."""
    print("\n" + "=" * 60)
    print("Example: Shape Propagation")
    print("=" * 60)

    response = requests.post(
        "http://localhost:8000/api/shape-propagation",
        json={"model_data": model_data, "framework": "tensorflow"},
    )

    result = response.json()
    if result["success"]:
        print("✓ Shape propagation successful!")
        print("\nShape History:")
        for entry in result["shape_history"]:
            print(f"  {entry['layer']}: {entry['output_shape']}")

        if result.get("issues"):
            print(f"\nIssues detected: {len(result['issues'])}")
        if result.get("optimizations"):
            print(f"Optimization suggestions: {len(result['optimizations'])}")
    else:
        print(f"✗ Shape propagation failed: {result['error']}")


def example_code_generation(model_data):
    """Example: Generate backend code."""
    print("\n" + "=" * 60)
    print("Example: Code Generation")
    print("=" * 60)

    for backend in ["tensorflow", "pytorch"]:
        response = requests.post(
            "http://localhost:8000/api/generate-code",
            json={"model_data": model_data, "backend": backend},
        )

        result = response.json()
        if result["success"]:
            print(f"✓ {backend.upper()} code generation successful!")
            print(f"  Code length: {len(result['code'])} characters")
        else:
            print(f"✗ {backend.upper()} code generation failed: {result['error']}")


def example_compile():
    """Example: Complete compilation pipeline."""
    print("\n" + "=" * 60)
    print("Example: Complete Compilation")
    print("=" * 60)

    dsl_code = """
    network MNISTModel {
        input: (28, 28, 1)
        layers:
            Conv2D(filters=32, kernel_size=3)
            MaxPooling2D(pool_size=2)
            Flatten()
            Dense(64)
            Output(10)
        optimizer: Adam(learning_rate=0.001)
        loss: categorical_crossentropy
    }
    """

    response = requests.post(
        "http://localhost:8000/api/compile",
        json={"dsl_code": dsl_code, "backend": "tensorflow"},
    )

    result = response.json()
    if result["success"]:
        print("✓ Compilation successful!")
        print(f"  Code generated: {len(result['code'])} characters")
        if result.get("shape_history"):
            print(f"  Shape history entries: {len(result['shape_history'])}")
    else:
        print(f"✗ Compilation failed: {result['error']}")

    return result


def example_training_job():
    """Example: Start and monitor a training job."""
    print("\n" + "=" * 60)
    print("Example: Training Job Management")
    print("=" * 60)

    training_code = """
import time
print("Training job started...")
for epoch in range(5):
    time.sleep(1)
    print(f"Epoch {epoch + 1}/5 - Loss: {0.5 / (epoch + 1):.4f}")
print("Training completed!")
"""

    response = requests.post(
        "http://localhost:8000/api/jobs/start",
        json={"code": training_code, "job_name": "example_training"},
    )

    result = response.json()
    if result["success"]:
        job_id = result["job_id"]
        print(f"✓ Training job started: {job_id}")

        print("\nMonitoring job status...")
        for _ in range(10):
            response = requests.get(f"http://localhost:8000/api/jobs/{job_id}/status")
            status = response.json()

            print(f"  Status: {status['status']}")
            if status["output"]:
                print(f"  Last output: {status['output'].split(chr(10))[-1][:50]}")

            if status["status"] in ["completed", "failed"]:
                if status["status"] == "completed":
                    print("✓ Job completed successfully!")
                else:
                    print(f"✗ Job failed: {status.get('error')}")
                break

            import time
            time.sleep(1)
    else:
        print(f"✗ Failed to start job: {result.get('error')}")


def example_list_jobs():
    """Example: List all jobs."""
    print("\n" + "=" * 60)
    print("Example: List All Jobs")
    print("=" * 60)

    response = requests.get("http://localhost:8000/api/jobs")
    result = response.json()

    jobs = result.get("jobs", [])
    print(f"Total jobs: {len(jobs)}")
    for job in jobs:
        print(f"  {job['job_name']} ({job['job_id'][:8]}...): {job['status']}")


async def example_websocket():
    """Example: WebSocket connection for real-time updates."""
    print("\n" + "=" * 60)
    print("Example: WebSocket Real-Time Updates")
    print("=" * 60)

    try:
        import websockets

        async with websockets.connect("ws://localhost:8000/ws") as websocket:
            print("✓ WebSocket connected")

            await websocket.send("Hello from client!")
            message = await websocket.recv()
            print(f"  Received: {message}")

            await websocket.close()
            print("✓ WebSocket closed")

    except ImportError:
        print("✗ websockets library not installed. Install with: pip install websockets")
    except Exception as e:
        print(f"✗ WebSocket error: {e}")


def main():
    """Run all examples."""
    print("Neural DSL Backend Bridge - Examples")
    print("=" * 60)

    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            print("✗ Backend server is not running!")
            print("  Start the server with: python -m neural.aquarium.backend.run")
            return
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to backend server!")
        print("  Start the server with: python -m neural.aquarium.backend.run")
        return

    result = example_parse()

    if result.get("success"):
        model_data = result["model_data"]

        example_shape_propagation(model_data)
        example_code_generation(model_data)

    example_compile()
    example_training_job()
    example_list_jobs()

    print("\n" + "=" * 60)
    print("WebSocket Example (async)")
    print("=" * 60)
    print("Run separately with: python -c 'import asyncio; from neural.aquarium.backend.examples import example_websocket; asyncio.run(example_websocket())'")

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
