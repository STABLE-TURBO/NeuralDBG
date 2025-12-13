"""Quick start script for Neural DSL Backend Bridge.

This script demonstrates the basic functionality of the backend bridge.
Run the server first with: python -m neural.aquarium.backend.run
"""

import sys
import time

try:
    import requests
except ImportError:
    print("Error: requests library not found.")
    print("Install with: pip install requests")
    sys.exit(1)


def check_server():
    """Check if server is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def main():
    """Run quick start demonstration."""
    print("=" * 70)
    print("Neural DSL Backend Bridge - Quick Start")
    print("=" * 70)

    if not check_server():
        print("\n❌ Backend server is not running!")
        print("\nStart the server with one of these commands:")
        print("  python -m neural.aquarium.backend.run")
        print("  python -m neural.aquarium.backend.cli serve")
        print("  uvicorn neural.aquarium.backend.server:app")
        print("\nThen run this script again.")
        sys.exit(1)

    print("\n✅ Backend server is running\n")

    base_url = "http://localhost:8000"

    dsl_code = """
    network QuickStartModel {
        input: (28, 28, 1)
        layers:
            Conv2D(filters=32, kernel_size=3, activation="relu")
            MaxPooling2D(pool_size=2)
            Flatten()
            Dense(64, activation="relu")
            Output(10, activation="softmax")
        optimizer: Adam(learning_rate=0.001)
        loss: categorical_crossentropy
    }
    """

    print("Step 1: Parse DSL Code")
    print("-" * 70)
    response = requests.post(
        f"{base_url}/api/parse",
        json={"dsl_code": dsl_code, "parser_type": "network"}
    )
    result = response.json()

    if result["success"]:
        print("✅ Parsing successful")
        model_data = result["model_data"]
        print(f"   Network: {model_data.get('network_name', 'Unknown')}")
        print(f"   Layers: {len(model_data.get('layers', []))}")
    else:
        print(f"❌ Parsing failed: {result['error']}")
        return

    print("\nStep 2: Shape Propagation")
    print("-" * 70)
    response = requests.post(
        f"{base_url}/api/shape-propagation",
        json={"model_data": model_data, "framework": "tensorflow"}
    )
    result = response.json()

    if result["success"]:
        print("✅ Shape propagation successful")
        for i, entry in enumerate(result["shape_history"][:3]):
            print(f"   {i+1}. {entry['layer']}: {entry['output_shape']}")
        if len(result["shape_history"]) > 3:
            print(f"   ... and {len(result['shape_history']) - 3} more layers")
    else:
        print(f"❌ Shape propagation failed: {result['error']}")

    print("\nStep 3: Generate TensorFlow Code")
    print("-" * 70)
    response = requests.post(
        f"{base_url}/api/generate-code",
        json={"model_data": model_data, "backend": "tensorflow"}
    )
    result = response.json()

    if result["success"]:
        print("✅ Code generation successful")
        code_lines = result["code"].split("\n")
        print(f"   Generated {len(code_lines)} lines of code")
        print("\n   Preview (first 10 lines):")
        for i, line in enumerate(code_lines[:10], 1):
            print(f"   {i:2d}: {line}")
    else:
        print(f"❌ Code generation failed: {result['error']}")

    print("\nStep 4: Complete Compilation Pipeline")
    print("-" * 70)
    response = requests.post(
        f"{base_url}/api/compile",
        json={"dsl_code": dsl_code, "backend": "pytorch"}
    )
    result = response.json()

    if result["success"]:
        print("✅ Compilation successful (PyTorch)")
        print(f"   Code length: {len(result['code'])} characters")
        print(f"   Shape history: {len(result.get('shape_history', []))} entries")
    else:
        print(f"❌ Compilation failed: {result['error']}")

    print("\nStep 5: Start a Training Job")
    print("-" * 70)

    training_code = """
import time
print("Training started...")
for epoch in range(3):
    time.sleep(0.5)
    print(f"Epoch {epoch+1}/3 - Loss: {1.0 / (epoch+2):.4f}")
print("Training complete!")
"""

    response = requests.post(
        f"{base_url}/api/jobs/start",
        json={"code": training_code, "job_name": "quickstart_demo"}
    )
    result = response.json()

    if result["success"]:
        job_id = result["job_id"]
        print(f"✅ Training job started: {job_id[:16]}...")

        print("\n   Monitoring job (5 seconds)...")
        for _ in range(10):
            time.sleep(0.5)
            response = requests.get(f"{base_url}/api/jobs/{job_id}/status")
            status = response.json()

            if status["status"] == "completed":
                print("\n   ✅ Job completed successfully!")
                if status["output"]:
                    print("\n   Output:")
                    for line in status["output"].split("\n")[-5:]:
                        if line.strip():
                            print(f"      {line}")
                break
            elif status["status"] == "failed":
                print(f"\n   ❌ Job failed: {status.get('error')}")
                break
        else:
            print("\n   Job still running...")
    else:
        print(f"❌ Failed to start job: {result.get('error')}")

    print("\n" + "=" * 70)
    print("Quick Start Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  • View API docs: http://localhost:8000/docs")
    print("  • Run examples: python neural/aquarium/backend/examples.py")
    print("  • Read README: neural/aquarium/backend/README.md")
    print("  • Try the client: from neural.aquarium.backend.client import create_client")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
