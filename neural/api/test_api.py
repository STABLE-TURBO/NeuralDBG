"""
Test script for Neural API.
"""
import pytest
pytest.skip("API server not available in test environment", allow_module_level=True)

import json
import time
from typing import Dict, Any

import requests

BASE_URL = "http://localhost:8000"
API_KEY = "demo_api_key_12345"

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}


def print_response(response: requests.Response, title: str = "Response"):
    """Print formatted response."""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    print(f"Status Code: {response.status_code}")
    print(f"Headers: {dict(response.headers)}")
    try:
        data = response.json()
        print(f"Body:\n{json.dumps(data, indent=2)}")
    except:
        print(f"Body: {response.text}")
    print(f"{'=' * 80}\n")


def test_health_check():
    """Test health check endpoint."""
    print("\n🏥 Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print_response(response, "Health Check")
    return response.status_code == 200


def test_sync_compilation():
    """Test synchronous compilation."""
    print("\n⚙️ Testing synchronous compilation...")
    
    dsl_code = """
Model SimpleMLP {
    Input(shape=[784])
    Dense(units=128, activation=relu)
    Dropout(rate=0.2)
    Dense(units=64, activation=relu)
    Dense(units=10, activation=softmax)
}
"""
    
    payload = {
        "dsl_code": dsl_code,
        "backend": "tensorflow",
        "dataset": "MNIST",
        "auto_flatten_output": False,
        "enable_hpo": False
    }
    
    response = requests.post(
        f"{BASE_URL}/compile/sync",
        headers=HEADERS,
        json=payload
    )
    
    print_response(response, "Synchronous Compilation")
    
    if response.status_code == 200:
        data = response.json()
        if data.get("compiled_code"):
            print(f"✅ Compilation successful! Code length: {len(data['compiled_code'])} chars")
            return True
    
    print("❌ Compilation failed!")
    return False


def test_async_compilation():
    """Test asynchronous compilation."""
    print("\n⚙️ Testing asynchronous compilation...")
    
    dsl_code = """
Model SimpleCNN {
    Input(shape=[28, 28, 1])
    Conv2D(filters=32, kernel_size=3, activation=relu)
    MaxPooling2D(pool_size=2)
    Conv2D(filters=64, kernel_size=3, activation=relu)
    MaxPooling2D(pool_size=2)
    Flatten()
    Dense(units=128, activation=relu)
    Dense(units=10, activation=softmax)
}
"""
    
    payload = {
        "dsl_code": dsl_code,
        "backend": "pytorch",
        "dataset": "MNIST"
    }
    
    response = requests.post(
        f"{BASE_URL}/compile/",
        headers=HEADERS,
        json=payload
    )
    
    print_response(response, "Async Compilation - Submit")
    
    if response.status_code == 202:
        data = response.json()
        job_id = data.get("job_id")
        print(f"✅ Job submitted! Job ID: {job_id}")
        
        return check_job_status(job_id)
    
    print("❌ Failed to submit compilation job!")
    return False


def test_training_job():
    """Test training job submission."""
    print("\n🏋️ Testing training job...")
    
    dsl_code = """
Model SimpleMLP {
    Input(shape=[784])
    Dense(units=64, activation=relu)
    Dense(units=10, activation=softmax)
}
"""
    
    payload = {
        "dsl_code": dsl_code,
        "backend": "tensorflow",
        "dataset": "MNIST",
        "training_config": {
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.001
        },
        "experiment_name": "test_experiment"
    }
    
    response = requests.post(
        f"{BASE_URL}/jobs/train",
        headers=HEADERS,
        json=payload
    )
    
    print_response(response, "Training Job - Submit")
    
    if response.status_code == 202:
        data = response.json()
        job_id = data.get("job_id")
        print(f"✅ Training job submitted! Job ID: {job_id}")
        
        return check_job_status(job_id, max_wait=60)
    
    print("❌ Failed to submit training job!")
    return False


def check_job_status(job_id: str, max_wait: int = 30) -> bool:
    """
    Check job status until completion.
    
    Args:
        job_id: Job ID to check
        max_wait: Maximum time to wait in seconds
        
    Returns:
        True if job completed successfully
    """
    print(f"\n📊 Checking job status for {job_id}...")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        response = requests.get(
            f"{BASE_URL}/jobs/{job_id}",
            headers=HEADERS
        )
        
        if response.status_code == 200:
            data = response.json()
            status = data.get("status")
            progress = data.get("progress")
            
            print(f"Status: {status} - Progress: {progress}%")
            
            if status == "completed":
                print_response(response, f"Job {job_id} - Completed")
                print(f"✅ Job completed successfully!")
                return True
            elif status == "failed":
                print_response(response, f"Job {job_id} - Failed")
                print(f"❌ Job failed!")
                return False
        
        time.sleep(2)
    
    print(f"⏱️ Timeout waiting for job completion")
    return False


def test_list_experiments():
    """Test listing experiments."""
    print("\n📋 Testing list experiments...")
    
    response = requests.get(
        f"{BASE_URL}/experiments/?skip=0&limit=10",
        headers=HEADERS
    )
    
    print_response(response, "List Experiments")
    
    if response.status_code == 200:
        data = response.json()
        total = data.get("total", 0)
        print(f"✅ Found {total} experiments")
        return True
    
    print("❌ Failed to list experiments!")
    return False


def test_list_models():
    """Test listing models."""
    print("\n📦 Testing list models...")
    
    response = requests.get(
        f"{BASE_URL}/models/?skip=0&limit=10",
        headers=HEADERS
    )
    
    print_response(response, "List Models")
    
    if response.status_code == 200:
        data = response.json()
        total = data.get("total", 0)
        print(f"✅ Found {total} models")
        return True
    
    print("❌ Failed to list models!")
    return False


def run_all_tests():
    """Run all API tests."""
    print("\n" + "=" * 80)
    print("🚀 Starting Neural API Tests")
    print("=" * 80)
    
    tests = [
        ("Health Check", test_health_check),
        ("Synchronous Compilation", test_sync_compilation),
        ("Asynchronous Compilation", test_async_compilation),
        ("Training Job", test_training_job),
        ("List Experiments", test_list_experiments),
        ("List Models", test_list_models),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{test_name}' raised an exception: {str(e)}")
            results[test_name] = False
    
    print("\n" + "=" * 80)
    print("📊 Test Results Summary")
    print("=" * 80)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    print("=" * 80 + "\n")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
