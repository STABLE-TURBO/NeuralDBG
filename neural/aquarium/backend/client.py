"""Python client library for Neural DSL Backend Bridge."""

import asyncio
import json
from typing import Any, Dict, List, Optional

import requests


class NeuralBackendClient:
    """Client for interacting with Neural DSL Backend Bridge."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """Initialize the client.

        Args:
            base_url: Base URL of the backend server
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request to the backend.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            **kwargs: Additional arguments to pass to requests

        Returns:
            Response JSON data

        Raises:
            requests.HTTPError: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

    def health_check(self) -> Dict[str, str]:
        """Check backend health.

        Returns:
            Health status dictionary
        """
        return self._make_request("GET", "/health")

    def parse(self, dsl_code: str, parser_type: str = "network") -> Dict[str, Any]:
        """Parse DSL code.

        Args:
            dsl_code: Neural DSL code to parse
            parser_type: Parser type ('network' or 'research')

        Returns:
            Parse result with model_data
        """
        return self._make_request(
            "POST",
            "/api/parse",
            json={"dsl_code": dsl_code, "parser_type": parser_type}
        )

    def propagate_shapes(
        self, model_data: Dict[str, Any], framework: str = "tensorflow"
    ) -> Dict[str, Any]:
        """Propagate shapes through model.

        Args:
            model_data: Parsed model data
            framework: Framework name ('tensorflow' or 'pytorch')

        Returns:
            Shape propagation results
        """
        return self._make_request(
            "POST",
            "/api/shape-propagation",
            json={"model_data": model_data, "framework": framework}
        )

    def generate_code(
        self,
        model_data: Dict[str, Any],
        backend: str = "tensorflow",
        best_params: Optional[Dict[str, Any]] = None,
        auto_flatten_output: bool = False,
    ) -> Dict[str, Any]:
        """Generate backend-specific code.

        Args:
            model_data: Parsed model data
            backend: Backend name ('tensorflow', 'pytorch', or 'onnx')
            best_params: Optional HPO best parameters
            auto_flatten_output: Auto-flatten before Dense/Output layers

        Returns:
            Code generation results
        """
        return self._make_request(
            "POST",
            "/api/generate-code",
            json={
                "model_data": model_data,
                "backend": backend,
                "best_params": best_params,
                "auto_flatten_output": auto_flatten_output,
            }
        )

    def compile(
        self,
        dsl_code: str,
        backend: str = "tensorflow",
        parser_type: str = "network",
        auto_flatten_output: bool = False,
    ) -> Dict[str, Any]:
        """Complete compilation pipeline.

        Args:
            dsl_code: Neural DSL code
            backend: Backend name ('tensorflow', 'pytorch', or 'onnx')
            parser_type: Parser type ('network' or 'research')
            auto_flatten_output: Auto-flatten before Dense/Output layers

        Returns:
            Compilation results with code and model data
        """
        return self._make_request(
            "POST",
            "/api/compile",
            json={
                "dsl_code": dsl_code,
                "backend": backend,
                "parser_type": parser_type,
                "auto_flatten_output": auto_flatten_output,
            }
        )

    def start_job(
        self,
        code: str,
        job_name: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Start a training job.

        Args:
            code: Python code to execute
            job_name: Optional job name
            env_vars: Optional environment variables

        Returns:
            Job start result with job_id
        """
        return self._make_request(
            "POST",
            "/api/jobs/start",
            json={"code": code, "job_name": job_name, "env_vars": env_vars}
        )

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status.

        Args:
            job_id: Job identifier

        Returns:
            Job status information
        """
        return self._make_request("GET", f"/api/jobs/{job_id}/status")

    def stop_job(self, job_id: str) -> Dict[str, Any]:
        """Stop a running job.

        Args:
            job_id: Job identifier

        Returns:
            Stop result
        """
        return self._make_request("POST", f"/api/jobs/{job_id}/stop")

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs.

        Returns:
            List of job information dictionaries
        """
        result = self._make_request("GET", "/api/jobs")
        return result.get("jobs", [])

    def wait_for_job(self, job_id: str, poll_interval: float = 1.0) -> Dict[str, Any]:
        """Wait for a job to complete.

        Args:
            job_id: Job identifier
            poll_interval: Polling interval in seconds

        Returns:
            Final job status
        """
        import time

        while True:
            status = self.get_job_status(job_id)
            if status["status"] in ["completed", "failed", "stopped"]:
                return status
            time.sleep(poll_interval)

    async def watch_job(self, job_id: str, callback=None):
        """Watch a job via WebSocket for real-time updates.

        Args:
            job_id: Job identifier
            callback: Optional callback function for status updates

        Requires websockets library: pip install websockets
        """
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets library required for watch_job. "
                "Install with: pip install websockets"
            )

        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        uri = f"{ws_url}/ws/jobs/{job_id}"

        async with websockets.connect(uri) as websocket:
            while True:
                message = await websocket.recv()
                data = json.loads(message)

                if callback:
                    callback(data)

                if data["status"] in ["completed", "failed"]:
                    return data

    def compile_and_run(
        self,
        dsl_code: str,
        backend: str = "tensorflow",
        job_name: Optional[str] = None,
        wait: bool = True,
    ) -> Dict[str, Any]:
        """Compile DSL code and run training job.

        Args:
            dsl_code: Neural DSL code
            backend: Backend name
            job_name: Optional job name
            wait: Whether to wait for job completion

        Returns:
            Job result
        """
        compile_result = self.compile(dsl_code, backend=backend)
        if not compile_result["success"]:
            return compile_result

        code = compile_result["code"]
        job_result = self.start_job(code, job_name=job_name)

        if not job_result["success"]:
            return job_result

        job_id = job_result["job_id"]

        if wait:
            return self.wait_for_job(job_id)

        return job_result


def create_client(base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
    """Create a Neural Backend Client.

    Args:
        base_url: Base URL of the backend server
        api_key: Optional API key for authentication

    Returns:
        NeuralBackendClient instance
    """
    return NeuralBackendClient(base_url=base_url, api_key=api_key)
