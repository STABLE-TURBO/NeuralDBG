"""Process manager for running training jobs."""

import asyncio
import io
import logging
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProcessManager:
    """Manages training job processes."""

    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.max_output_lines = 1000

    async def start_job(
        self,
        code: str,
        job_name: Optional[str] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> str:
        """Start a new training job.

        Args:
            code: Python code to execute
            job_name: Optional job name for identification
            env_vars: Optional environment variables

        Returns:
            job_id: Unique identifier for the job
        """
        job_id = str(uuid.uuid4())
        if job_name is None:
            job_name = f"job_{job_id[:8]}"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as temp_file:
            temp_file.write(code)
            script_path = temp_file.name

        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            self.jobs[job_id] = {
                "job_id": job_id,
                "job_name": job_name,
                "status": "running",
                "process": process,
                "script_path": script_path,
                "output": [],
                "error": None,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
            }

            asyncio.create_task(self._monitor_job(job_id))

            logger.info(f"Started job {job_id} ({job_name})")
            return job_id

        except Exception as e:
            logger.error(f"Failed to start job: {e}", exc_info=True)
            if os.path.exists(script_path):
                os.unlink(script_path)
            raise

    async def _monitor_job(self, job_id: str):
        """Monitor job output and status."""
        job_info = self.jobs.get(job_id)
        if not job_info:
            return

        process = job_info["process"]
        output_lines = []
        error_lines = []

        try:
            async def read_stream(stream, lines_list):
                async for line in stream:
                    decoded_line = line.decode().strip()
                    lines_list.append(decoded_line)
                    if len(lines_list) > self.max_output_lines:
                        lines_list.pop(0)
                    job_info["output"] = lines_list

            await asyncio.gather(
                read_stream(process.stdout, output_lines),
                read_stream(process.stderr, error_lines),
            )

            returncode = await process.wait()

            if returncode == 0:
                job_info["status"] = "completed"
                logger.info(f"Job {job_id} completed successfully")
            else:
                job_info["status"] = "failed"
                job_info["error"] = "\n".join(error_lines) if error_lines else f"Process exited with code {returncode}"
                logger.error(f"Job {job_id} failed with return code {returncode}")

        except Exception as e:
            job_info["status"] = "failed"
            job_info["error"] = str(e)
            logger.error(f"Job {job_id} monitoring error: {e}", exc_info=True)

        finally:
            job_info["end_time"] = datetime.now().isoformat()
            script_path = job_info.get("script_path")
            if script_path and os.path.exists(script_path):
                try:
                    os.unlink(script_path)
                except Exception as e:
                    logger.warning(f"Failed to delete script {script_path}: {e}")

    async def stop_job(self, job_id: str) -> bool:
        """Stop a running job.

        Args:
            job_id: Job identifier

        Returns:
            bool: True if job was stopped, False if not found
        """
        job_info = self.jobs.get(job_id)
        if not job_info:
            return False

        if job_info["status"] == "running":
            process = job_info["process"]
            try:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()

                job_info["status"] = "stopped"
                job_info["end_time"] = datetime.now().isoformat()
                logger.info(f"Job {job_id} stopped")
            except Exception as e:
                logger.error(f"Error stopping job {job_id}: {e}", exc_info=True)
                return False

        return True

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status and output.

        Args:
            job_id: Job identifier

        Returns:
            Dict with job status information or None if not found
        """
        job_info = self.jobs.get(job_id)
        if not job_info:
            return None

        return {
            "job_id": job_info["job_id"],
            "job_name": job_info["job_name"],
            "status": job_info["status"],
            "output": "\n".join(job_info["output"]) if job_info["output"] else None,
            "error": job_info.get("error"),
            "start_time": job_info["start_time"],
            "end_time": job_info.get("end_time"),
        }

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs.

        Returns:
            List of job information dictionaries
        """
        return [
            {
                "job_id": job_info["job_id"],
                "job_name": job_info["job_name"],
                "status": job_info["status"],
                "start_time": job_info["start_time"],
                "end_time": job_info.get("end_time"),
            }
            for job_info in self.jobs.values()
        ]

    async def cleanup(self):
        """Clean up all running jobs."""
        logger.info("Cleaning up running jobs...")
        for job_id in list(self.jobs.keys()):
            await self.stop_job(job_id)
