"""FastAPI server for Neural DSL backend bridge."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from neural.code_generation.code_generator import generate_code
from neural.parser.parser import ModelTransformer, create_parser
from neural.shape_propagation.shape_propagator import ShapePropagator

from .process_manager import ProcessManager
from .websocket_manager import ConnectionManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ParseRequest(BaseModel):
    dsl_code: str = Field(..., description="Neural DSL code to parse")
    parser_type: str = Field(default="network", description="Parser type: 'network' or 'research'")


class ParseResponse(BaseModel):
    success: bool
    model_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    warnings: Optional[List[str]] = None


class ShapePropagationRequest(BaseModel):
    model_data: Dict[str, Any] = Field(..., description="Parsed model data")
    framework: str = Field(default="tensorflow", description="Framework: 'tensorflow' or 'pytorch'")


class ShapePropagationResponse(BaseModel):
    success: bool
    shape_history: Optional[List[Dict[str, Any]]] = None
    trace: Optional[List[Dict[str, Any]]] = None
    issues: Optional[List[Dict[str, Any]]] = None
    optimizations: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class CodeGenerationRequest(BaseModel):
    model_data: Dict[str, Any] = Field(..., description="Parsed model data")
    backend: str = Field(default="tensorflow", description="Backend: 'tensorflow', 'pytorch', or 'onnx'")
    best_params: Optional[Dict[str, Any]] = Field(None, description="HPO best parameters")
    auto_flatten_output: bool = Field(False, description="Auto-flatten output before Dense/Output layers")


class CodeGenerationResponse(BaseModel):
    success: bool
    code: Optional[str] = None
    error: Optional[str] = None


class CompileRequest(BaseModel):
    dsl_code: str = Field(..., description="Neural DSL code to compile")
    backend: str = Field(default="tensorflow", description="Backend: 'tensorflow', 'pytorch', or 'onnx'")
    parser_type: str = Field(default="network", description="Parser type: 'network' or 'research'")
    auto_flatten_output: bool = Field(False, description="Auto-flatten output before Dense/Output layers")


class CompileResponse(BaseModel):
    success: bool
    code: Optional[str] = None
    model_data: Optional[Dict[str, Any]] = None
    shape_history: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class TrainingJobRequest(BaseModel):
    code: str = Field(..., description="Python code to execute")
    job_name: Optional[str] = Field(None, description="Job name for identification")
    env_vars: Optional[Dict[str, str]] = Field(None, description="Environment variables")


class TrainingJobResponse(BaseModel):
    success: bool
    job_id: str
    message: str
    error: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    output: Optional[str] = None
    error: Optional[str] = None


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Neural DSL Backend Bridge",
        description="Backend API for Neural DSL parsing, shape propagation, code generation, and training",
        version="0.3.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    process_manager = ProcessManager()
    websocket_manager = ConnectionManager()

    @app.on_event("startup")
    async def startup_event():
        logger.info("Neural DSL Backend Bridge starting up...")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Neural DSL Backend Bridge shutting down...")
        await process_manager.cleanup()

    @app.get("/")
    async def root():
        return {
            "service": "Neural DSL Backend Bridge",
            "version": "0.3.0",
            "status": "running",
        }

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    @app.post("/api/parse", response_model=ParseResponse)
    async def parse_dsl(request: ParseRequest):
        """Parse Neural DSL code and return model data."""
        try:
            parser = create_parser(request.parser_type)
            tree = parser.parse(request.dsl_code)
            transformer = ModelTransformer()
            model_data = transformer.transform(tree)

            return ParseResponse(
                success=True,
                model_data=model_data,
            )
        except Exception as e:
            logger.error(f"Parse error: {e}", exc_info=True)
            return ParseResponse(
                success=False,
                error=str(e),
            )

    @app.post("/api/shape-propagation", response_model=ShapePropagationResponse)
    async def propagate_shapes(request: ShapePropagationRequest):
        """Propagate shapes through the model and return analysis."""
        try:
            propagator = ShapePropagator(debug=False)
            model_data = request.model_data

            if "input" not in model_data or "shape" not in model_data["input"]:
                raise ValueError("model_data must contain 'input' with 'shape'")

            input_shape = (None,) + tuple(model_data["input"]["shape"])
            current_shape = input_shape

            for layer in model_data.get("layers", []):
                current_shape = propagator.propagate(current_shape, layer, request.framework)

            shape_history = [
                {"layer": layer, "output_shape": list(shape)}
                for layer, shape in propagator.shape_history
            ]

            trace = propagator.get_trace()
            issues = propagator.detect_issues()
            optimizations = propagator.suggest_optimizations()

            return ShapePropagationResponse(
                success=True,
                shape_history=shape_history,
                trace=trace,
                issues=issues,
                optimizations=optimizations,
            )
        except Exception as e:
            logger.error(f"Shape propagation error: {e}", exc_info=True)
            return ShapePropagationResponse(
                success=False,
                error=str(e),
            )

    @app.post("/api/generate-code", response_model=CodeGenerationResponse)
    async def generate_model_code(request: CodeGenerationRequest):
        """Generate backend-specific code from model data."""
        try:
            code = generate_code(
                model_data=request.model_data,
                backend=request.backend,
                best_params=request.best_params,
                auto_flatten_output=request.auto_flatten_output,
            )

            return CodeGenerationResponse(
                success=True,
                code=code,
            )
        except Exception as e:
            logger.error(f"Code generation error: {e}", exc_info=True)
            return CodeGenerationResponse(
                success=False,
                error=str(e),
            )

    @app.post("/api/compile", response_model=CompileResponse)
    async def compile_dsl(request: CompileRequest):
        """Complete pipeline: parse DSL, propagate shapes, and generate code."""
        try:
            parser = create_parser(request.parser_type)
            tree = parser.parse(request.dsl_code)
            transformer = ModelTransformer()
            model_data = transformer.transform(tree)

            propagator = ShapePropagator(debug=False)
            if "input" in model_data and "shape" in model_data["input"]:
                input_shape = (None,) + tuple(model_data["input"]["shape"])
                current_shape = input_shape

                for layer in model_data.get("layers", []):
                    current_shape = propagator.propagate(
                        current_shape, layer, request.backend
                    )

                shape_history = [
                    {"layer": layer, "output_shape": list(shape)}
                    for layer, shape in propagator.shape_history
                ]
            else:
                shape_history = None

            code = generate_code(
                model_data=model_data,
                backend=request.backend,
                auto_flatten_output=request.auto_flatten_output,
            )

            return CompileResponse(
                success=True,
                code=code,
                model_data=model_data,
                shape_history=shape_history,
            )
        except Exception as e:
            logger.error(f"Compilation error: {e}", exc_info=True)
            return CompileResponse(
                success=False,
                error=str(e),
            )

    @app.post("/api/jobs/start", response_model=TrainingJobResponse)
    async def start_training_job(request: TrainingJobRequest):
        """Start a training job in a separate process."""
        try:
            job_id = await process_manager.start_job(
                code=request.code,
                job_name=request.job_name,
                env_vars=request.env_vars,
            )

            return TrainingJobResponse(
                success=True,
                job_id=job_id,
                message=f"Training job {job_id} started successfully",
            )
        except Exception as e:
            logger.error(f"Failed to start training job: {e}", exc_info=True)
            return TrainingJobResponse(
                success=False,
                job_id="",
                message="",
                error=str(e),
            )

    @app.get("/api/jobs/{job_id}/status", response_model=JobStatusResponse)
    async def get_job_status(job_id: str):
        """Get the status and output of a training job."""
        try:
            status_info = process_manager.get_job_status(job_id)
            if status_info is None:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

            return JobStatusResponse(
                job_id=job_id,
                status=status_info["status"],
                output=status_info.get("output"),
                error=status_info.get("error"),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get job status: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/jobs/{job_id}/stop")
    async def stop_training_job(job_id: str):
        """Stop a running training job."""
        try:
            success = await process_manager.stop_job(job_id)
            if not success:
                raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

            return {"success": True, "message": f"Job {job_id} stopped"}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to stop job: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/jobs")
    async def list_jobs():
        """List all training jobs."""
        try:
            jobs = process_manager.list_jobs()
            return {"jobs": jobs}
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        await websocket_manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                await websocket_manager.broadcast(f"Echo: {data}")
        except WebSocketDisconnect:
            websocket_manager.disconnect(websocket)
            logger.info("WebSocket client disconnected")

    @app.websocket("/ws/jobs/{job_id}")
    async def job_websocket(websocket: WebSocket, job_id: str):
        """WebSocket endpoint for real-time job updates."""
        await websocket_manager.connect(websocket)
        try:
            while True:
                status_info = process_manager.get_job_status(job_id)
                if status_info:
                    await websocket.send_json(status_info)

                if status_info and status_info["status"] in ["completed", "failed"]:
                    break

                await asyncio.sleep(1)
        except WebSocketDisconnect:
            websocket_manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)
            await websocket.close()

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
