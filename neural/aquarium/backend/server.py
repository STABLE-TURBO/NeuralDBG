"""FastAPI server for Neural DSL backend bridge."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field

from neural.code_generation.code_generator import generate_code
from neural.parser.parser import ModelTransformer, create_parser
from neural.shape_propagation.shape_propagator import ShapePropagator

from .process_manager import ProcessManager
from .websocket_manager import ConnectionManager
from .terminal_handler import TerminalManager

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
    terminal_manager = TerminalManager()

    @app.on_event("startup")
    async def startup_event():
        logger.info("Neural DSL Backend Bridge starting up...")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Neural DSL Backend Bridge shutting down...")
        await process_manager.cleanup()
        terminal_manager.cleanup_all()

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

    @app.websocket("/terminal/{session_id}")
    async def terminal_websocket(websocket: WebSocket, session_id: str):
        """WebSocket endpoint for terminal sessions."""
        await websocket.accept()
        
        try:
            session = terminal_manager.get_session(session_id)
            if not session:
                session = terminal_manager.create_session(session_id, shell="bash")
            
            logger.info(f"Terminal WebSocket connected for session: {session_id}")
            
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    if message["type"] == "command":
                        command = message["data"]
                        output = await session.execute_command(command)
                        
                        await websocket.send_text(json.dumps({
                            "type": "output",
                            "data": output
                        }))
                        
                        await websocket.send_text(json.dumps({
                            "type": "prompt",
                            "data": "$ "
                        }))
                    
                    elif message["type"] == "autocomplete":
                        partial = message["data"]
                        suggestions = session.get_autocomplete_suggestions(partial)
                        
                        await websocket.send_text(json.dumps({
                            "type": "autocomplete",
                            "suggestions": suggestions
                        }))
                    
                    elif message["type"] == "change_shell":
                        new_shell = message["shell"]
                        success = session.change_shell(new_shell)
                        
                        if success:
                            await websocket.send_text(json.dumps({
                                "type": "shell_change",
                                "shell": new_shell
                            }))
                        else:
                            await websocket.send_text(json.dumps({
                                "type": "output",
                                "data": f"\x1b[1;31mFailed to change shell to {new_shell}\x1b[0m\r\n"
                            }))
                
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Terminal command error: {e}", exc_info=True)
                    await websocket.send_text(json.dumps({
                        "type": "output",
                        "data": f"\x1b[1;31mError: {str(e)}\x1b[0m\r\n"
                    }))
        
        except Exception as e:
            logger.error(f"Terminal WebSocket error: {e}", exc_info=True)
        finally:
            logger.info(f"Terminal WebSocket disconnected for session: {session_id}")

    @app.get("/api/examples/list")
    async def list_examples():
        """List available example models."""
        try:
            examples_dir = Path(__file__).parent.parent / 'examples'
            examples = []
            
            if not examples_dir.exists():
                return JSONResponse(content={'examples': [], 'count': 0})
            
            for example_file in examples_dir.glob('*.neural'):
                with open(example_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                name = example_file.stem.replace('_', ' ').title()
                category = 'General'
                tags = []
                complexity = 'Intermediate'
                
                if 'cnn' in example_file.stem.lower() or 'conv' in content.lower():
                    category = 'Computer Vision'
                    tags.extend(['cnn', 'computer-vision'])
                elif 'lstm' in content.lower() or 'rnn' in content.lower():
                    category = 'NLP'
                    tags.extend(['nlp', 'recurrent'])
                elif 'gan' in example_file.stem.lower() or 'vae' in example_file.stem.lower():
                    category = 'Generative'
                    tags.extend(['generative'])
                
                if 'mnist' in example_file.stem.lower():
                    description = 'Convolutional Neural Network for MNIST digit classification'
                    tags.append('mnist')
                    complexity = 'Beginner'
                elif 'text' in example_file.stem.lower():
                    description = 'LSTM network for text classification and sentiment analysis'
                    tags.append('text')
                    complexity = 'Beginner'
                else:
                    description = f'Neural network model: {name}'
                
                examples.append({
                    'name': name,
                    'path': str(example_file.relative_to(examples_dir.parent)),
                    'description': description,
                    'category': category,
                    'tags': tags,
                    'complexity': complexity
                })
            
            return JSONResponse(content={'examples': examples, 'count': len(examples)})
        except Exception as e:
            logger.error(f"Failed to list examples: {e}", exc_info=True)
            return JSONResponse(content={'error': str(e), 'examples': []}, status_code=500)

    @app.get("/api/examples/load")
    async def load_example(path: str):
        """Load an example model file."""
        try:
            full_path = Path(__file__).parent.parent / path
            
            if not full_path.exists():
                raise HTTPException(status_code=404, detail='Example file not found')
            
            if not full_path.is_file() or not str(full_path).endswith('.neural'):
                raise HTTPException(status_code=400, detail='Invalid example file')
            
            with open(full_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            return JSONResponse(content={
                'code': code,
                'path': path,
                'name': full_path.stem
            })
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to load example: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/docs/{doc_path:path}")
    async def get_documentation(doc_path: str):
        """Serve documentation files."""
        try:
            docs_dir = Path(__file__).parent.parent
            search_paths = [
                docs_dir / doc_path,
                docs_dir / doc_path.upper(),
                docs_dir.parent.parent / 'docs' / doc_path,
                docs_dir.parent.parent / 'docs' / doc_path.upper(),
            ]
            
            file_path = None
            for path in search_paths:
                if path.exists() and path.is_file():
                    file_path = path
                    break
            
            if not file_path:
                raise HTTPException(status_code=404, detail='Documentation file not found')
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return PlainTextResponse(content=content)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to load documentation: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
