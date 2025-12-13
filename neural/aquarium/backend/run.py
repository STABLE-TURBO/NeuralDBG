"""Run the backend bridge server."""

import uvicorn

from .config import settings

if __name__ == "__main__":
    uvicorn.run(
        "neural.aquarium.backend.server:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
