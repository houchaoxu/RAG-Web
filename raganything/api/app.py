"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from raganything.api.routers import health, config, documents, query, batch, graph
from raganything.api.services.rag_service import rag_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("RAG-Anything API starting up")
    await rag_service.auto_initialize()
    yield
    logger.info("RAG-Anything API shutting down")
    await rag_service.shutdown()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="RAG-Anything API",
        description="All-in-One Multimodal Document Processing RAG API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health.router, prefix="/api")
    app.include_router(config.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(query.router, prefix="/api")
    app.include_router(batch.router, prefix="/api")
    app.include_router(graph.router, prefix="/api")

    # Serve frontend static files
    import os
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_frontend():
            return FileResponse(os.path.join(static_dir, "index.html"))

    return app


app = create_app()
