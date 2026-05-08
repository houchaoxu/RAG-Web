"""Health check endpoint."""

from fastapi import APIRouter
from raganything import __version__
from raganything.api.schemas import HealthResponse
from raganything.api.services.rag_service import rag_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    parser_installed = False
    if rag_service.is_initialized:
        try:
            parser_installed = rag_service.rag.check_parser_installation()
        except Exception:
            pass

    return HealthResponse(
        status="ok",
        version=__version__,
        parser_installed=parser_installed,
        lightrag_initialized=rag_service.is_initialized,
    )
