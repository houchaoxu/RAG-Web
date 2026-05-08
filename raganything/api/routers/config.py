"""Configuration management endpoints."""

from fastapi import APIRouter, HTTPException
from raganything.api.schemas import (
    LLMProviderConfig,
    EmbeddingConfig,
    RAGConfig,
    FullConfig,
    LLMTestRequest,
)
from raganything.api.services.rag_service import rag_service
from raganything.api.services.llm_factory import test_llm_connection, PROVIDER_MAP

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/llm", response_model=LLMProviderConfig)
async def get_llm_config():
    """Get current LLM configuration."""
    config = rag_service.get_llm_config()
    if config is None:
        return LLMProviderConfig()
    return config


@router.put("/llm")
async def update_llm_config(llm: LLMProviderConfig):
    """Update LLM configuration. Rebuilds model functions with new settings."""
    try:
        await rag_service.update_llm_config(llm)
        return {"message": "LLM configuration updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/embedding", response_model=EmbeddingConfig)
async def get_embedding_config():
    """Get current embedding configuration."""
    config = rag_service.get_embedding_config()
    if config is None:
        return EmbeddingConfig()
    return config


@router.put("/embedding")
async def update_embedding_config(embedding: EmbeddingConfig):
    """Update embedding configuration."""
    try:
        llm = rag_service.get_llm_config()
        if llm is None:
            raise HTTPException(
                status_code=400,
                detail="LLM config must be set before updating embedding config",
            )
        await rag_service.update_llm_config(llm, embedding)
        return {"message": "Embedding configuration updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/rag", response_model=RAGConfig)
async def get_rag_config():
    """Get current RAG processing configuration."""
    config = rag_service.get_rag_config()
    if config is None:
        return RAGConfig()
    return config


@router.put("/rag")
async def update_rag_config(rag: RAGConfig):
    """Update RAG processing configuration."""
    try:
        await rag_service.update_rag_config(rag)
        return {"message": "RAG configuration updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/full", response_model=FullConfig)
async def get_full_config():
    """Get complete configuration (LLM + Embedding + RAG)."""
    return FullConfig(
        llm=rag_service.get_llm_config() or LLMProviderConfig(),
        embedding=rag_service.get_embedding_config() or EmbeddingConfig(),
        rag=rag_service.get_rag_config() or RAGConfig(),
    )


@router.put("/full")
async def update_full_config(config: FullConfig):
    """Update all configuration at once."""
    try:
        await rag_service.initialize(config.llm, config.embedding, config.rag)
        return {"message": "Full configuration updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/llm/providers")
async def list_providers():
    """List supported LLM providers."""
    return {"providers": list(PROVIDER_MAP.keys())}


@router.post("/llm/test")
async def test_llm(req: LLMTestRequest):
    """Test LLM connection with a simple prompt."""
    config = LLMProviderConfig(
        provider=req.provider, model=req.model, api_key=req.api_key, base_url=req.base_url
    )
    result = await test_llm_connection(config)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result
