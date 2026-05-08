"""Query endpoints."""

import json
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from lightrag.base import QueryParam
from raganything.api.schemas import QueryRequest, MultimodalQueryRequest, QueryResponse
from raganything.api.services.rag_service import rag_service

router = APIRouter(prefix="/query", tags=["query"])


async def _get_rag():
    rag = rag_service.rag
    if rag.lightrag is None:
        await rag._ensure_lightrag_initialized()
    if rag.lightrag is None:
        raise HTTPException(status_code=503, detail="RAG not initialized. Configure LLM first.")
    return rag


@router.post("", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Execute a text query against the RAG knowledge base."""
    try:
        rag = await _get_rag()
        kwargs = {}
        if req.vlm_enhanced is not None:
            kwargs["vlm_enhanced"] = req.vlm_enhanced

        result = await rag.aquery(
            req.query, mode=req.mode, system_prompt=req.system_prompt, **kwargs
        )
        return QueryResponse(answer=result, mode=req.mode)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _extract_references(raw_data: dict, top_n: int = 3) -> list:
    """Extract top-N entity/relationship references from raw query data.

    Entities are pre-sorted by cosine similarity, relations by rank+weight.
    """
    data = raw_data.get("data", {})
    refs = []
    for entity in data.get("entities", [])[:top_n]:
        refs.append({
            "type": "entity",
            "name": entity.get("entity_name", entity.get("id", "")),
            "description": (entity.get("description") or "").split("<SEP>")[0][:120],
        })
    for rel in data.get("relationships", [])[:top_n]:
        refs.append({
            "type": "relation",
            "source": rel.get("src_id", rel.get("source", "")),
            "target": rel.get("tgt_id", rel.get("target", "")),
            "description": (rel.get("description") or "").split("<SEP>")[0][:120],
        })
    return refs


@router.post("/stream")
async def query_stream(req: QueryRequest):
    """Execute a text query with SSE streaming.

    Event types:
    - {"type": "token", "content": "...", "done": false}  — LLM token
    - {"type": "references", "references": [...], "done": false}  — retrieved entities/relations
    - {"type": "done", "content": "", "done": true}  — stream finished
    """
    try:
        rag = await _get_rag()

        # Use lightrag.aquery_llm directly to get structured data + streaming
        query_param = QueryParam(mode=req.mode, stream=True)
        llm_kwargs = {}
        if req.vlm_enhanced is not None:
            llm_kwargs["vlm_enhanced"] = req.vlm_enhanced

        # For VLM enhanced, fall back to rag.aquery
        use_vlm = req.vlm_enhanced and hasattr(rag, "vision_model_func") and rag.vision_model_func
        if use_vlm:
            result = await rag.aquery(
                req.query, mode=req.mode, system_prompt=req.system_prompt, **llm_kwargs
            )
            if isinstance(result, str):
                async def single():
                    yield f"data: {json.dumps({'type': 'token', 'content': result, 'done': False})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'content': '', 'done': True})}\n\n"
                return StreamingResponse(single(), media_type="text/event-stream")
            async def vlm_stream():
                async for chunk in result:
                    if chunk:
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk, 'done': False})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'content': '', 'done': True})}\n\n"
            return StreamingResponse(vlm_stream(), media_type="text/event-stream")

        # Normal path: use aquery_llm to get references + streaming
        full_result = await rag.lightrag.aquery_llm(
            req.query, param=query_param, system_prompt=req.system_prompt
        )

        llm_response = full_result.get("llm_response", {})
        references = _extract_references(full_result)

        # Non-streaming fallback
        if not llm_response.get("is_streaming"):
            content = llm_response.get("content", "")
            async def non_stream():
                yield f"data: {json.dumps({'type': 'token', 'content': content, 'done': False})}\n\n"
                yield f"data: {json.dumps({'type': 'references', 'references': references, 'done': False})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'content': '', 'done': True})}\n\n"
            return StreamingResponse(non_stream(), media_type="text/event-stream")

        # Streaming path
        iterator = llm_response.get("response_iterator")

        async def event_stream():
            # Stream LLM tokens
            if iterator:
                async for chunk in iterator:
                    if chunk:
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk, 'done': False})}\n\n"
            # Send references after content is done
            yield f"data: {json.dumps({'type': 'references', 'references': references, 'done': False})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'content': '', 'done': True})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/multimodal", response_model=QueryResponse)
async def multimodal_query(req: MultimodalQueryRequest):
    """Execute a multimodal query with images, tables, or equations."""
    try:
        rag = rag_service.rag
        mc = [item.model_dump(exclude_none=True) for item in req.multimodal_content]
        result = await rag.aquery_with_multimodal(
            req.query, multimodal_content=mc, mode=req.mode, system_prompt=req.system_prompt
        )
        return QueryResponse(answer=result, mode=req.mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
