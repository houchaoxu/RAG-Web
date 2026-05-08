"""Document processing endpoints."""

import os
import uuid
import logging
import tempfile
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Query
from raganything.api.schemas import (
    DocumentProcessRequest,
    DocumentParseRequest,
    DocumentInsertRequest,
    DocumentStatusResponse,
    ContentItem,
)
from raganything.api.services.rag_service import rag_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# In-memory task tracking
_tasks: Dict[str, Dict] = {}


_STAGES = {
    "init": "Initializing",
    "parse": "Parsing document",
    "text_insert": "Extracting & indexing text",
    "multimodal": "Processing images/tables",
    "done": "Completed",
}


async def _run_process_task(task_id: str, file_path: str, kwargs: dict, cleanup_path: str = None):
    """Background task: process a document with stage tracking."""
    def _on_stage(stage: str):
        _tasks[task_id]["stage"] = stage
        _tasks[task_id]["stage_label"] = _STAGES.get(stage, stage)

    try:
        _tasks[task_id]["status"] = "processing"
        _on_stage("init")

        rag = rag_service.rag
        _on_stage("parse")
        await rag.process_document_complete(file_path, stage_callback=_on_stage, **kwargs)
        _on_stage("done")
        _tasks[task_id]["status"] = "completed"
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)
    finally:
        if cleanup_path:
            p = Path(cleanup_path)
            if p.exists():
                p.unlink()


@router.post("/process")
async def process_document(req: DocumentProcessRequest, background_tasks: BackgroundTasks):
    """Process a document: parse + multimodal handling + insert into RAG.

    Returns immediately with a task_id. Poll GET /documents/tasks/{task_id} for status.
    """
    try:
        _ = rag_service.rag  # verify initialized
    except RuntimeError:
        raise HTTPException(status_code=503, detail="RAG not initialized. Configure LLM first.")

    if not Path(req.file_path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    task_id = uuid.uuid4().hex[:12]
    _tasks[task_id] = {"status": "pending", "file": req.file_path}

    kwargs = {}
    if req.output_dir:
        kwargs["output_dir"] = req.output_dir
    if req.parse_method:
        kwargs["parse_method"] = req.parse_method
    if req.display_stats is not None:
        kwargs["display_stats"] = req.display_stats

    background_tasks.add_task(_run_process_task, task_id, req.file_path, kwargs)
    return {"task_id": task_id, "status": "pending", "message": "Processing started"}


@router.post("/upload")
async def upload_and_process(
    file: UploadFile = File(...),
    parse_method: str = Form(""),
    background_tasks: BackgroundTasks = None,
):
    """Upload a file and process it through RAG.

    Returns immediately with a task_id. Poll GET /documents/tasks/{task_id} for status.
    """
    try:
        _ = rag_service.rag
    except RuntimeError:
        raise HTTPException(status_code=503, detail="RAG not initialized. Configure LLM first.")

    upload_dir = Path("./uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename

    contents = await file.read()
    dest.write_bytes(contents)

    task_id = uuid.uuid4().hex[:12]
    _tasks[task_id] = {"status": "pending", "file": file.filename}

    kwargs = {}
    if parse_method:
        kwargs["parse_method"] = parse_method

    background_tasks.add_task(_run_process_task, task_id, str(dest), kwargs, cleanup_path=str(dest))
    return {"task_id": task_id, "status": "pending", "message": f"Upload received, processing: {file.filename}"}


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a background processing task."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return _tasks[task_id]


@router.get("/list")
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """List all documents with their processing status."""
    try:
        rag = rag_service.rag
        if rag.lightrag is None:
            return {"documents": [], "total": 0, "page": page, "page_size": page_size}
        docs, total = await rag.lightrag.doc_status.get_docs_paginated(
            page=page, page_size=page_size, sort_field="updated_at", sort_direction="DESC"
        )
        documents = []
        for doc_id, raw_status in docs:
            status_str = getattr(raw_status, "status", "")
            if hasattr(status_str, "value"):
                status_str = status_str.value
            text_ok = status_str == "processed"
            mm_ok = getattr(raw_status, "multimodal_processed", False) or False
            documents.append({
                "doc_id": doc_id,
                "status": status_str,
                "text_processed": text_ok,
                "multimodal_processed": mm_ok,
                "fully_processed": text_ok and mm_ok,
                "chunks_count": getattr(raw_status, "chunks_count", 0) or 0,
                "updated_at": getattr(raw_status, "updated_at", ""),
                "file_path": getattr(raw_status, "file_path", ""),
            })
        return {"documents": documents, "total": total, "page": page, "page_size": page_size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/parse")
async def parse_document(req: DocumentParseRequest):
    """Parse a document and return its content list without inserting into RAG."""
    try:
        rag = rag_service.rag
        kwargs = {}
        if req.output_dir:
            kwargs["output_dir"] = req.output_dir
        if req.parse_method:
            kwargs["parse_method"] = req.parse_method

        content_list, doc_id = await rag.parse_document(req.file_path, **kwargs)
        return {
            "doc_id": doc_id,
            "content_count": len(content_list),
            "content_list": content_list,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/insert")
async def insert_content(req: DocumentInsertRequest):
    """Insert pre-parsed content list directly into RAG."""
    try:
        rag = rag_service.rag
        content_dicts = [item.model_dump(exclude_none=True) for item in req.content_list]
        kwargs = {}
        if req.file_path:
            kwargs["file_path"] = req.file_path
        if req.doc_id:
            kwargs["doc_id"] = req.doc_id

        await rag.insert_content_list(content_dicts, **kwargs)
        return {"message": f"Inserted {len(content_dicts)} content items"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{doc_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(doc_id: str):
    """Get processing status of a document."""
    try:
        rag = rag_service.rag
        fully_processed = await rag.is_document_fully_processed(doc_id)
        details = await rag.get_document_processing_status(doc_id)
        return DocumentStatusResponse(
            doc_id=doc_id,
            fully_processed=fully_processed,
            details=details,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
