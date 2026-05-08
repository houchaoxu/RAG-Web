"""Batch processing endpoints."""

from fastapi import APIRouter, HTTPException
from raganything.api.schemas import BatchProcessRequest, BatchFolderRequest
from raganything.api.services.rag_service import rag_service

router = APIRouter(prefix="/batch", tags=["batch"])


@router.post("/process")
async def batch_process(req: BatchProcessRequest):
    """Batch process multiple documents."""
    try:
        rag = rag_service.rag
        kwargs = {}
        if req.output_dir:
            kwargs["output_dir"] = req.output_dir
        if req.parse_method:
            kwargs["parse_method"] = req.parse_method
        if req.max_workers:
            kwargs["max_workers"] = req.max_workers
        if req.recursive is not None:
            kwargs["recursive"] = req.recursive

        result = await rag.process_documents_batch_async(req.file_paths, **kwargs)
        return {
            "total": result.total_files,
            "successful": result.successful_files,
            "failed": result.failed_files,
            "skipped": result.skipped_files,
            "errors": [{"file": e.file_path, "error": e.error_message} for e in result.errors],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-folder")
async def batch_process_folder(req: BatchFolderRequest):
    """Process all supported files in a folder."""
    try:
        rag = rag_service.rag
        kwargs = {}
        if req.output_dir:
            kwargs["output_dir"] = req.output_dir
        if req.parse_method:
            kwargs["parse_method"] = req.parse_method
        if req.max_workers:
            kwargs["max_workers"] = req.max_workers
        if req.recursive is not None:
            kwargs["recursive"] = req.recursive
        if req.file_extensions:
            kwargs["file_extensions"] = req.file_extensions

        await rag.process_folder_complete(req.folder_path, **kwargs)
        return {"message": f"Folder processed: {req.folder_path}"}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Folder not found: {req.folder_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
