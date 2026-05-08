"""Pydantic request/response models for RAG-Anything API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ============ Document Schemas ============

class DocumentProcessRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path to the document file")
    output_dir: Optional[str] = Field(None, description="Output directory for parsed files")
    parse_method: Optional[str] = Field(None, description="Parse method: auto, ocr, txt")
    display_stats: Optional[bool] = Field(None, description="Display content statistics")


class DocumentParseRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path to the document file")
    output_dir: Optional[str] = Field(None, description="Output directory for parsed files")
    parse_method: Optional[str] = Field(None, description="Parse method: auto, ocr, txt")


class ContentItem(BaseModel):
    type: str = Field(..., description="Content type: text, image, table, equation")
    text: Optional[str] = Field(None, description="Text content")
    img_path: Optional[str] = Field(None, description="Absolute path to image file")
    image_caption: Optional[List[str]] = Field(None, description="Image captions")
    table_body: Optional[str] = Field(None, description="Table content in markdown")
    table_caption: Optional[List[str]] = Field(None, description="Table captions")
    latex: Optional[str] = Field(None, description="LaTeX equation")
    page_idx: Optional[int] = Field(None, description="Page index (0-based)")


class DocumentInsertRequest(BaseModel):
    content_list: List[ContentItem] = Field(..., description="Pre-parsed content items")
    file_path: Optional[str] = Field("unknown_document", description="Reference file path")
    doc_id: Optional[str] = Field(None, description="Document ID (auto-generated if not set)")


# ============ Batch Schemas ============

class BatchProcessRequest(BaseModel):
    file_paths: List[str] = Field(..., description="List of file paths or directories")
    output_dir: Optional[str] = Field(None, description="Output directory")
    parse_method: Optional[str] = Field(None, description="Parse method")
    max_workers: Optional[int] = Field(None, description="Max concurrent workers")
    recursive: Optional[bool] = Field(None, description="Process directories recursively")


class BatchFolderRequest(BaseModel):
    folder_path: str = Field(..., description="Path to folder to process")
    output_dir: Optional[str] = Field(None, description="Output directory")
    parse_method: Optional[str] = Field(None, description="Parse method")
    max_workers: Optional[int] = Field(None, description="Max concurrent workers")
    recursive: Optional[bool] = Field(None, description="Process subdirectories")
    file_extensions: Optional[List[str]] = Field(None, description="File extensions to include")


# ============ Query Schemas ============

class QueryRequest(BaseModel):
    query: str = Field(..., description="Query text")
    mode: str = Field("mix", description="Query mode: local, global, hybrid, naive, mix, bypass")
    vlm_enhanced: Optional[bool] = Field(None, description="Enable VLM enhanced query")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")


class MultimodalContentItem(BaseModel):
    type: str = Field(..., description="Content type: image, table, equation")
    img_path: Optional[str] = Field(None, description="Image file path")
    table_data: Optional[str] = Field(None, description="Table data in CSV/markdown")
    table_caption: Optional[str] = Field(None, description="Table caption")
    latex: Optional[str] = Field(None, description="LaTeX formula")
    equation_caption: Optional[str] = Field(None, description="Equation caption")


class MultimodalQueryRequest(BaseModel):
    query: str = Field(..., description="Query text")
    multimodal_content: List[MultimodalContentItem] = Field(
        default_factory=list, description="Multimodal content items"
    )
    mode: str = Field("mix", description="Query mode")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")


# ============ Config Schemas ============

class LLMProviderConfig(BaseModel):
    provider: str = Field("openai", description="LLM provider: openai, ollama, azure_openai, lmstudio, vllm")
    model: str = Field("gpt-4o-mini", description="LLM model name")
    vision_model: Optional[str] = Field(None, description="Vision model name")
    api_key: Optional[str] = Field(None, description="API key (not needed for local providers)")
    base_url: Optional[str] = Field(None, description="API base URL")
    temperature: float = Field(0.7, description="Generation temperature")
    max_tokens: Optional[int] = Field(None, description="Max output tokens")
    timeout: Optional[int] = Field(None, description="Request timeout in seconds")


class EmbeddingConfig(BaseModel):
    provider: str = Field("openai", description="Embedding provider: openai, ollama, azure_openai, local")
    model: str = Field("text-embedding-3-large", description="Embedding model name or HuggingFace model ID")
    model_path: Optional[str] = Field(None, description="Local model path (for provider=local). If set, loads from this path instead of downloading")
    api_key: Optional[str] = Field(None, description="API key (not needed for local)")
    base_url: Optional[str] = Field(None, description="API base URL (not needed for local)")
    dimension: int = Field(3072, description="Embedding dimension")
    batch_size: int = Field(100, description="Embedding batch size")


class RAGConfig(BaseModel):
    working_dir: Optional[str] = Field(None, description="RAG storage directory")
    parser: Optional[str] = Field(None, description="Document parser: mineru, docling, paddleocr")
    parse_method: Optional[str] = Field(None, description="Parse method: auto, ocr, txt")
    mineru_backend: Optional[str] = Field(None, description="MinerU backend: pipeline (fast), hybrid-auto-engine (accurate), vlm-auto-engine")
    enable_image_processing: Optional[bool] = Field(None, description="Enable image content processing")
    enable_table_processing: Optional[bool] = Field(None, description="Enable table content processing")
    enable_equation_processing: Optional[bool] = Field(None, description="Enable equation content processing")
    max_concurrent_files: Optional[int] = Field(None, description="Maximum concurrent file processing workers")
    context_window: Optional[int] = Field(None, description="Number of surrounding content items for context")
    context_mode: Optional[str] = Field(None, description="Context extraction mode: page, chunk, token")
    max_context_tokens: Optional[int] = Field(None, description="Maximum tokens for context extraction")


class FullConfig(BaseModel):
    llm: LLMProviderConfig = Field(default_factory=LLMProviderConfig, description="LLM provider configuration")
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig, description="Embedding model configuration")
    rag: RAGConfig = Field(default_factory=RAGConfig, description="RAG processing configuration")


class LLMTestRequest(BaseModel):
    provider: str = Field("openai", description="LLM provider")
    model: str = Field(..., description="Model name")
    api_key: Optional[str] = Field(None, description="API key")
    base_url: Optional[str] = Field(None, description="API base URL")


# ============ Response Schemas ============

class TaskResponse(BaseModel):
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Status message")


class DocumentStatusResponse(BaseModel):
    doc_id: str = Field(..., description="Document identifier")
    fully_processed: bool = Field(..., description="Whether document is fully processed")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed processing status")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Query answer")
    mode: str = Field(..., description="Query mode used")


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Service status")
    version: str = Field("", description="RAG-Anything version")
    parser_installed: bool = Field(False, description="Whether document parser is installed")
    lightrag_initialized: bool = Field(False, description="Whether LightRAG is initialized")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error detail message")
