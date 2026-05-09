# RAG-Web

A RESTful API and Web UI for [RAG-Anything](https://github.com/HKUDS/RAG-Anything), turning the original Python library into a fully accessible web service with multimodal document processing, knowledge graph visualization, and interactive querying.

## Features

- **RESTful API** — FastAPI-based service with full CRUD endpoints for documents, queries, config, and knowledge graph
- **Web Frontend** — Single-page app for uploading documents, querying the knowledge base, and exploring the graph
- **Multi-Provider LLM** — Supports OpenAI, Ollama, Azure OpenAI, LM Studio, vLLM
- **Local Embedding** — Run embedding models locally via sentence-transformers (BGE, E5, GTE, etc.) without external API calls
- **Hot-Reloadable Config** — Update LLM, embedding, and RAG settings at runtime via API without restarting
- **Async Document Processing** — Background task processing with real-time stage tracking (parse → text extraction → multimodal → done)
- **Knowledge Graph Explorer** — Browse entities, relationships, and visualize the graph as an interactive network
- **Multimodal Query** — Query with text, images, tables, and equations in a single request

## Screenshots

| Documents | Upload |
|-----------|--------|
| ![Documents](web-ui/Documents.jpg) | ![Upload](web-ui/upload.jpg) |

| Query | Knowledge Graph |
|-------|-----------------|
| ![Query](web-ui/Query.jpg) | ![Knowledge Graph](web-ui/KnowledgeGraph.jpg) |

| Configuration | Status |
|---------------|--------|
| ![Configuration](web-ui/Configuration.jpg) | ![Status](web-ui/Status.jpg) |

## Quick Start

### Install

```bash
pip install -e .
pip install fastapi uvicorn[standard] python-multipart
```

### Run the Server

```bash
python -m raganything.api.server --port 8000
```

Options:
- `--host` — Bind address (default: `0.0.0.0`)
- `--port` — Port (default: `8000`)
- `--reload` — Auto-reload on code changes (development mode)

Open http://localhost:8000 for the web UI, or http://localhost:8000/docs for the interactive API documentation (Swagger).

## API Endpoints

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Service health status, version, parser installation check |

### Configuration

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/config/llm` | Get current LLM configuration |
| PUT | `/api/config/llm` | Update LLM config (provider, model, api_key, temperature, etc.) |
| GET | `/api/config/embedding` | Get current embedding configuration |
| PUT | `/api/config/embedding` | Update embedding config (provider, model, dimension, etc.) |
| GET | `/api/config/rag` | Get RAG settings (working_dir, parser, context mode) |
| PUT | `/api/config/rag` | Update RAG settings |
| GET | `/api/config/full` | Get full configuration |
| PUT | `/api/config/full` | Update full configuration in one request |
| POST | `/api/config/test-llm` | Test LLM connectivity with a simple prompt |

### Documents

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/documents/upload` | Upload a file (returns task_id) |
| POST | `/api/documents/process` | Process a document by file path (async, returns task_id) |
| GET | `/api/documents/tasks/{task_id}` | Poll task status and progress stage |

### Batch Processing

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/batch/process` | Process multiple documents by file paths |
| POST | `/api/batch/process-folder` | Process all supported files in a folder |

### Query

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/query` | Text query against the knowledge base |
| POST | `/api/query/multimodal` | Query with multimodal content (images, tables, equations) |
| POST | `/api/query/stream` | Streaming query response (SSE) |

Query modes: `local`, `global`, `hybrid`, `naive`, `mix`, `bypass`

### Knowledge Graph

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/graph/stats` | Entity/relation counts and type breakdown |
| GET | `/api/graph/entities` | List entities with search and pagination |
| GET | `/api/graph/relations` | List relationships with search and pagination |
| GET | `/api/graph/network` | Full graph data in vis-network format for visualization |

## Configuration Examples

### LLM (OpenAI)

```json
PUT /api/config/llm
{
  "provider": "openai",
  "model": "gpt-4o",
  "api_key": "sk-...",
  "temperature": 0.7
}
```

### LLM (Ollama local)

```json
PUT /api/config/llm
{
  "provider": "ollama",
  "model": "qwen2.5:7b",
  "base_url": "http://localhost:11434"
}
```

### Embedding (local sentence-transformers)

```json
PUT /api/config/embedding
{
  "provider": "local",
  "model": "BAAI/bge-large-zh-v1.5",
  "dimension": 1024,
  "batch_size": 64
}
```

### Embedding (OpenAI)

```json
PUT /api/config/embedding
{
  "provider": "openai",
  "model": "text-embedding-3-large",
  "api_key": "sk-...",
  "dimension": 3072
}
```

## Project Structure

```
raganything/api/
├── app.py                  # FastAPI application factory
├── server.py               # CLI entry point (uvicorn)
├── schemas.py              # Pydantic request/response models
├── routers/
│   ├── health.py           # Health check endpoint
│   ├── config.py           # Configuration CRUD endpoints
│   ├── documents.py        # Document upload & processing
│   ├── query.py            # Text & multimodal query
│   ├── batch.py            # Batch document processing
│   └── graph.py            # Knowledge graph endpoints
├── services/
│   ├── rag_service.py      # Singleton RAG lifecycle manager
│   └── llm_factory.py      # Multi-provider LLM/embedding factory
└── static/
    └── index.html          # Web frontend (single-page app)
```

## License

This project is based on [RAG-Anything](https://github.com/HKUDS/RAG-Anything) by HKUDS.
