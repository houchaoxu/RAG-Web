"""Entry point for running the RAG-Anything API server.

Usage:
    python -m raganything.api.server [--host 0.0.0.0] [--port 8000] [--reload]
"""

import argparse
import logging
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="RAG-Anything API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    uvicorn.run(
        "raganything.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
