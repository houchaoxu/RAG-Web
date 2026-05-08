"""Singleton service for managing the RAGAnything instance.

Provides lazy initialization, config-driven reinitialization, and cleanup.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from raganything import RAGAnything, RAGAnythingConfig
from raganything.api.schemas import LLMProviderConfig, EmbeddingConfig, RAGConfig
from raganything.api.services.llm_factory import (
    create_llm_func,
    create_vision_func,
    create_embedding_func,
)

logger = logging.getLogger(__name__)


class RAGService:
    """Manages a singleton RAGAnything instance with runtime reconfiguration."""

    CONFIG_PATH = Path("./rag_storage/api_config.json")

    def __init__(self):
        self._rag: Optional[RAGAnything] = None
        self._llm_config: Optional[LLMProviderConfig] = None
        self._embedding_config: Optional[EmbeddingConfig] = None
        self._rag_config: Optional[RAGConfig] = None
        self._initializing = False
        self._load_config()

    async def auto_initialize(self):
        """Re-initialize RAGAnything from persisted configs on startup."""
        if self._llm_config and self._rag is None:
            emb = self._embedding_config or EmbeddingConfig()
            await self.initialize(self._llm_config, emb, self._rag_config)

    @property
    def is_initialized(self) -> bool:
        return self._rag is not None

    @property
    def rag(self) -> RAGAnything:
        if self._rag is None:
            raise RuntimeError(
                "RAGAnything not initialized. Call initialize() first or configure LLM settings."
            )
        return self._rag

    def get_llm_config(self) -> Optional[LLMProviderConfig]:
        return self._llm_config

    def get_embedding_config(self) -> Optional[EmbeddingConfig]:
        return self._embedding_config

    def get_rag_config(self) -> Optional[RAGConfig]:
        return self._rag_config

    def _save_config(self):
        """Persist current configs to disk."""
        try:
            data = {}
            if self._llm_config:
                data["llm"] = self._llm_config.model_dump()
            if self._embedding_config:
                data["embedding"] = self._embedding_config.model_dump()
            if self._rag_config:
                data["rag"] = self._rag_config.model_dump()
            self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"Failed to save config: {e}")

    def _load_config(self):
        """Load persisted configs from disk."""
        if not self.CONFIG_PATH.exists():
            return
        try:
            data = json.loads(self.CONFIG_PATH.read_text())
            if "llm" in data:
                self._llm_config = LLMProviderConfig(**data["llm"])
            if "embedding" in data:
                self._embedding_config = EmbeddingConfig(**data["embedding"])
            if "rag" in data:
                self._rag_config = RAGConfig(**data["rag"])
            logger.info(f"Loaded config from {self.CONFIG_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    async def initialize(
        self,
        llm_config: LLMProviderConfig,
        embedding_config: EmbeddingConfig,
        rag_config: Optional[RAGConfig] = None,
    ):
        """Initialize or reinitialize RAGAnything with new config."""
        if self._initializing:
            raise RuntimeError("Initialization already in progress")
        self._initializing = True

        try:
            # Cleanup existing instance
            if self._rag is not None:
                await self._cleanup()

            # Create model functions from config
            llm_func = create_llm_func(llm_config)
            vision_func = create_vision_func(llm_config)
            embedding_func = create_embedding_func(embedding_config)

            # Build RAGAnything config
            rag_kwargs = {}
            if rag_config:
                for field_name in [
                    "working_dir", "parser", "parse_method", "mineru_backend",
                    "enable_image_processing", "enable_table_processing",
                    "enable_equation_processing", "max_concurrent_files",
                    "context_window", "context_mode", "max_context_tokens",
                ]:
                    val = getattr(rag_config, field_name, None)
                    if val is not None:
                        rag_kwargs[field_name] = val

            config = RAGAnythingConfig(**rag_kwargs)

            # Create RAGAnything instance
            self._rag = RAGAnything(
                config=config,
                llm_model_func=llm_func,
                vision_model_func=vision_func,
                embedding_func=embedding_func,
            )

            self._llm_config = llm_config
            self._embedding_config = embedding_config
            self._rag_config = rag_config

            self._save_config()
            logger.info("RAGAnything initialized successfully")

        finally:
            self._initializing = False

    async def update_rag_config(self, rag_config: RAGConfig):
        """Update RAG config. Stores config; reinitializes if LLM is already configured."""
        self._rag_config = rag_config

        if self._rag is not None and self._llm_config is not None:
            emb = self._embedding_config or EmbeddingConfig()
            await self.initialize(self._llm_config, emb, rag_config)
        else:
            self._save_config()

    async def update_llm_config(
        self,
        llm_config: LLMProviderConfig,
        embedding_config: Optional[EmbeddingConfig] = None,
    ):
        """Update LLM config and reinitialize."""
        emb = embedding_config or self._embedding_config or EmbeddingConfig()

        await self.initialize(llm_config, emb, self._rag_config)

    async def _cleanup(self):
        """Cleanup the current RAGAnything instance."""
        if self._rag is not None:
            try:
                await self._rag.finalize_storages()
            except Exception as e:
                logger.warning(f"Error during RAG cleanup: {e}")
            self._rag = None

    async def shutdown(self):
        """Shutdown and cleanup."""
        await self._cleanup()
        logger.info("RAGService shut down")


# Global singleton
rag_service = RAGService()
