"""Tests for LLM factory module."""

import pytest
from raganything.api.schemas import LLMProviderConfig, EmbeddingConfig
from raganything.api.services.llm_factory import (
    create_llm_func,
    create_vision_func,
    create_embedding_func,
    PROVIDER_MAP,
)


class TestLLMFactory:
    def test_supported_providers(self):
        assert "openai" in PROVIDER_MAP
        assert "ollama" in PROVIDER_MAP

    def test_unsupported_provider_raises(self):
        config = LLMProviderConfig(provider="nonexistent", model="test")
        with pytest.raises(ValueError, match="Unsupported"):
            create_llm_func(config)

    def test_create_llm_func_returns_callable(self):
        config = LLMProviderConfig(provider="openai", model="gpt-4o-mini", api_key="test-key")
        func = create_llm_func(config)
        assert callable(func)

    def test_create_vision_func_returns_none_without_model(self):
        config = LLMProviderConfig(provider="openai", model="gpt-4o-mini", vision_model=None)
        assert create_vision_func(config) is None

    def test_create_vision_func_returns_callable(self):
        config = LLMProviderConfig(provider="openai", model="gpt-4o-mini", vision_model="gpt-4o", api_key="test")
        func = create_vision_func(config)
        assert callable(func)

    def test_create_embedding_func_returns_embedding_func(self):
        config = EmbeddingConfig(provider="openai", model="text-embedding-3-large", api_key="test")
        func = create_embedding_func(config)
        assert func is not None
