"""Factory for creating LLM/Embedding callables from config objects.

RAGAnything expects callable functions (llm_model_func, vision_model_func, embedding_func).
This module bridges config objects -> callables, supporting multiple providers.
"""

from typing import Optional
from functools import partial

from raganything.api.schemas import LLMProviderConfig, EmbeddingConfig


# Provider -> (module_path, complete_func_name, embed_func_name)
PROVIDER_MAP = {
    "openai": ("lightrag.llm.openai", "openai_complete_if_cache", "openai_embed"),
    "ollama": ("lightrag.llm.ollama", "ollama_complete_if_cache", "ollama_embed"),
    "azure_openai": ("lightrag.llm.openai", "openai_complete_if_cache", "openai_embed"),
    "lmstudio": ("lightrag.llm.lmstudio", "lmstudio_complete", "lmstudio_embed"),
    "vllm": ("lightrag.llm.vllm", "vllm_complete", "vllm_embed"),
}


def create_llm_func(config: LLMProviderConfig):
    """Create an llm_model_func callable from config.

    Returns an async function with signature:
        (prompt, system_prompt=None, history_messages=[], **kwargs) -> str
    """
    if config.provider not in PROVIDER_MAP:
        raise ValueError(
            f"Unsupported LLM provider: {config.provider}. "
            f"Supported: {list(PROVIDER_MAP.keys())}"
        )

    module_path, func_name, _ = PROVIDER_MAP[config.provider]
    import importlib
    module = importlib.import_module(module_path)
    complete_func = getattr(module, func_name)

    kwargs = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    if config.timeout is not None:
        kwargs["timeout"] = config.timeout

    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **extra):
        merged = {**kwargs, **extra}
        return await complete_func(
            config.model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **merged,
        )

    return llm_model_func


def create_vision_func(config: LLMProviderConfig):
    """Create a vision_model_func callable from config.

    Returns an async function with signature:
        (prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs) -> str
    """
    if not config.vision_model:
        return None

    if config.provider not in PROVIDER_MAP:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")

    module_path, func_name, _ = PROVIDER_MAP[config.provider]
    import importlib
    module = importlib.import_module(module_path)
    complete_func = getattr(module, func_name)

    kwargs = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url

    async def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **extra
    ):
        merged = {**kwargs, **extra}
        if messages:
            return await complete_func(
                config.vision_model, "", system_prompt=None,
                history_messages=[], messages=messages, **merged,
            )
        elif image_data:
            msg = []
            if system_prompt:
                msg.append({"role": "system", "content": system_prompt})
            msg.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                ],
            })
            return await complete_func(
                config.vision_model, "", system_prompt=None,
                history_messages=[], messages=msg, **merged,
            )
        else:
            return await complete_func(
                config.vision_model, prompt,
                system_prompt=system_prompt, history_messages=history_messages, **merged,
            )

    return vision_model_func


def create_local_embedding_func(config: EmbeddingConfig):
    """Create an EmbeddingFunc from a local sentence-transformers model.

    Loads model from config.model_path (or config.model as HuggingFace model ID).
    Supports BGE, E5, GTE, and other sentence-transformers compatible models.
    """
    from lightrag.utils import EmbeddingFunc
    import numpy as np

    model_id = config.model_path or config.model

    # Lazy-load model on first call
    _model = None

    def _get_model():
        nonlocal _model
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(model_id)
        return _model

    async def local_embed(texts: list[str]) -> np.ndarray:
        model = _get_model()
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(embeddings)

    return EmbeddingFunc(
        embedding_dim=config.dimension,
        max_token_size=8192,
        func=local_embed,
    )


def create_embedding_func(config: EmbeddingConfig):
    """Create an EmbeddingFunc from config.

    Returns an EmbeddingFunc instance for use with RAGAnything/LightRAG.
    """
    if config.provider == "local":
        return create_local_embedding_func(config)

    if config.provider not in PROVIDER_MAP:
        raise ValueError(f"Unsupported embedding provider: {config.provider}")

    from lightrag.utils import EmbeddingFunc

    module_path, _, embed_func_name = PROVIDER_MAP[config.provider]
    import importlib
    module = importlib.import_module(module_path)
    embed_func = getattr(module, embed_func_name)

    kwargs = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url

    return EmbeddingFunc(
        embedding_dim=config.dimension,
        max_token_size=8192,
        func=partial(embed_func.func, model=config.model, **kwargs),
    )


async def test_llm_connection(config: LLMProviderConfig) -> dict:
    """Test LLM connection by sending a simple prompt.

    Returns dict with keys: success (bool), message (str), latency_ms (float).
    """
    import time

    try:
        func = create_llm_func(config)
        start = time.time()
        result = await func("Say 'hello' in one word.", max_tokens=512)
        latency = (time.time() - start) * 1000
        return {"success": True, "message": result.strip(), "latency_ms": round(latency, 1)}
    except Exception as e:
        return {"success": False, "message": str(e), "latency_ms": 0}
