"""
Local HuggingFace embedding model management using sentence-transformers.
Avoids top-level model downloads during module import with lazy initialization.
"""

from typing import Optional, Dict, Any
import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_CACHED_EMBEDDINGS: Dict[str, HuggingFaceEmbeddings] = {}


def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
    normalize_embeddings: bool = True,
    model_kwargs: Optional[Dict[str, Any]] = None,
    encode_kwargs: Optional[Dict[str, Any]] = None,
) -> HuggingFaceEmbeddings:
    """
    Initialize and return a reusable HuggingFaceEmbeddings instance.
    Runs locally on CPU by default and lazily caches model instances.

    Args:
        model_name: HuggingFace model identifier (default: all-MiniLM-L6-v2).
        device: Target execution device ('cpu' or 'cuda').
        normalize_embeddings: Whether to normalize vector outputs.
        model_kwargs: Optional model config overrides.
        encode_kwargs: Optional encoding config overrides.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    cache_key = f"{model_name}_{device}_{normalize_embeddings}"

    if cache_key in _CACHED_EMBEDDINGS:
        return _CACHED_EMBEDDINGS[cache_key]

    resolved_model_kwargs = {"device": device}
    if model_kwargs:
        resolved_model_kwargs.update(model_kwargs)

    resolved_encode_kwargs = {"normalize_embeddings": normalize_embeddings}
    if encode_kwargs:
        resolved_encode_kwargs.update(encode_kwargs)

    logger.info(f"Initializing local embeddings: {model_name} on device='{device}'")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=resolved_model_kwargs,
        encode_kwargs=resolved_encode_kwargs,
    )

    _CACHED_EMBEDDINGS[cache_key] = embeddings
    return embeddings
