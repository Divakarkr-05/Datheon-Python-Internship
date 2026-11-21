"""
Document chunking and splitting strategies using RecursiveCharacterTextSplitter.
Preserves metadata (source, page) and enforces chunking parameter validation.
"""

from typing import List, Optional
import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def validate_chunking_parameters(chunk_size: int, chunk_overlap: int) -> None:
    """
    Validate chunk size and overlap boundaries.

    Raises:
        ValueError: If constraints are violated.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )


def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Optional[List[str]] = None,
) -> List[Document]:
    """
    Split documents into smaller, overlapping chunks while preserving metadata.

    Args:
        documents: List of input LangChain Document instances.
        chunk_size: Maximum characters per chunk (default: 1000).
        chunk_overlap: Overlapping character count (default: 200).
        separators: Custom hierarchy of separators (default: None, uses standard).

    Returns:
        List of chunked Document instances with preserved metadata.
    """
    validate_chunking_parameters(chunk_size, chunk_overlap)

    if not documents:
        return []

    valid_docs = [doc for doc in documents if doc.page_content and doc.page_content.strip()]
    if not valid_docs:
        return []

    splitter_kwargs = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "length_function": len,
        "is_separator_regex": False,
    }
    if separators:
        splitter_kwargs["separators"] = separators

    text_splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)
    chunks = text_splitter.split_documents(valid_docs)

    for idx, chunk in enumerate(chunks):
        chunk.metadata = dict(chunk.metadata or {})
        chunk.metadata["chunk_index"] = idx
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = 1
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = "unknown"

    logger.info(f"Split {len(valid_docs)} document(s) into {len(chunks)} chunk(s).")
    return chunks
